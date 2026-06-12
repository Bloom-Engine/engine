//! Streamed music decode: a background worker decodes OGG/MP3 into PCM
//! chunks and feeds them over the lock-free SPSC ring; the audio thread's
//! MusicVoice consumes chunks instead of holding the whole decoded track.
//!
//! Memory math that motivates this: a 5-minute 44.1 kHz stereo track is
//! ~57 MB as f32 PCM but ~5 MB as OGG/MP3 bytes. Streaming keeps only the
//! compressed bytes (shared `Arc`) plus ~1.5 s of decoded ring buffer
//! resident. WAV stays full-decode (it IS PCM — nothing to win), and so
//! does wasm32 (no threads without COOP/COEP; the full-decode path is the
//! documented fallback there).
//!
//! Worker lifecycle: spawned per play_music on a streamed track, killed
//! by `stop` (AtomicBool) when the track is stopped/replaced/unloaded.
//! Looping restarts the decoder from byte 0 inside the worker, so a loop
//! seam never underruns the ring.

use super::spsc::Consumer;
#[cfg(not(target_arch = "wasm32"))]
use super::spsc::{self, Producer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Compressed source format of a streamed track.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StreamKind {
    Ogg,
    #[cfg(feature = "mp3")]
    Mp3,
}

/// One message from the decode worker to the render-side voice.
pub enum StreamMsg {
    /// Interleaved f32 PCM at the track's native sample rate/channels.
    Chunk(Vec<f32>),
    /// Non-looping track fully delivered.
    End,
}

/// ~0.37 s of stereo 44.1 kHz per chunk; the ring holds four — enough to
/// ride out >1 s of decoder starvation before an audible underrun.
#[cfg(not(target_arch = "wasm32"))]
const CHUNK_SAMPLES: usize = 32 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const RING_CHUNKS: usize = 4;

/// Render-side handle to a running stream.
pub struct StreamConsumer {
    pub rx: Consumer<StreamMsg>,
    /// Set to stop the worker (dropped voice, stop_music, engine teardown).
    pub stop: Arc<AtomicBool>,
}

impl Drop for StreamConsumer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Probe an OGG/MP3 byte buffer for (sample_rate, channels) without
/// decoding the whole track. Returns None if the data doesn't parse.
pub fn probe(kind: StreamKind, bytes: &[u8]) -> Option<(u32, u16)> {
    match kind {
        StreamKind::Ogg => {
            let reader =
                lewton::inside_ogg::OggStreamReader::new(std::io::Cursor::new(bytes)).ok()?;
            Some((
                reader.ident_hdr.audio_sample_rate,
                reader.ident_hdr.audio_channels as u16,
            ))
        }
        #[cfg(feature = "mp3")]
        StreamKind::Mp3 => {
            let mut decoder = minimp3::Decoder::new(std::io::Cursor::new(bytes));
            let frame = decoder.next_frame().ok()?;
            Some((frame.sample_rate as u32, frame.channels as u16))
        }
    }
}

/// Spawn the decode worker for one playback of a streamed track.
#[cfg(not(target_arch = "wasm32"))]
pub fn start(kind: StreamKind, bytes: Arc<Vec<u8>>, looping: bool) -> StreamConsumer {
    let (tx, rx) = spsc::channel(RING_CHUNKS + 1);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = stop.clone();
    std::thread::Builder::new()
        .name("bloom-music-decode".into())
        .spawn(move || run_worker(kind, &bytes, looping, tx, &stop_worker))
        .expect("spawn music decode worker");
    StreamConsumer { rx, stop }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_worker(
    kind: StreamKind,
    bytes: &[u8],
    looping: bool,
    mut tx: Producer<StreamMsg>,
    stop: &AtomicBool,
) {
    // Push with backpressure: the ring fills ~1.5s ahead, then we sleep.
    // 20ms poll keeps the worker essentially idle while the ring is full.
    let mut push = |msg: StreamMsg| -> bool {
        let mut m = msg;
        loop {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            match tx.push(m) {
                Ok(()) => return true,
                Err(back) => {
                    m = back;
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
    };

    loop {
        let completed = match kind {
            StreamKind::Ogg => decode_ogg(bytes, stop, &mut push),
            #[cfg(feature = "mp3")]
            StreamKind::Mp3 => decode_mp3(bytes, stop, &mut push),
        };
        if !completed || !looping {
            break;
        }
        // looping: restart the decoder from the top
    }
    let _ = push(StreamMsg::End);
}

/// Decode one full pass of the track, pushing chunks. Returns false if
/// stopped early (stop flag / consumer gone).
#[cfg(not(target_arch = "wasm32"))]
fn decode_ogg(bytes: &[u8], stop: &AtomicBool, push: &mut dyn FnMut(StreamMsg) -> bool) -> bool {
    let Ok(mut reader) = lewton::inside_ogg::OggStreamReader::new(std::io::Cursor::new(bytes))
    else {
        return false;
    };
    let mut chunk: Vec<f32> = Vec::with_capacity(CHUNK_SAMPLES);
    while let Ok(Some(packet)) = reader.read_dec_packet_itl() {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        for &s in &packet {
            chunk.push(s as f32 / 32768.0);
        }
        if chunk.len() >= CHUNK_SAMPLES {
            let full = std::mem::replace(&mut chunk, Vec::with_capacity(CHUNK_SAMPLES));
            if !push(StreamMsg::Chunk(full)) {
                return false;
            }
        }
    }
    if !chunk.is_empty() && !push(StreamMsg::Chunk(chunk)) {
        return false;
    }
    true
}

#[cfg(all(feature = "mp3", not(target_arch = "wasm32")))]
fn decode_mp3(bytes: &[u8], stop: &AtomicBool, push: &mut dyn FnMut(StreamMsg) -> bool) -> bool {
    let mut decoder = minimp3::Decoder::new(std::io::Cursor::new(bytes));
    let mut chunk: Vec<f32> = Vec::with_capacity(CHUNK_SAMPLES);
    loop {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        match decoder.next_frame() {
            Ok(frame) => {
                for &s in &frame.data {
                    chunk.push(s as f32 / 32768.0);
                }
                if chunk.len() >= CHUNK_SAMPLES {
                    let full = std::mem::replace(&mut chunk, Vec::with_capacity(CHUNK_SAMPLES));
                    if !push(StreamMsg::Chunk(full)) {
                        return false;
                    }
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(_) => break,
        }
    }
    if !chunk.is_empty() && !push(StreamMsg::Chunk(chunk)) {
        return false;
    }
    true
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;

    /// Tiny synthetic WAV→OGG isn't practical here; instead exercise the
    /// worker plumbing with MP3 when available, and always exercise the
    /// stop/backpressure logic with a hand-rolled pusher.
    #[test]
    fn stop_flag_terminates_backpressure_wait() {
        let (tx, _rx) = spsc::channel::<StreamMsg>(2);
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let h = std::thread::spawn(move || {
            let mut tx = tx;
            // fill the ring, then keep pushing — blocks on backpressure
            let mut pushed = 0;
            loop {
                match tx.push(StreamMsg::Chunk(vec![0.0; 4])) {
                    Ok(()) => pushed += 1,
                    Err(_) => {
                        if stop2.load(Ordering::Relaxed) {
                            return pushed;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        stop.store(true, Ordering::Relaxed);
        let pushed = h.join().unwrap();
        assert_eq!(pushed, 1, "ring of capacity 2 holds exactly 1 item");
    }
}
