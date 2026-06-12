//! Audio file decoding: WAV, OGG Vorbis, and (behind the `mp3` feature)
//! MP3, plus the unified extension-dispatch + sniffing entry point
//! [`decode_audio`]. Pure functions over byte slices — no engine state.

use super::SoundData;

/// Parse a WAV file into SoundData.
pub fn parse_wav(data: &[u8]) -> Option<SoundData> {
    if data.len() < 44 { return None; }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" { return None; }

    let channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    let mut offset = 36;
    while offset + 8 < data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let pcm_data = &data[offset + 8..std::cmp::min(offset + 8 + chunk_size, data.len())];
            let samples = match bits_per_sample {
                16 => pcm_data.chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                    .collect(),
                8 => pcm_data.iter()
                    .map(|&b| (b as f32 - 128.0) / 128.0)
                    .collect(),
                _ => return None,
            };
            return Some(SoundData { samples, sample_rate, channels });
        }
        offset += 8 + chunk_size;
    }
    None
}

/// Parse an MP3 file into SoundData.
#[cfg(feature = "mp3")]
pub fn parse_mp3(data: &[u8]) -> Option<SoundData> {
    let mut decoder = minimp3::Decoder::new(std::io::Cursor::new(data));
    let mut samples = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0u16;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                if sample_rate == 0 {
                    sample_rate = frame.sample_rate as u32;
                    channels = frame.channels as u16;
                }
                for &s in &frame.data {
                    samples.push(s as f32 / 32768.0);
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(_) => return None,
        }
    }

    if samples.is_empty() { return None; }
    Some(SoundData { samples, sample_rate, channels })
}

/// Parse an OGG Vorbis file into SoundData.
pub fn parse_ogg(data: &[u8]) -> Option<SoundData> {
    let cursor = std::io::Cursor::new(data);
    let mut reader = lewton::inside_ogg::OggStreamReader::new(cursor).ok()?;
    let sample_rate = reader.ident_hdr.audio_sample_rate;
    let channels = reader.ident_hdr.audio_channels as u16;
    let mut samples = Vec::new();

    while let Ok(Some(packet)) = reader.read_dec_packet_itl() {
        for &sample in &packet {
            samples.push(sample as f32 / 32768.0);
        }
    }

    if samples.is_empty() { return None; }
    Some(SoundData { samples, sample_rate, channels })
}

/// Decode an audio file by extension, falling back to format sniffing.
///
/// Unifies the two historical platform behaviors: extension dispatch
/// (macOS/iOS/tvOS) and try-every-parser (Linux/Windows/Android). The
/// extension picks the first parser; if it fails — or the extension is
/// unrecognized — the remaining parsers run in turn, so a mislabelled
/// file still decodes.
pub fn decode_audio(path: &str, data: &[u8]) -> Option<SoundData> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".ogg") {
        if let Some(s) = parse_ogg(data) {
            return Some(s);
        }
    } else if lower.ends_with(".mp3") {
        #[cfg(feature = "mp3")]
        if let Some(s) = parse_mp3(data) {
            return Some(s);
        }
    } else if lower.ends_with(".wav") {
        if let Some(s) = parse_wav(data) {
            return Some(s);
        }
    }
    // Sniff: wav first (cheap header check), then ogg, then mp3 (no
    // reliable magic — minimp3 will chew on almost anything, keep it last).
    if let Some(s) = parse_wav(data) {
        return Some(s);
    }
    if let Some(s) = parse_ogg(data) {
        return Some(s);
    }
    #[cfg(feature = "mp3")]
    if let Some(s) = parse_mp3(data) {
        return Some(s);
    }
    None
}
