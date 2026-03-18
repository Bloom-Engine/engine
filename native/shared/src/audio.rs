use crate::handles::HandleRegistry;

/// A loaded sound effect (PCM samples).
pub struct SoundData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// A currently playing sound instance.
struct PlayingSound {
    data_handle: f64,
    position: usize,
    volume: f32,
    playing: bool,
    // Spatial audio
    spatial: bool,
    source_x: f32,
    source_y: f32,
    source_z: f32,
}

/// Music data (fully decoded for simplicity).
pub struct MusicData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub position: usize,
    pub playing: bool,
    pub volume: f32,
    pub looping: bool,
}

/// Platform-agnostic audio mixer.
pub struct AudioMixer {
    pub sounds: HandleRegistry<SoundData>,
    playing: Vec<PlayingSound>,
    pub master_volume: f32,
    sound_volumes: Vec<(f64, f32)>,
    pub music: HandleRegistry<MusicData>,
    // Spatial audio listener
    pub listener_x: f32,
    pub listener_y: f32,
    pub listener_z: f32,
    pub listener_forward_x: f32,
    pub listener_forward_y: f32,
    pub listener_forward_z: f32,
}

impl AudioMixer {
    pub fn new() -> Self {
        Self {
            sounds: HandleRegistry::new(),
            playing: Vec::new(),
            master_volume: 1.0,
            sound_volumes: Vec::new(),
            music: HandleRegistry::new(),
            listener_x: 0.0,
            listener_y: 0.0,
            listener_z: 0.0,
            listener_forward_x: 0.0,
            listener_forward_y: 0.0,
            listener_forward_z: -1.0,
        }
    }

    pub fn load_sound(&mut self, data: SoundData) -> f64 {
        self.sounds.alloc(data)
    }

    pub fn play_sound(&mut self, handle: f64) {
        self.playing.push(PlayingSound {
            data_handle: handle,
            position: 0,
            volume: self.get_sound_volume(handle),
            playing: true,
            spatial: false,
            source_x: 0.0, source_y: 0.0, source_z: 0.0,
        });
    }

    pub fn play_sound_3d(&mut self, handle: f64, x: f32, y: f32, z: f32) {
        self.playing.push(PlayingSound {
            data_handle: handle,
            position: 0,
            volume: self.get_sound_volume(handle),
            playing: true,
            spatial: true,
            source_x: x, source_y: y, source_z: z,
        });
    }

    pub fn set_listener_position(&mut self, x: f32, y: f32, z: f32, fx: f32, fy: f32, fz: f32) {
        self.listener_x = x;
        self.listener_y = y;
        self.listener_z = z;
        let len = (fx*fx + fy*fy + fz*fz).sqrt();
        if len > 0.0 {
            self.listener_forward_x = fx / len;
            self.listener_forward_y = fy / len;
            self.listener_forward_z = fz / len;
        }
    }

    pub fn stop_sound(&mut self, handle: f64) {
        self.playing.retain(|p| p.data_handle != handle);
    }

    pub fn set_sound_volume(&mut self, handle: f64, volume: f32) {
        for entry in &mut self.sound_volumes {
            if entry.0 == handle {
                entry.1 = volume;
                for p in &mut self.playing {
                    if p.data_handle == handle { p.volume = volume; }
                }
                return;
            }
        }
        self.sound_volumes.push((handle, volume));
    }

    fn get_sound_volume(&self, handle: f64) -> f32 {
        for entry in &self.sound_volumes {
            if entry.0 == handle { return entry.1; }
        }
        1.0
    }

    // Music functions

    pub fn load_music(&mut self, data: SoundData) -> f64 {
        self.music.alloc(MusicData {
            samples: data.samples,
            sample_rate: data.sample_rate,
            channels: data.channels,
            position: 0,
            playing: false,
            volume: 1.0,
            looping: true,
        })
    }

    pub fn play_music(&mut self, handle: f64) {
        if let Some(m) = self.music.get_mut(handle) {
            m.position = 0;
            m.playing = true;
        }
    }

    pub fn stop_music(&mut self, handle: f64) {
        if let Some(m) = self.music.get_mut(handle) {
            m.playing = false;
            m.position = 0;
        }
    }

    pub fn set_music_volume(&mut self, handle: f64, volume: f32) {
        if let Some(m) = self.music.get_mut(handle) {
            m.volume = volume;
        }
    }

    pub fn is_music_playing(&self, handle: f64) -> bool {
        self.music.get(handle).map(|m| m.playing).unwrap_or(false)
    }

    pub fn update_music_stream(&mut self, _handle: f64) {
        // No-op: we decode everything upfront. This exists for API compatibility.
    }

    /// Mix all playing sounds and music into the output buffer.
    pub fn mix_output(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = 0.0;
        }

        // Spatial audio: compute listener-relative parameters once
        let lx = self.listener_x;
        let ly = self.listener_y;
        let lz = self.listener_z;
        let lfx = self.listener_forward_x;
        let lfy = self.listener_forward_y;
        let lfz = self.listener_forward_z;
        // Listener right vector (cross product of forward and up=[0,1,0])
        let lrx = lfz;
        let lrz = -lfx;
        let lr_len = (lrx * lrx + lrz * lrz).sqrt().max(0.001);

        // Mix sound effects
        self.playing.retain_mut(|p| {
            if !p.playing { return false; }
            let sound = match self.sounds.get(p.data_handle) {
                Some(s) => s,
                None => return false,
            };

            // Compute spatial gain and pan
            let (gain_l, gain_r) = if p.spatial {
                let dx = p.source_x - lx;
                let dy = p.source_y - ly;
                let dz = p.source_z - lz;
                let dist = (dx*dx + dy*dy + dz*dz).sqrt().max(0.1);
                // Distance attenuation: 1/distance, clamped
                let attenuation = (1.0 / dist).min(1.0);
                // Pan: dot product of source direction with listener right
                let pan = ((dx * lrx + dz * lrz) / (dist * lr_len)).clamp(-1.0, 1.0);
                let left = attenuation * (1.0 - pan) * 0.5;
                let right = attenuation * (1.0 + pan) * 0.5;
                (left, right)
            } else {
                (1.0, 1.0)
            };

            let base_vol = p.volume * self.master_volume;
            let vol_l = base_vol * gain_l;
            let vol_r = base_vol * gain_r;
            let mut i = 0;
            while i < output.len() && p.position < sound.samples.len() {
                if sound.channels == 1 {
                    let sample = sound.samples[p.position];
                    output[i] += sample * vol_l;
                    if i + 1 < output.len() { output[i + 1] += sample * vol_r; }
                    p.position += 1;
                    i += 2;
                } else {
                    // For stereo sources, apply gain to each channel
                    output[i] += sound.samples[p.position] * vol_l;
                    p.position += 1;
                    if i + 1 < output.len() && p.position < sound.samples.len() {
                        output[i + 1] += sound.samples[p.position] * vol_r;
                        p.position += 1;
                    }
                    i += 2;
                }
            }
            p.position < sound.samples.len()
        });

        // Mix music (iterate all music handles)
        // We need to collect handles first to avoid borrow issues
        let handles: Vec<f64> = {
            let mut h = Vec::new();
            // HandleRegistry stores items as Vec<Option<T>>, handles are 1-based
            for i in 0..100 {
                let handle = (i + 1) as f64;
                if self.music.get(handle).is_some() {
                    h.push(handle);
                } else if i > 10 && h.is_empty() {
                    break;
                }
            }
            h
        };

        for handle in handles {
            if let Some(m) = self.music.get_mut(handle) {
                if !m.playing { continue; }
                let vol = m.volume * self.master_volume;
                let mut i = 0;
                while i < output.len() && m.position < m.samples.len() {
                    if m.channels == 1 {
                        let sample = m.samples[m.position] * vol;
                        output[i] += sample;
                        if i + 1 < output.len() { output[i + 1] += sample; }
                        m.position += 1;
                        i += 2;
                    } else {
                        output[i] += m.samples[m.position] * vol;
                        m.position += 1;
                        i += 1;
                    }
                }
                if m.position >= m.samples.len() {
                    if m.looping {
                        m.position = 0;
                    } else {
                        m.playing = false;
                    }
                }
            }
        }
    }
}

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
