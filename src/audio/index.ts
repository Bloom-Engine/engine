import { Sound, Music } from '../core/types';

// FFI declarations
declare function bloom_init_audio(): void;
declare function bloom_close_audio(): void;
declare function bloom_load_sound(path: number): number;
declare function bloom_play_sound(handle: number): void;
declare function bloom_stop_sound(handle: number): void;
declare function bloom_set_sound_volume(handle: number, volume: number): void;
declare function bloom_set_master_volume(volume: number): void;
declare function bloom_load_music(path: number): number;
declare function bloom_play_music(handle: number): void;
declare function bloom_stop_music(handle: number): void;
declare function bloom_update_music_stream(handle: number): void;
declare function bloom_set_music_volume(handle: number, volume: number): void;
declare function bloom_is_music_playing(handle: number): number;

export function initAudio(): void {
  bloom_init_audio();
}

export function closeAudio(): void {
  bloom_close_audio();
}

export function loadSound(path: string): Sound {
  const handle = bloom_load_sound(path as any);
  return { handle };
}

export function playSound(sound: Sound): void {
  bloom_play_sound(sound.handle);
}

export function stopSound(sound: Sound): void {
  bloom_stop_sound(sound.handle);
}

export function setSoundVolume(sound: Sound, volume: number): void {
  bloom_set_sound_volume(sound.handle, volume);
}

export function setMasterVolume(volume: number): void {
  bloom_set_master_volume(volume);
}

// Music functions

export function loadMusic(path: string): Music {
  const handle = bloom_load_music(path as any);
  return { handle };
}

export function playMusic(music: Music): void {
  bloom_play_music(music.handle);
}

export function stopMusic(music: Music): void {
  bloom_stop_music(music.handle);
}

export function updateMusicStream(music: Music): void {
  bloom_update_music_stream(music.handle);
}

export function setMusicVolume(music: Music, volume: number): void {
  bloom_set_music_volume(music.handle, volume);
}

export function isMusicPlaying(music: Music): boolean {
  return bloom_is_music_playing(music.handle) !== 0;
}
