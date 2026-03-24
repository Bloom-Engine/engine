import { spawn, parallelMap } from 'perry/thread';
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
declare function bloom_play_sound_3d(handle: number, x: number, y: number, z: number): void;
declare function bloom_set_listener_position(x: number, y: number, z: number, fx: number, fy: number, fz: number): void;

export function initAudio(): void {
  bloom_init_audio();
}

export function closeAudio(): void {
  bloom_close_audio();
}

// Spec-compliant aliases
export function initAudioDevice(): void { bloom_init_audio(); }
export function closeAudioDevice(): void { bloom_close_audio(); }

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

// Spec-compliant alias
export function updateMusic(music: Music): void { bloom_update_music_stream(music.handle); }

export function setMusicVolume(music: Music, volume: number): void {
  bloom_set_music_volume(music.handle, volume);
}

export function isMusicPlaying(music: Music): boolean {
  return bloom_is_music_playing(music.handle) !== 0;
}

// Spatial audio

export function playSound3D(sound: Sound, x: number, y: number, z: number): void {
  bloom_play_sound_3d(sound.handle, x, y, z);
}

export function setListenerPosition(x: number, y: number, z: number, forwardX: number, forwardY: number, forwardZ: number): void {
  bloom_set_listener_position(x, y, z, forwardX, forwardY, forwardZ);
}

// Async / threaded loading

declare function bloom_stage_sound(path: number): number;
declare function bloom_commit_sound(handle: number): number;
declare function bloom_commit_music(handle: number): number;

export async function loadSoundAsync(path: string): Promise<Sound> {
  const stagingHandle = await spawn(() => bloom_stage_sound(path as any));
  const handle = bloom_commit_sound(stagingHandle);
  return { handle };
}

export async function loadMusicAsync(path: string): Promise<Music> {
  const stagingHandle = await spawn(() => bloom_stage_sound(path as any));
  const handle = bloom_commit_music(stagingHandle);
  return { handle };
}

export function stageSounds(paths: string[]): number[] {
  return parallelMap(paths, (path: string) => bloom_stage_sound(path as any));
}

export function commitSound(stagingHandle: number): Sound {
  const handle = bloom_commit_sound(stagingHandle);
  return { handle };
}

export function commitMusic(stagingHandle: number): Music {
  const handle = bloom_commit_music(stagingHandle);
  return { handle };
}
