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

/// Returns the raw numeric handle. Prefer this on Android (aarch64): Perry 0.5.x
/// miscompiles `music.handle` field reads that feed straight into an f64 FFI slot,
/// producing NaN and dropping the music.
export function loadMusicRaw(path: string): number {
  return bloom_load_music(path as any);
}

export function playMusic(music: Music): void {
  bloom_play_music(music.handle);
}

/**
 * @internal Compiler workaround — not part of the public API.
 * Identical to the non-Raw version but takes primitives instead of
 * reading object fields (aarch64 Android Perry miscompilation where
 * obj.field reads feeding f64 FFI args arrive as NaN). Use the
 * non-Raw version; these disappear when the Perry fix ships.
 */
export function playMusicRaw(handle: number): void {
  bloom_play_music(handle);
}

export function stopMusic(music: Music): void {
  bloom_stop_music(music.handle);
}

/**
 * @internal Compiler workaround — not part of the public API.
 * Identical to the non-Raw version but takes primitives instead of
 * reading object fields (aarch64 Android Perry miscompilation where
 * obj.field reads feeding f64 FFI args arrive as NaN). Use the
 * non-Raw version; these disappear when the Perry fix ships.
 */
export function stopMusicRaw(handle: number): void {
  bloom_stop_music(handle);
}

export function updateMusicStream(music: Music): void {
  bloom_update_music_stream(music.handle);
}

/**
 * @internal Compiler workaround — not part of the public API.
 * Identical to the non-Raw version but takes primitives instead of
 * reading object fields (aarch64 Android Perry miscompilation where
 * obj.field reads feeding f64 FFI args arrive as NaN). Use the
 * non-Raw version; these disappear when the Perry fix ships.
 */
export function updateMusicStreamRaw(handle: number): void {
  bloom_update_music_stream(handle);
}

// Spec-compliant alias
export function updateMusic(music: Music): void { bloom_update_music_stream(music.handle); }

export function setMusicVolume(music: Music, volume: number): void {
  bloom_set_music_volume(music.handle, volume);
}

/**
 * @internal Compiler workaround — not part of the public API.
 * Identical to the non-Raw version but takes primitives instead of
 * reading object fields (aarch64 Android Perry miscompilation where
 * obj.field reads feeding f64 FFI args arrive as NaN). Use the
 * non-Raw version; these disappear when the Perry fix ships.
 */
export function setMusicVolumeRaw(handle: number, volume: number): void {
  bloom_set_music_volume(handle, volume);
}

export function isMusicPlaying(music: Music): boolean {
  return bloom_is_music_playing(music.handle) !== 0;
}

/**
 * @internal Compiler workaround — not part of the public API.
 * Identical to the non-Raw version but takes primitives instead of
 * reading object fields (aarch64 Android Perry miscompilation where
 * obj.field reads feeding f64 FFI args arrive as NaN). Use the
 * non-Raw version; these disappear when the Perry fix ships.
 */
export function isMusicPlayingRaw(handle: number): boolean {
  return bloom_is_music_playing(handle) !== 0;
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

// ---- EN-029: mix buses, reverb send, occlusion low-pass ---------------------
//
// The mixer used to be master gain + per-voice gain. That plays sounds; it
// does not make a space sound like a place. Three additions cover the moves a
// shooter actually needs:
//
//   - a bus you can duck   ("drop the music when the player is hit")
//   - a tail you can send to ("this gunshot is indoors")
//   - a filter per source  ("that shriek is behind the building")
//
// Routing is per *sound*, set once at load — a footstep is always SFX, a menu
// blip is always UI — so the per-shot call sites stay unchanged.

declare function bloom_set_sound_bus(handle: number, bus: number): void;
declare function bloom_set_sound_reverb_send(handle: number, send: number): void;
declare function bloom_set_sound_lowpass(handle: number, cutoff: number): void;
declare function bloom_set_bus_gain(bus: number, gain: number): void;
declare function bloom_duck_bus(bus: number, amount: number, attack: number, release: number, hold: number): void;
declare function bloom_set_reverb(size: number, damp: number, wet: number): void;

export const BUS_SFX = 0;
export const BUS_MUSIC = 1;
export const BUS_UI = 2;

/// Route a sound to a mix bus. Music loaded via `loadMusic` is already on
/// BUS_MUSIC; this is for sound effects that belong somewhere other than SFX
/// (menu clicks → BUS_UI, so they never duck with the rest of the mix).
export function setSoundBus(sound: Sound, bus: number): void {
  bloom_set_sound_bus(sound.handle, bus);
}

/// How much of this sound feeds the reverb, 0..1. This is what gives a weapon
/// its tail — and raising it near walls is what makes a fight "indoors".
export function setSoundReverbSend(sound: Sound, send: number): void {
  bloom_set_sound_reverb_send(sound.handle, send);
}

/// Low-pass this sound at `cutoffHz` (0 = bypass). The occlusion primitive:
/// the game raycasts to the emitter and muffles what it can't see. Muffling
/// reads as geometry in a way that simply lowering the volume never does.
export function setSoundLowpass(sound: Sound, cutoffHz: number): void {
  bloom_set_sound_lowpass(sound.handle, cutoffHz);
}

export function setBusGain(bus: number, gain: number): void {
  bloom_set_bus_gain(bus, gain);
}

/// Momentarily pull a bus down — `amount` 0..1 — over `attack` seconds, hold
/// it for `hold`, then recover over `release`. Call it again to re-trigger;
/// the hold restarts, so repeated hits keep the music down.
export function duckBus(bus: number, amount: number, attack: number, release: number, hold: number): void {
  bloom_duck_bus(bus, amount, attack, release, hold);
}

/// Global reverb: `size` 0..1 (tail length), `damp` 0..1 (HF absorption),
/// `wet` 0..1 (how much returns to the mix). wet = 0 bypasses the entire
/// reverb path, so it is free until you ask for it — ramp it up as the player
/// moves indoors.
export function setReverb(size: number, damp: number, wet: number): void {
  bloom_set_reverb(size, damp, wet);
}
