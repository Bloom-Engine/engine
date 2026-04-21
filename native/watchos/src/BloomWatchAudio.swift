// BloomWatchAudio — AVAudioPlayer-backed implementation of bloom's audio FFI.
//
// The Rust side of bloom-watchos calls into these C-exported functions via
// extern "C" so sound/music paths stay string-based at the Perry/Rust boundary
// but decoding, decoding, and playback happen on the Swift side where we
// have AVAudioPlayer (and, on watchOS, no obvious Rust-native audio path).
//
// Handles are 1-based to match the rest of bloom-watchos's texture/font
// registries — 0 signals "none / not loaded".

import Foundation
import AVFoundation

// MARK: - Registry

final class BloomAudioManager {
    static let shared = BloomAudioManager()

    private var sounds: [AVAudioPlayer] = []
    private var musics: [AVAudioPlayer] = []
    private var masterVolume: Float = 1.0
    private let lock = NSLock()
    private var sessionStarted = false

    func initAudio() {
        guard !sessionStarted else { return }
        sessionStarted = true
        // AVAudioSession on watchOS requires explicit activation. Ambient
        // category = play alongside other audio without interruption; good
        // default for games.
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.ambient, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true, options: [])
    }

    func closeAudio() {
        lock.lock()
        sounds.forEach { $0.stop() }
        musics.forEach { $0.stop() }
        sounds.removeAll()
        musics.removeAll()
        lock.unlock()
    }

    /// Load a sound from a filesystem path. Returns a 1-based handle or 0 on
    /// failure. The path is expected to be already bundle-resolved on the
    /// Rust side (textures::resolve_bundle_path).
    func loadSound(path: String) -> UInt32 {
        guard let player = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: path)) else {
            return 0
        }
        player.prepareToPlay()
        lock.lock()
        sounds.append(player)
        let handle = UInt32(sounds.count)
        lock.unlock()
        return handle
    }

    func playSound(_ handle: UInt32) {
        guard let p = sound(handle) else { return }
        p.volume = masterVolume
        p.currentTime = 0
        p.play()
    }

    func stopSound(_ handle: UInt32) {
        sound(handle)?.stop()
    }

    func setSoundVolume(_ handle: UInt32, _ v: Float) {
        sound(handle)?.volume = v * masterVolume
    }

    func setMasterVolume(_ v: Float) {
        lock.lock()
        masterVolume = v
        sounds.forEach { $0.volume = v }
        musics.forEach { $0.volume = v }
        lock.unlock()
    }

    func loadMusic(path: String) -> UInt32 {
        guard let player = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: path)) else {
            return 0
        }
        player.numberOfLoops = -1  // loop forever
        player.prepareToPlay()
        lock.lock()
        musics.append(player)
        let handle = UInt32(musics.count)
        lock.unlock()
        return handle
    }

    func playMusic(_ handle: UInt32) {
        guard let p = music(handle) else { return }
        p.volume = masterVolume
        p.play()
    }

    func stopMusic(_ handle: UInt32) { music(handle)?.stop() }
    func setMusicVolume(_ handle: UInt32, _ v: Float) { music(handle)?.volume = v * masterVolume }
    func isMusicPlaying(_ handle: UInt32) -> Bool { music(handle)?.isPlaying ?? false }

    private func sound(_ handle: UInt32) -> AVAudioPlayer? {
        lock.lock(); defer { lock.unlock() }
        let i = Int(handle)
        return (i >= 1 && i <= sounds.count) ? sounds[i - 1] : nil
    }

    private func music(_ handle: UInt32) -> AVAudioPlayer? {
        lock.lock(); defer { lock.unlock() }
        let i = Int(handle)
        return (i >= 1 && i <= musics.count) ? musics[i - 1] : nil
    }
}

// MARK: - C-ABI exports (called from bloom-watchos Rust crate)

@_cdecl("bloom_watchos_audio_init")
public func bloom_watchos_audio_init() {
    BloomAudioManager.shared.initAudio()
}

@_cdecl("bloom_watchos_audio_close")
public func bloom_watchos_audio_close() {
    BloomAudioManager.shared.closeAudio()
}

@_cdecl("bloom_watchos_sound_load")
public func bloom_watchos_sound_load(_ path: UnsafePointer<CChar>) -> UInt32 {
    BloomAudioManager.shared.loadSound(path: String(cString: path))
}

@_cdecl("bloom_watchos_sound_play")
public func bloom_watchos_sound_play(_ handle: UInt32) {
    BloomAudioManager.shared.playSound(handle)
}

@_cdecl("bloom_watchos_sound_stop")
public func bloom_watchos_sound_stop(_ handle: UInt32) {
    BloomAudioManager.shared.stopSound(handle)
}

@_cdecl("bloom_watchos_sound_volume")
public func bloom_watchos_sound_volume(_ handle: UInt32, _ v: Float) {
    BloomAudioManager.shared.setSoundVolume(handle, v)
}

@_cdecl("bloom_watchos_master_volume")
public func bloom_watchos_master_volume(_ v: Float) {
    BloomAudioManager.shared.setMasterVolume(v)
}

@_cdecl("bloom_watchos_music_load")
public func bloom_watchos_music_load(_ path: UnsafePointer<CChar>) -> UInt32 {
    BloomAudioManager.shared.loadMusic(path: String(cString: path))
}

@_cdecl("bloom_watchos_music_play")
public func bloom_watchos_music_play(_ handle: UInt32) {
    BloomAudioManager.shared.playMusic(handle)
}

@_cdecl("bloom_watchos_music_stop")
public func bloom_watchos_music_stop(_ handle: UInt32) {
    BloomAudioManager.shared.stopMusic(handle)
}

@_cdecl("bloom_watchos_music_volume")
public func bloom_watchos_music_volume(_ handle: UInt32, _ v: Float) {
    BloomAudioManager.shared.setMusicVolume(handle, v)
}

@_cdecl("bloom_watchos_music_is_playing")
public func bloom_watchos_music_is_playing(_ handle: UInt32) -> UInt32 {
    BloomAudioManager.shared.isMusicPlaying(handle) ? 1 : 0
}
