//! Phase 6 — material WGSL hot reload.
//!
//! When a game compiles a material via `compile_material_from_file`,
//! the source path is registered here, the parent directory is added
//! to a `notify` recursive watcher, and a worker thread funnels
//! file-changed events into a `mpsc::Receiver` that the main thread
//! drains once per frame from `Renderer::end_frame_with_scene`.
//!
//! Recompilation happens on the main thread (wgpu's `Device` is
//! `!Send` for our purposes here): each pending event re-reads the
//! file from disk, runs the same `compile_material` it originally
//! used, and replaces the entry in `MaterialSystem::pipelines` at the
//! same handle index. Existing draws keep working — they look up by
//! handle, never by `&MaterialPipeline`.
//!
//! Failures (parse errors, validation) are logged but don't kill the
//! game — the previous pipeline stays bound until the file is edited
//! again to something valid.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::material_pipeline::{Bucket, FragmentProfile};

/// What we remember about each file-backed material so we can
/// rebuild it when the file changes.
#[derive(Clone)]
pub struct FileMaterialDesc {
    pub path:             PathBuf,
    pub profile:          FragmentProfile,
    pub bucket:           Bucket,
    pub reads_scene:      bool,
    /// EN-001 — preserved across reloads so a file-backed instanced
    /// material continues to opt into the per-instance vertex layout
    /// after hot reload.
    pub wants_instancing: bool,
}

pub struct MaterialHotReload {
    /// Path → handle (1-based). Multiple handles per path possible if
    /// the same file is compiled into multiple buckets — rare, but
    /// supported via `Vec<u32>`.
    by_path: HashMap<PathBuf, Vec<u32>>,
    /// Handle (1-based, used as Vec<Option<…>> idx via -1) → desc.
    /// Indexed parallel to `MaterialSystem::pipelines`.
    descriptors: HashMap<u32, FileMaterialDesc>,
    /// Cross-thread channel from the watcher's worker into the main
    /// thread's per-frame poll. Use Mutex<Receiver> because the
    /// renderer is held behind `&mut self` from many places.
    rx: Mutex<Receiver<PathBuf>>,
    _tx: Sender<PathBuf>,
    /// Holding the watcher in the struct keeps the worker alive.
    /// `Option` because creation is fallible and we want a clean
    /// fallback (no hot reload) if `notify` errors at startup.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
    _watcher: Option<notify::RecommendedWatcher>,
    /// Last-seen change time per path — `notify` fires multiple
    /// events for one save (truncate + write + close on macOS).
    /// Coalesce within `DEBOUNCE_WINDOW`.
    last_event: Mutex<HashMap<PathBuf, Instant>>,
    /// Directories already passed to `watcher.watch` — repeatedly
    /// adding the same dir is fine but creates duplicate events.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
    watched_dirs: Mutex<Vec<PathBuf>>,
}

const DEBOUNCE_WINDOW: Duration = Duration::from_millis(120);

impl MaterialHotReload {
    pub fn new() -> Self {
        let (tx, rx) = channel::<PathBuf>();

        // Watcher is on by default — even in release builds, since
        // our daily dev cycle uses --release for perf. Shipped game
        // binaries can opt out two ways:
        //   - runtime: set `BLOOM_NO_HOT_RELOAD=1`, short-circuits the
        //     watcher thread (compile_material_from_file still works,
        //     drain_pending just always returns empty)
        //   - compile-time: build with `--no-default-features` (or
        //     without the `hot-reload` feature), which drops `notify`
        //     from the dep tree entirely (EN-008)
        #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
        let watcher = if std::env::var("BLOOM_NO_HOT_RELOAD").map(|v| v == "1").unwrap_or(false) {
            None
        } else {
            use notify::{Event, EventKind};
            let tx_clone = tx.clone();
            let cb = move |res: Result<Event, notify::Error>| {
                if let Ok(ev) = res {
                    if matches!(ev.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        for p in ev.paths {
                            let _ = tx_clone.send(p);
                        }
                    }
                }
            };
            match notify::recommended_watcher(cb) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!("[hot_reload] failed to start file watcher: {e:?}");
                    None
                }
            }
        };

        Self {
            by_path: HashMap::new(),
            descriptors: HashMap::new(),
            rx: Mutex::new(rx),
            _tx: tx,
            #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
            _watcher: watcher,
            last_event: Mutex::new(HashMap::new()),
            #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
            watched_dirs: Mutex::new(Vec::new()),
        }
    }

    /// Called by the FFI `compile_material_from_file` after a successful
    /// initial compile. Records the path → handle mapping and starts
    /// watching the file's parent directory.
    pub fn register(&mut self, handle: u32, desc: FileMaterialDesc) {
        let path = desc.path.clone();
        self.by_path.entry(path.clone()).or_default().push(handle);
        self.descriptors.insert(handle, desc);
        self.ensure_dir_watched(&path);
    }

    /// Drain pending file-change events. Returns the unique set of
    /// (handle, descriptor) pairs that need recompilation this frame.
    /// `notify` fires multiple events per save; we coalesce within a
    /// 120 ms window.
    pub fn drain_pending(&self) -> Vec<(u32, FileMaterialDesc)> {
        let mut paths: Vec<PathBuf> = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(p) = rx.try_recv() {
                paths.push(p);
            }
        }
        if paths.is_empty() { return Vec::new(); }

        let now = Instant::now();
        let mut last = self.last_event.lock().expect("hot_reload poisoned");
        let mut out: Vec<(u32, FileMaterialDesc)> = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        for raw in paths {
            // Resolve canonically so the keys we matched on register
            // line up with the events. notify on macOS sometimes
            // returns the realpath, sometimes the symlink path.
            let p = std::fs::canonicalize(&raw).unwrap_or(raw);
            if let Some(t) = last.get(&p) {
                if now.duration_since(*t) < DEBOUNCE_WINDOW { continue; }
            }
            last.insert(p.clone(), now);
            if let Some(handles) = self.by_path.get(&p) {
                for h in handles {
                    if seen.insert(*h) {
                        if let Some(d) = self.descriptors.get(h) {
                            out.push((*h, d.clone()));
                        }
                    }
                }
            }
        }
        out
    }

    /// Test-only: push a path through the channel as if `notify` had
    /// fired a Modify event. Lets unit tests exercise the drain +
    /// debounce logic without a real file system.
    #[cfg(test)]
    pub fn test_inject_event(&self, path: PathBuf) {
        let _ = self._tx.send(path);
    }

    fn ensure_dir_watched(&mut self, path: &std::path::Path) {
        #[cfg(all(not(target_arch = "wasm32"), feature = "hot-reload"))]
        {
            use notify::{RecursiveMode, Watcher};
            let dir = match path.parent() {
                Some(d) => match std::fs::canonicalize(d) {
                    Ok(c) => c,
                    Err(_) => return,
                },
                None => return,
            };
            let mut watched = self.watched_dirs.lock().expect("hot_reload poisoned");
            if watched.contains(&dir) { return; }
            if let Some(w) = self._watcher.as_mut() {
                if let Err(e) = w.watch(&dir, RecursiveMode::NonRecursive) {
                    eprintln!("[hot_reload] failed to watch {dir:?}: {e:?}");
                    return;
                }
                watched.push(dir);
            }
        }
        #[cfg(any(target_arch = "wasm32", not(feature = "hot-reload")))]
        {
            let _ = path;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::material_pipeline::{Bucket, FragmentProfile};

    fn desc(path: &str) -> FileMaterialDesc {
        FileMaterialDesc {
            path:             PathBuf::from(path),
            profile:          FragmentProfile::Translucent,
            bucket:           Bucket::Refractive,
            reads_scene:      true,
            wants_instancing: false,
        }
    }

    #[test]
    fn drain_returns_registered_handle_for_event_path() {
        let mut hr = MaterialHotReload::new();
        let p = PathBuf::from("/tmp/bloom_test_water.wgsl");
        hr.register(7, FileMaterialDesc {
            path:             p.clone(),
            profile:          FragmentProfile::Translucent,
            bucket:           Bucket::Refractive,
            reads_scene:      true,
            wants_instancing: false,
        });
        // canonicalize() in drain_pending falls back to the raw path
        // when the file doesn't exist, so this resolves to itself.
        hr.test_inject_event(p.clone());
        let pending = hr.drain_pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 7);
        assert_eq!(pending[0].1.path, p);
    }

    #[test]
    fn drain_dedups_within_debounce_window() {
        let mut hr = MaterialHotReload::new();
        let p = PathBuf::from("/tmp/bloom_test_dedup.wgsl");
        hr.register(3, desc("/tmp/bloom_test_dedup.wgsl"));
        // notify on macOS fires multiple events per save; same path
        // 5x within the 120 ms window must collapse to one drain.
        for _ in 0..5 {
            hr.test_inject_event(p.clone());
        }
        let pending = hr.drain_pending();
        assert_eq!(pending.len(), 1, "debounce should collapse 5 events to 1");
    }

    #[test]
    fn drain_returns_distinct_handles_for_distinct_paths() {
        let mut hr = MaterialHotReload::new();
        let p1 = PathBuf::from("/tmp/bloom_test_a.wgsl");
        let p2 = PathBuf::from("/tmp/bloom_test_b.wgsl");
        hr.register(1, desc("/tmp/bloom_test_a.wgsl"));
        hr.register(2, desc("/tmp/bloom_test_b.wgsl"));
        hr.test_inject_event(p1);
        hr.test_inject_event(p2);
        let pending = hr.drain_pending();
        assert_eq!(pending.len(), 2);
        let mut handles: Vec<u32> = pending.iter().map(|(h, _)| *h).collect();
        handles.sort();
        assert_eq!(handles, vec![1, 2]);
    }

    #[test]
    fn drain_ignores_unregistered_paths() {
        let hr = MaterialHotReload::new();
        hr.test_inject_event(PathBuf::from("/tmp/never_registered.wgsl"));
        let pending = hr.drain_pending();
        assert!(pending.is_empty());
    }

    #[test]
    fn empty_channel_drains_to_empty_vec() {
        let hr = MaterialHotReload::new();
        assert!(hr.drain_pending().is_empty());
    }
}
