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
    pub path:        PathBuf,
    pub profile:     FragmentProfile,
    pub bucket:      Bucket,
    pub reads_scene: bool,
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
    #[cfg(not(target_arch = "wasm32"))]
    _watcher: Option<notify::RecommendedWatcher>,
    /// Last-seen change time per path — `notify` fires multiple
    /// events for one save (truncate + write + close on macOS).
    /// Coalesce within `DEBOUNCE_WINDOW`.
    last_event: Mutex<HashMap<PathBuf, Instant>>,
    /// Directories already passed to `watcher.watch` — repeatedly
    /// adding the same dir is fine but creates duplicate events.
    watched_dirs: Mutex<Vec<PathBuf>>,
}

const DEBOUNCE_WINDOW: Duration = Duration::from_millis(120);

impl MaterialHotReload {
    pub fn new() -> Self {
        let (tx, rx) = channel::<PathBuf>();

        #[cfg(not(target_arch = "wasm32"))]
        let watcher = {
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
            #[cfg(not(target_arch = "wasm32"))]
            _watcher: watcher,
            last_event: Mutex::new(HashMap::new()),
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

    fn ensure_dir_watched(&mut self, path: &std::path::Path) {
        #[cfg(not(target_arch = "wasm32"))]
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
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
        }
    }
}
