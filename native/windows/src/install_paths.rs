//! Where a shipped game finds its files and where it may write them.
//!
//! Games address everything with paths relative to the working directory:
//! `assets/...` to read, `settings.json` to save. That holds in a checkout and
//! in an unpacked zip started from Explorer, but not for an installed game:
//!
//! * A shortcut, the Start menu or an MSIX launch may start the process in
//!   another directory. When the working directory has no `assets` folder and
//!   the executable's folder does, we change to the executable's folder.
//! * An MSIX package (Microsoft Store) installs read-only under
//!   `C:\Program Files\WindowsApps`, and a zip unpacked under Program Files is
//!   read-only too. Then the files the game WRITES go to
//!   `%LOCALAPPDATA%\<exe name>` (inside a package Windows redirects that
//!   folder to the package's own storage, which uninstall removes).
//!
//! The redirect is decided per relative path, so the core FFI needs no notion
//! of "read" and "write": a path resolves to the save folder when the file is
//! already there, or when the install folder does not have it (a file about to
//! be created). A file the install folder ships (an asset) still reads from
//! there. Nothing changes in a writable folder, so development runs and
//! portable copies keep their files next to the executable.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static SAVE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// The folder writes are redirected to, or None when files stay next to the
/// executable. The first call also fixes the working directory.
fn save_dir() -> Option<&'static Path> {
    SAVE_DIR
        .get_or_init(|| {
            let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
            if !Path::new("assets").is_dir() && exe_dir.join("assets").is_dir() {
                let _ = std::env::set_current_dir(&exe_dir);
            }
            let cwd = std::env::current_dir().ok()?;
            if !is_packaged() && is_writable(&cwd) {
                return None;
            }
            let name = std::env::current_exe()
                .ok()?
                .file_stem()?
                .to_string_lossy()
                .into_owned();
            let dir = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join(name);
            std::fs::create_dir_all(&dir).ok()?;
            Some(dir)
        })
        .as_deref()
}

/// Called once at window creation so the working directory is fixed even if
/// the game touched no file before it.
pub fn init() {
    let _ = save_dir();
}

/// The core-FFI path hook (see `bloom_resolve_asset_path` in lib.rs).
pub fn resolve(path: &str) -> std::borrow::Cow<'_, str> {
    let Some(dir) = save_dir() else {
        return std::borrow::Cow::Borrowed(path);
    };
    redirect(dir, path)
}

fn redirect<'a>(dir: &Path, path: &'a str) -> std::borrow::Cow<'a, str> {
    let p = Path::new(path);
    if path.is_empty() || p.is_absolute() || p.has_root() {
        return std::borrow::Cow::Borrowed(path);
    }
    let saved = dir.join(p);
    if saved.exists() || !p.exists() {
        if let Some(parent) = saved.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        return std::borrow::Cow::Owned(saved.to_string_lossy().into_owned());
    }
    std::borrow::Cow::Borrowed(path)
}

fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".bloom-write-test-{}", std::process::id()));
    match std::fs::OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// True inside an MSIX/AppX package (a Store install or a sideloaded package).
#[cfg(windows)]
fn is_packaged() -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentPackageFullName(len: *mut u32, name: *mut u16) -> i32;
    }
    const APPMODEL_ERROR_NO_PACKAGE: i32 = 15700;
    let mut len: u32 = 0;
    let rc = unsafe { GetCurrentPackageFullName(&mut len, std::ptr::null_mut()) };
    rc != APPMODEL_ERROR_NO_PACKAGE
}

#[cfg(not(windows))]
fn is_packaged() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirect_keeps_shipped_files_and_moves_new_ones() {
        let root = std::env::temp_dir().join(format!("bloom-paths-{}", std::process::id()));
        let save = root.join("save");
        std::fs::create_dir_all(&save).unwrap();
        // A file the install folder ships reads from there: Cargo.toml is in
        // the crate directory, which is the working directory under `cargo test`.
        assert_eq!(redirect(&save, "Cargo.toml"), "Cargo.toml");
        // A new file goes to the save folder, and its parent folder is made.
        let r = redirect(&save, "sub/settings.json").into_owned();
        assert_eq!(PathBuf::from(&r), save.join("sub/settings.json"));
        assert!(save.join("sub").is_dir());
        // Once saved there, the save folder wins.
        std::fs::write(&r, "{}").unwrap();
        assert_eq!(PathBuf::from(redirect(&save, "sub/settings.json").as_ref()), save.join("sub/settings.json"));
        // Absolute paths and empty paths pass through.
        let abs = root.join("x.txt").to_string_lossy().into_owned();
        assert_eq!(redirect(&save, &abs), abs.as_str());
        assert_eq!(redirect(&save, ""), "");
        let _ = std::fs::remove_dir_all(&root);
    }
}
