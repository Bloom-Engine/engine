//! Crash capture for every platform that is not Windows (which has its own
//! minidump filter in `native/windows`).
//!
//! Two ways a game dies, two hooks:
//!
//! - **A Rust panic.** `panic = "abort"` prints the message to stderr and
//!   aborts. A game launched from an icon has no stderr anyone will read, so
//!   the message that names the bug is lost. The panic hook here writes it,
//!   with a backtrace, to the report file first.
//! - **A fatal signal** (SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT): a bad
//!   pointer in the engine, in Jolt, or in the compiled game. The handler
//!   writes the signal, the faulting address and a backtrace, then hands the
//!   signal on (to whatever handler was there before, then to the default
//!   action), so the operating system's own crash report still happens.
//!
//! The report goes to stderr always, and to a file once the game names one
//! with `setCrashReportPath`. The game reads that file on its next launch
//! (`readFile`), shows or sends it, and clears it (`writeFile(path, '')`);
//! the engine sends nothing anywhere.
//!
//! The signal handler allocates nothing and takes no locks: the path is a
//! fixed buffer filled in beforehand, numbers are formatted on the stack, and
//! output is `write(2)`. The backtrace comes from `backtrace(3)` where the C
//! library has it (Apple platforms, glibc); elsewhere the report has the
//! signal and address only, and the system's own report has the frames.

use std::sync::atomic::{AtomicBool, Ordering};

static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Install the hooks. Safe to call more than once; only the first call acts.
pub fn install() {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    install_panic_hook();
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unix::install();
}

/// Where the next crash report is written (appended). An empty path turns the
/// file off; stderr stays on.
pub fn set_report_path(path: &str) {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unix::set_path(path);
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        *fallback_path().lock().unwrap() = path.to_string();
    }
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn fallback_path() -> &'static std::sync::Mutex<String> {
    static PATH: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
    &PATH
}

fn install_panic_hook() {
    // wasm has no file to write and the browser console already shows the
    // panic; leave its hook alone.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                String::from("(no message)")
            };
            let place = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| String::from("(unknown location)"));
            let trace = std::backtrace::Backtrace::force_capture();
            let report = format!(
                "bloom: PANIC {message}\n  at {place}\n  thread {:?}\n{trace}\n",
                std::thread::current().name().unwrap_or("(unnamed)")
            );
            append_to_report(report.as_bytes());
            #[cfg(all(unix, not(target_arch = "wasm32")))]
            unix::PANICKING.store(true, Ordering::SeqCst);
            // The default hook prints the message to stderr; abort follows.
            previous(info);
        }));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn append_to_report(bytes: &[u8]) {
    #[cfg(unix)]
    unix::append(bytes);
    #[cfg(not(unix))]
    {
        use std::io::Write;
        let path = fallback_path()
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        if !path.is_empty() {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(bytes);
            }
        }
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
mod unix {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const PATH_CAP: usize = 1024;
    /// NUL-terminated report path. Written by `set_path` (game thread, rarely),
    /// read by the handler. `PATH_LEN` is stored last, so a handler that fires
    /// mid-update sees either the old length or a complete new path.
    static mut PATH: [u8; PATH_CAP] = [0; PATH_CAP];
    static PATH_LEN: AtomicUsize = AtomicUsize::new(0);
    pub(super) static PANICKING: AtomicBool = AtomicBool::new(false);
    static IN_HANDLER: AtomicBool = AtomicBool::new(false);

    const SIGNALS: [libc::c_int; 5] = [
        libc::SIGSEGV,
        libc::SIGBUS,
        libc::SIGILL,
        libc::SIGFPE,
        libc::SIGABRT,
    ];
    /// What was installed before us, per entry of `SIGNALS`.
    static mut PREVIOUS: [libc::sigaction; 5] = unsafe { std::mem::zeroed() };

    pub(super) fn set_path(path: &str) {
        let bytes = path.as_bytes();
        let n = bytes.len().min(PATH_CAP - 1);
        PATH_LEN.store(0, Ordering::SeqCst);
        unsafe {
            let dst = std::ptr::addr_of_mut!(PATH) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, n);
            *dst.add(n) = 0;
        }
        PATH_LEN.store(n, Ordering::SeqCst);
    }

    /// Open the report file for appending, or -1.
    unsafe fn open_report() -> libc::c_int {
        if PATH_LEN.load(Ordering::SeqCst) == 0 {
            return -1;
        }
        libc::open(
            std::ptr::addr_of!(PATH) as *const libc::c_char,
            libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND,
            0o644 as libc::c_uint,
        )
    }

    unsafe fn write_all(fd: libc::c_int, bytes: &[u8]) {
        if fd < 0 {
            return;
        }
        let mut off = 0;
        while off < bytes.len() {
            let n = libc::write(
                fd,
                bytes.as_ptr().add(off) as *const libc::c_void,
                bytes.len() - off,
            );
            if n <= 0 {
                break;
            }
            off += n as usize;
        }
    }

    pub(super) fn append(bytes: &[u8]) {
        unsafe {
            let fd = open_report();
            if fd >= 0 {
                write_all(fd, bytes);
                libc::close(fd);
            }
        }
    }

    /// A line builder on the stack: no allocation, usable in a signal handler.
    struct Line {
        buf: [u8; 192],
        len: usize,
    }

    impl Line {
        fn new() -> Self {
            Self {
                buf: [0; 192],
                len: 0,
            }
        }
        fn text(&mut self, s: &str) {
            for &b in s.as_bytes() {
                if self.len < self.buf.len() {
                    self.buf[self.len] = b;
                    self.len += 1;
                }
            }
        }
        fn dec(&mut self, mut v: u64) {
            let mut digits = [0u8; 20];
            let mut n = 0;
            loop {
                digits[n] = b'0' + (v % 10) as u8;
                n += 1;
                v /= 10;
                if v == 0 {
                    break;
                }
            }
            while n > 0 {
                n -= 1;
                if self.len < self.buf.len() {
                    self.buf[self.len] = digits[n];
                    self.len += 1;
                }
            }
        }
        fn hex(&mut self, v: u64) {
            self.text("0x");
            let mut started = false;
            for shift in (0..16).rev() {
                let nibble = ((v >> (shift * 4)) & 0xf) as u8;
                if nibble != 0 || started || shift == 0 {
                    started = true;
                    let c = if nibble < 10 {
                        b'0' + nibble
                    } else {
                        b'a' + nibble - 10
                    };
                    if self.len < self.buf.len() {
                        self.buf[self.len] = c;
                        self.len += 1;
                    }
                }
            }
        }
        fn bytes(&self) -> &[u8] {
            &self.buf[..self.len]
        }
    }

    fn signal_name(sig: libc::c_int) -> &'static str {
        match sig {
            libc::SIGSEGV => "SIGSEGV",
            libc::SIGBUS => "SIGBUS",
            libc::SIGILL => "SIGILL",
            libc::SIGFPE => "SIGFPE",
            libc::SIGABRT => "SIGABRT",
            _ => "signal",
        }
    }

    #[cfg(any(target_vendor = "apple", all(target_os = "linux", target_env = "gnu")))]
    unsafe fn write_backtrace(fd: libc::c_int) {
        extern "C" {
            fn backtrace(buffer: *mut *mut libc::c_void, size: libc::c_int) -> libc::c_int;
            fn backtrace_symbols_fd(
                buffer: *const *mut libc::c_void,
                size: libc::c_int,
                fd: libc::c_int,
            );
        }
        if fd < 0 {
            return;
        }
        let mut frames = [std::ptr::null_mut::<libc::c_void>(); 64];
        let n = backtrace(frames.as_mut_ptr(), frames.len() as libc::c_int);
        if n > 0 {
            backtrace_symbols_fd(frames.as_ptr(), n, fd);
        }
    }

    #[cfg(not(any(target_vendor = "apple", all(target_os = "linux", target_env = "gnu"))))]
    unsafe fn write_backtrace(fd: libc::c_int) {
        write_all(
            fd,
            b"  (no backtrace on this C library; see the system crash report)\n",
        );
    }

    extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
        unsafe {
            // A fault inside the handler: do not loop, let it die.
            if !IN_HANDLER.swap(true, Ordering::SeqCst) {
                let mut line = Line::new();
                line.text("bloom: FATAL ");
                line.text(signal_name(sig));
                line.text(" (signal ");
                line.dec(sig as u64);
                line.text(")");
                if !info.is_null() && (sig == libc::SIGSEGV || sig == libc::SIGBUS) {
                    line.text(" at address ");
                    line.hex((*info).si_addr() as usize as u64);
                }
                line.text(", pid ");
                line.dec(libc::getpid() as u64);
                line.text("\n");

                let file = open_report();
                // After a Rust panic the report already holds the message and
                // a symbolised trace; the abort that follows adds one line.
                let after_panic = sig == libc::SIGABRT && PANICKING.load(Ordering::SeqCst);
                for fd in [2, file] {
                    write_all(fd, line.bytes());
                    if !after_panic {
                        write_backtrace(fd);
                    }
                }
                if file >= 0 {
                    libc::close(file);
                }
            }

            // Hand the signal on: first to whoever was here before us, then to
            // the default action, so the system's crash report still happens.
            let index = SIGNALS.iter().position(|&s| s == sig);
            if let Some(i) = index {
                let previous = std::ptr::addr_of!(PREVIOUS[i]).read();
                let action = previous.sa_sigaction;
                if action != libc::SIG_DFL && action != libc::SIG_IGN {
                    if previous.sa_flags & libc::SA_SIGINFO != 0 {
                        let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                            std::mem::transmute(action);
                        f(sig, info, ctx);
                    } else {
                        let f: extern "C" fn(libc::c_int) = std::mem::transmute(action);
                        f(sig);
                    }
                }
            }
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        }
    }

    pub(super) fn install() {
        unsafe {
            // A stack of its own for the handler, so a stack overflow (which
            // arrives as SIGSEGV with no stack left) can still be reported.
            const ALT_STACK: usize = 64 * 1024;
            let stack = libc::malloc(ALT_STACK);
            if !stack.is_null() {
                let alt = libc::stack_t {
                    ss_sp: stack,
                    ss_flags: 0,
                    ss_size: ALT_STACK,
                };
                libc::sigaltstack(&alt, std::ptr::null_mut());
            }

            for (i, &sig) in SIGNALS.iter().enumerate() {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = handler as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(sig, &action, std::ptr::addr_of_mut!(PREVIOUS[i]));
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Line;

        #[test]
        fn line_formats_without_allocating() {
            let mut l = Line::new();
            l.text("sig ");
            l.dec(11);
            l.text(" at ");
            l.hex(0xdead_beef);
            l.text(" and ");
            l.hex(0);
            l.dec(0);
            assert_eq!(l.bytes(), b"sig 11 at 0xdeadbeef and 0x00");
        }

        #[test]
        fn line_truncates_instead_of_overflowing() {
            let mut l = Line::new();
            for _ in 0..100 {
                l.text("0123456789");
            }
            assert_eq!(l.bytes().len(), 192);
        }
    }
}
