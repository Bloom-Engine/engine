//! Build script for bloom-shared.
//!
//! Only does work when the `jolt` feature is enabled. In that case it builds
//! the `bloom_jolt` C++ shim (and JoltPhysics, via its own CMakeLists) via the
//! `cmake` crate and emits link directives so rustc picks up both archives.

fn main() {
    // Always re-run when these change.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_JOLT");

    if std::env::var_os("CARGO_FEATURE_JOLT").is_none() {
        return;
    }

    // The Jolt C++ shim cannot target WebAssembly via the normal cmake path —
    // Emscripten + Jolt is a separate build pipeline (see JoltPhysics.js).
    // On wasm32 the web crate routes bloom_physics_* FFI to JoltPhysics.js
    // directly via wasm_bindgen; no static library is needed here.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        return;
    }

    build_jolt();
}

#[cfg(not(feature = "jolt"))]
fn build_jolt() {}

#[cfg(feature = "jolt")]
fn build_jolt() {
    use std::path::PathBuf;

    // Locate native/third_party/bloom_jolt relative to this crate.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shim_dir = manifest_dir
        .parent()
        .unwrap()
        .join("third_party")
        .join("bloom_jolt");

    if !shim_dir.join("CMakeLists.txt").exists() {
        panic!(
            "bloom_jolt shim not found at {}; did the third_party submodules init?",
            shim_dir.display()
        );
    }

    println!("cargo:rerun-if-changed={}", shim_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", shim_dir.join("include/bloom_jolt.h").display());
    println!("cargo:rerun-if-changed={}", shim_dir.join("src/bloom_jolt.cpp").display());

    let dst = cmake::Config::new(&shim_dir)
        .profile("Release")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=bloom_jolt");
    println!("cargo:rustc-link-lib=static=Jolt");

    // C++ standard library — required because we're linking static archives
    // that pull in libc++ / libstdc++ symbols.
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target.as_str() {
        "macos" | "ios" | "tvos" | "watchos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
        }
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        "windows" => {
            // MSVC toolchain handles libc++ automatically; MinGW would need stdc++.
        }
        _ => {}
    }
}
