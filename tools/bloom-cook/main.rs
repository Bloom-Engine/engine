//! bloom-cook — offline asset cooking for the Bloom engine.
//!
//! Today: texture cooking. PNG/JPEG/BMP/TGA → DDS with a full precomputed
//! mip chain. Native desktop color/data profiles use BC7 for 4x less VRAM on
//! BC-capable adapters; the portable profile uses capability-neutral RGBA8.
//! Normal maps remain RGBA8 so their vector-aware direction and variance mips
//! are not damaged by a color-error BC7 objective. Cooked DDS also avoids
//! source-image inflate and runtime mip generation. Disk size varies with
//! content, so cook for runtime memory, quality, and load time.
//!
//! The engine loads cooked .dds transparently through the same
//! loadTexture() path as raw images (magic-sniffed).
//!
//! Usage:
//!   bloom-cook texture <in.(png|jpg|bmp|tga)> <out.dds> [--normal] [--linear]
//!   bloom-cook texture-dir <in-dir> <out-dir> [--linear]
//!   bloom-cook texture-store <logical-id> <in> <store> [profile] [texture flags]
//!   bloom-cook texture-benchmark <in> [texture flags] [--iterations N]
//!   bloom-cook geometry <in.(glb|gltf)> <out.bgeo> [geometry limits]
//!   bloom-cook geometry-inspect <in.bgeo>
//!   bloom-cook geometry-store <logical-id> <in.(glb|gltf)> <store> [profile] [limits]
//!   bloom-cook geometry-load-benchmark <in.glb|gltf> <in.bgeo> [--iterations N]
//!   bloom-cook asset-inspect <logical-id> <store> [profile]
//!   bloom-cook asset-index <store>
//!   bloom-cook asset-index-inspect <store>
//!   bloom-cook asset-resolve <logical-id> <store> --platform ID --quality ID [fallbacks]
//!
//! --normal  treat as a normal map (linear RGBA8, vector/variance mips)
//! --linear  non-color data (masks, LUTs): skip the sRGB transfer
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod asset_benchmark;
mod asset_index;
mod asset_profile;
mod asset_resolver;
mod asset_store;
mod geometric_error;
mod geometry_cook;
mod geometry_format;
mod geometry_quantization;
mod hierarchy;
mod meshlet;
mod texture_cook;
mod texture_store;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("texture") if args.len() >= 3 => {
            let flags: Vec<&str> = args[3..].iter().map(String::as_str).collect();
            match cook_texture(Path::new(&args[1]), Path::new(&args[2]), &flags) {
                Ok(stats) => {
                    println!("{}", stats);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("bloom-cook: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("texture-dir") if args.len() >= 3 => {
            let flags: Vec<&str> = args[3..].iter().map(String::as_str).collect();
            match cook_dir(Path::new(&args[1]), Path::new(&args[2]), &flags) {
                Ok(n) => {
                    println!("cooked {n} textures");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("bloom-cook: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("texture-store") if args.len() >= 4 => {
            match texture_store::store_texture_command(
                &args[1],
                Path::new(&args[2]),
                Path::new(&args[3]),
                &args[4..],
            ) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("texture-benchmark") if args.len() >= 2 => {
            match asset_benchmark::benchmark_texture_command(Path::new(&args[1]), &args[2..]) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("geometry") if args.len() >= 3 => {
            match geometry_cook::cook_geometry_command(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &args[3..],
            ) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("geometry-inspect") if args.len() == 2 => {
            match geometry_cook::inspect_geometry_command(Path::new(&args[1])) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("geometry-store") if args.len() >= 4 => {
            match asset_store::store_geometry_command(
                &args[1],
                Path::new(&args[2]),
                Path::new(&args[3]),
                &args[4..],
            ) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("geometry-load-benchmark") if args.len() >= 3 => {
            match asset_benchmark::benchmark_geometry_command(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &args[3..],
            ) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("asset-inspect") if args.len() >= 3 => {
            match asset_store::inspect_asset_command(&args[1], Path::new(&args[2]), &args[3..]) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("asset-index") if args.len() == 2 => {
            match asset_index::build_asset_index_command(Path::new(&args[1])) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("asset-index-inspect") if args.len() == 2 => {
            match asset_index::inspect_asset_index_command(Path::new(&args[1])) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("asset-resolve") if args.len() >= 3 => {
            match asset_resolver::resolve_asset_command(&args[1], Path::new(&args[2]), &args[3..]) {
                Ok(report) => {
                    println!("{report}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("bloom-cook: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("usage: bloom-cook texture <in> <out.dds> [--normal] [--linear]");
            eprintln!("       bloom-cook texture-dir <in-dir> <out-dir> [--linear]");
            eprintln!(
                "       bloom-cook texture-store <logical-id> <in> <store-dir> \
                 [--platform ID --quality ID] [--normal] [--linear]"
            );
            eprintln!(
                "       bloom-cook texture-benchmark <in> [--normal] [--linear] \
                 [--iterations N]"
            );
            eprintln!(
                "       bloom-cook geometry <in.glb|gltf> <out.bgeo> \
                 [--max-vertices N] [--max-triangles N] [--page-bytes N] \
                 [--hierarchy-levels N] [--vertex-format float32|quantized32]"
            );
            eprintln!("       bloom-cook geometry-inspect <in.bgeo>");
            eprintln!(
                "       bloom-cook geometry-store <logical-id> <in.glb|gltf> <store-dir> \
                 [--platform ID --quality ID] [geometry limits]"
            );
            eprintln!(
                "       bloom-cook geometry-load-benchmark <in.glb|gltf> <in.bgeo> \
                 [--iterations N]"
            );
            eprintln!(
                "       bloom-cook asset-inspect <logical-id> <store-dir> \
                 [--platform ID --quality ID]"
            );
            eprintln!("       bloom-cook asset-index <store-dir>");
            eprintln!("       bloom-cook asset-index-inspect <store-dir>");
            eprintln!(
                "       bloom-cook asset-resolve <logical-id> <store-dir> \
                 --platform ID --quality ID [--fallback PLATFORM/QUALITY] \
                 [--allow-unprofiled]"
            );
            ExitCode::FAILURE
        }
    }
}

fn cook_texture(input: &Path, output: &Path, flags: &[&str]) -> Result<String, String> {
    let settings = texture_cook::TextureSettings::parse(flags.iter().copied())?;
    let prepared = texture_cook::PreparedTexture::read(input, settings)?;
    let source_bytes = prepared.source_bytes.len();
    let cooked =
        texture_cook::cook_prepared_texture(input, &prepared, settings.artifact_format(None))?;
    geometry_cook::write_atomically(output, &cooked.bytes)?;
    Ok(format!(
        "{} -> {} ({} KB -> {} KB, {}x{}, {} mips)",
        input.display(),
        output.display(),
        source_bytes / 1024,
        cooked.bytes.len() / 1024,
        cooked.width,
        cooked.height,
        cooked.mip_levels,
    ))
}

fn cook_dir(in_dir: &Path, out_dir: &Path, flags: &[&str]) -> Result<usize, String> {
    let mut count = 0;
    for entry in std::fs::read_dir(in_dir).map_err(|e| format!("{in_dir:?}: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(
            ext.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "bmp" | "tga"
        ) {
            continue;
        }
        let mut out: PathBuf = out_dir.join(path.file_name().unwrap());
        out.set_extension("dds");
        println!("{}", cook_texture(&path, &out, flags)?);
        count += 1;
    }
    Ok(count)
}
