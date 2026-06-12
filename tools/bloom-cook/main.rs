//! bloom-cook — offline asset cooking for the Bloom engine.
//!
//! Today: texture cooking. PNG/JPEG/BMP/TGA → BC7-compressed DDS with a
//! full precomputed mip chain. The wins, in order: 4x less VRAM than
//! RGBA8 on devices whose adapter exposes BC texture support (all
//! desktops), much faster loads (no PNG inflate, no runtime mip
//! generation), and precomputed mips identical across machines. Disk
//! size versus PNG varies with content (BC7 is a fixed 1 byte/px +
//! mips; PNG entropy-codes), so cook for runtime wins, not disk.
//! Devices without BC decode cooked files on the CPU at load — one
//! cooked artifact ships everywhere.
//!
//! The engine loads cooked .dds transparently through the same
//! loadTexture() path as raw images (magic-sniffed).
//!
//! Usage:
//!   bloom-cook texture <in.(png|jpg|bmp|tga)> <out.dds> [--normal] [--linear]
//!   bloom-cook texture-dir <in-dir> <out-dir> [--linear]
//!
//! --normal  treat as a normal map (linear color, BC7)
//! --linear  non-color data (masks, LUTs): skip the sRGB transfer
use std::path::{Path, PathBuf};
use std::process::ExitCode;

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
        _ => {
            eprintln!("usage: bloom-cook texture <in> <out.dds> [--normal] [--linear]");
            eprintln!("       bloom-cook texture-dir <in-dir> <out-dir> [--linear]");
            ExitCode::FAILURE
        }
    }
}

fn cook_texture(input: &Path, output: &Path, flags: &[&str]) -> Result<String, String> {
    let linear = flags.contains(&"--linear") || flags.contains(&"--normal");
    let src_len = std::fs::metadata(input).map_err(|e| format!("{input:?}: {e}"))?.len();
    let img = image::open(input)
        .map_err(|e| format!("{input:?}: {e}"))?
        .to_rgba8();

    // sRGB for color data, linear for normal maps / masks. BC7 keeps full
    // RGBA; normal maps could use BC5 later but BC7's quality is high
    // enough that one format keeps the pipeline simple.
    let format = if linear {
        image_dds::ImageFormat::BC7RgbaUnorm
    } else {
        image_dds::ImageFormat::BC7RgbaUnormSrgb
    };
    let dds = image_dds::dds_from_image(
        &img,
        format,
        image_dds::Quality::Normal,
        image_dds::Mipmaps::GeneratedAutomatic,
    )
    .map_err(|e| format!("encode {input:?}: {e}"))?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{parent:?}: {e}"))?;
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(output).map_err(|e| format!("{output:?}: {e}"))?,
    );
    dds.write(&mut out).map_err(|e| format!("write {output:?}: {e}"))?;
    drop(out);
    let dst_len = std::fs::metadata(output).map_err(|e| e.to_string())?.len();
    Ok(format!(
        "{} -> {} ({} KB -> {} KB, {}x{}, {} mips)",
        input.display(),
        output.display(),
        src_len / 1024,
        dst_len / 1024,
        img.width(),
        img.height(),
        dds.get_num_mipmap_levels(),
    ))
}

fn cook_dir(in_dir: &Path, out_dir: &Path, flags: &[&str]) -> Result<usize, String> {
    let mut count = 0;
    for entry in std::fs::read_dir(in_dir).map_err(|e| format!("{in_dir:?}: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga") {
            continue;
        }
        let mut out: PathBuf = out_dir.join(path.file_name().unwrap());
        out.set_extension("dds");
        println!("{}", cook_texture(&path, &out, flags)?);
        count += 1;
    }
    Ok(count)
}
