#[path = "../sheen_lut.rs"]
mod sheen_lut;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut output = None;
    let mut size = sheen_lut::DEFAULT_LUT_SIZE;
    let mut samples = sheen_lut::DEFAULT_SAMPLE_COUNT;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--out" => output = args.next().map(PathBuf::from),
            "--size" => {
                size = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0)
            }
            "--samples" => {
                samples = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0)
            }
            _ => {
                eprintln!("unknown argument: {argument}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(output) = output else {
        eprintln!("usage: generate_sheen_lut --out FILE [--size 128] [--samples 4096]");
        return ExitCode::from(2);
    };
    if size < 2 || samples == 0 {
        eprintln!("size must be at least 2 and samples must be non-zero");
        return ExitCode::from(2);
    }
    let values = sheen_lut::build_r16f_lut(size, samples);
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    if let Some(parent) = output.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("create {}: {error}", parent.display());
            return ExitCode::FAILURE;
        }
    }
    if let Err(error) = std::fs::write(&output, bytes) {
        eprintln!("write {}: {error}", output.display());
        return ExitCode::FAILURE;
    }
    println!(
        "wrote {}x{} R16F sheen directional-albedo LUT ({} spp) to {}",
        size,
        size,
        samples,
        output.display()
    );
    ExitCode::SUCCESS
}
