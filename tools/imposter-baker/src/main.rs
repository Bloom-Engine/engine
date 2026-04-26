//! imposter-baker — bake an octahedral imposter atlas from a GLB.
//!
//! V2 follow-up to EN-015: V1 expected games to use Blender's
//! ScreenSpace add-on or Unity's Tree Creator. This tool ships a
//! first-party bake. The runtime side already lives in
//! `native/shared/shaders/common/imposter.wgsl`; the atlas layout and
//! cell selection here mirror that exactly.

mod gltf_load;
mod octahedral;
mod render;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "imposter-baker",
    about = "Bake octahedral imposter atlases (color/normal/depth) for Bloom from a GLB."
)]
struct Cli {
    /// Input GLB.
    input: PathBuf,

    /// Output color PNG (default: <input>.imposter.png).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Grid size N (atlas is NxN cells, total N² views).
    #[arg(long, default_value_t = 8)]
    grid: u32,

    /// Pixels per atlas cell.
    #[arg(long, default_value_t = 256)]
    cell: u32,

    /// Also write a normal-encoded atlas (RGBA8, oct-encoded view-space normal in xy).
    #[arg(long)]
    normal: Option<PathBuf>,

    /// Also write a depth atlas (linear view-space depth, normalized to model radius).
    #[arg(long)]
    depth: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.grid == 0 || cli.cell == 0 {
        return Err("grid and cell must be > 0".into());
    }

    let output = cli.output.clone().unwrap_or_else(|| {
        let mut p = cli.input.clone();
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
        p.set_file_name(format!("{stem}.imposter.png"));
        p
    });

    println!("[imposter-baker] loading {}", cli.input.display());
    let mesh = gltf_load::load_glb(&cli.input)?;
    println!(
        "[imposter-baker] mesh: {} verts, {} indices, AABB [{:.3?} → {:.3?}], radius {:.3}",
        mesh.positions.len(),
        mesh.indices.len(),
        mesh.aabb_min,
        mesh.aabb_max,
        mesh.radius()
    );

    let atlas_px = cli.grid * cli.cell;
    println!(
        "[imposter-baker] baking {0}x{0} grid @ {1}px/cell → {2}x{2} atlas",
        cli.grid, cli.cell, atlas_px
    );

    let baked = render::bake(render::BakeOptions {
        grid: cli.grid,
        cell_px: cli.cell,
        bake_color: true,
        bake_normal: cli.normal.is_some(),
        bake_depth: cli.depth.is_some(),
        mesh: &mesh,
    })?;

    if let Some(bytes) = &baked.color {
        save_rgba(bytes, baked.width, baked.height, &output)?;
        println!("[imposter-baker] color → {}", output.display());
    }
    if let (Some(bytes), Some(path)) = (&baked.normal, &cli.normal) {
        save_rgba(bytes, baked.width, baked.height, path)?;
        println!("[imposter-baker] normal → {}", path.display());
    }
    if let (Some(bytes), Some(path)) = (&baked.depth, &cli.depth) {
        save_r8(bytes, baked.width, baked.height, path)?;
        println!("[imposter-baker] depth → {}", path.display());
    }

    Ok(())
}

fn save_rgba(bytes: &[u8], w: u32, h: u32, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let img: image::RgbaImage = image::ImageBuffer::from_raw(w, h, bytes.to_vec())
        .ok_or("size mismatch packing rgba png")?;
    img.save(path)?;
    Ok(())
}

fn save_r8(bytes: &[u8], w: u32, h: u32, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let img: image::GrayImage = image::ImageBuffer::from_raw(w, h, bytes.to_vec())
        .ok_or("size mismatch packing r8 png")?;
    img.save(path)?;
    Ok(())
}
