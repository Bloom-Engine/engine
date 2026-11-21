//! The joint palette in pages (`renderer/joint_pages.rs`): more than 1024
//! joints in a frame, each model posed with its own matrices, and a frame
//! past the cap that drops the models that do not fit instead of posing
//! them with another model's matrices.
//!
//! Self-checking, no reference image: every instance is one small square
//! moved to its own grid cell by the LAST joint of its palette (the others
//! point off screen), tinted red, green or blue by its index. A square that
//! read any other slot of any page is missing from its cell or has the wrong
//! colour.

#[allow(unused_imports)]
use super::*;

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const CAMERA_DISTANCE: f32 = 10.0;
const FOV_DEGREES: f32 = 48.0;

/// What a cell of the picture shows.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Cell {
    Empty,
    Tint(usize),
    Other,
}

struct Grid {
    columns: usize,
    rows: usize,
    /// Half the visible extent at the squares' depth.
    half: f32,
}

impl Grid {
    fn new(columns: usize, rows: usize) -> Self {
        Self {
            columns,
            rows,
            half: CAMERA_DISTANCE * (FOV_DEGREES.to_radians() * 0.5).tan(),
        }
    }
    fn pitch(&self) -> f32 {
        // Leave a margin of half a cell around the grid.
        2.0 * self.half / (self.columns.max(self.rows) as f32 + 1.0)
    }
    fn world(&self, index: usize) -> [f32; 3] {
        let column = (index % self.columns) as f32;
        let row = (index / self.columns) as f32;
        let pitch = self.pitch();
        [
            (column - (self.columns as f32 - 1.0) * 0.5) * pitch,
            ((self.rows as f32 - 1.0) * 0.5 - row) * pitch,
            0.0,
        ]
    }
    fn pixel(&self, index: usize, width: u32, height: u32) -> (u32, u32) {
        let [x, y, _] = self.world(index);
        let u = 0.5 + 0.5 * x / self.half;
        let v = 0.5 - 0.5 * y / self.half;
        (
            (u * width as f32).floor() as u32,
            (v * height as f32).floor() as u32,
        )
    }
}

fn tint_of(index: usize) -> [f32; 4] {
    match index % 3 {
        0 => [1.0, 0.0, 0.0, 1.0],
        1 => [0.0, 1.0, 0.0, 1.0],
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

fn classify(rgba: &[u8], width: u32, x: u32, y: u32) -> Cell {
    let at = ((y * width + x) * 4) as usize;
    let px = [rgba[at] as i32, rgba[at + 1] as i32, rgba[at + 2] as i32];
    let max = *px.iter().max().unwrap();
    let min = *px.iter().min().unwrap();
    if max - min < 24 {
        return Cell::Empty; // the grey clear colour
    }
    let channel = px.iter().position(|&value| value == max).unwrap();
    let others = (0..3)
        .filter(|&other| other != channel)
        .map(|other| px[other])
        .max()
        .unwrap();
    if max >= 60 && max >= others * 2 {
        Cell::Tint(channel)
    } else {
        Cell::Other
    }
}

/// Cache a square whose four vertices follow joint `joints - 1` only.
fn cache_square(eng: &mut EngineState, handle: u64, joints: usize, half_size: f32) {
    let last = (joints - 1) as f32;
    let vertex = |x: f32, y: f32| Vertex3D {
        position: [x * half_size, y * half_size, 0.0],
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv: [0.0; 2],
        joints: [last, 0.0, 0.0, 0.0],
        weights: [1.0, 0.0, 0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    assert!(eng.renderer.cache_model_if_static(
        handle,
        &[MeshData {
            vertices: vec![
                vertex(-1.0, -1.0),
                vertex(1.0, -1.0),
                vertex(1.0, 1.0),
                vertex(-1.0, 1.0),
            ],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
            morph: None,
        }]
    ));
    assert!(eng.renderer.is_model_skinned(handle));
}

/// Draw `instances` squares, each with a palette of `joints` matrices, and
/// return what each cell of the picture shows.
fn draw_grid(
    eng: &mut EngineState,
    handle: u64,
    joints: usize,
    instances: usize,
    grid: &Grid,
    label: &str,
) -> Vec<Cell> {
    // Every joint but the last throws the square far off screen.
    let mut palette = vec![IDENTITY; joints];
    for matrix in palette.iter_mut().take(joints - 1) {
        matrix[3] = [500.0, 500.0, 0.0, 1.0];
    }
    let (width, height, rgba) = render(eng, 3, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(40.0, 40.0, 40.0, 255.0);
        r.begin_mode_3d(
            0.0,
            0.0,
            CAMERA_DISTANCE,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            FOV_DEGREES,
            0.0,
        );
        r.add_directional_light(0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 3.0);
        for index in 0..instances {
            r.set_joint_matrices_scaled(
                0x10_0000 + index as u64,
                &palette,
                1.0,
                grid.world(index),
                0.0,
                1.0,
            );
            r.draw_model_cached_skinned(handle, [0.0; 3], 1.0, tint_of(index));
        }
    });
    if let Some(directory) = std::env::var_os("BLOOM_KEEP_JOINT_PAGES") {
        let path = Path::new(&directory).join(format!("joint-pages-{label}.png"));
        image::save_buffer(&path, &rgba, width, height, image::ColorType::Rgba8).unwrap();
        eprintln!("kept {path:?}");
    }
    (0..grid.columns * grid.rows)
        .map(|index| {
            let (x, y) = grid.pixel(index, width, height);
            classify(&rgba, width, x, y)
        })
        .collect()
}

#[test]
fn more_than_1024_joints_in_a_frame_pose_every_model_with_its_own_matrices() {
    const HANDLE: u64 = 0x701A_0001;
    const JOINTS: usize = 23; // a shooter creature
    const INSTANCES: usize = 100; // 2300 joints: three pages
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let grid = Grid::new(10, 10);
    cache_square(&mut eng, HANDLE, JOINTS, grid.pitch() * 0.3);

    // A frame within 1024 joints stays on the one original page.
    let few = draw_grid(&mut eng, HANDLE, JOINTS, 44, &grid, "few");
    assert_eq!(eng.renderer.joint_page_count(), 1);
    for (index, cell) in few.iter().enumerate() {
        let expected = if index < 44 {
            Cell::Tint(index % 3)
        } else {
            Cell::Empty
        };
        assert_eq!(*cell, expected, "44 models: cell {index}");
    }

    // 44 models fill 1012 joints; the 45th starts the second page. Before
    // paging it and every later one were posed with the first model's pose.
    let many = draw_grid(&mut eng, HANDLE, JOINTS, INSTANCES, &grid, "many");
    assert_eq!(eng.renderer.joint_page_count(), 3);
    for (index, cell) in many.iter().enumerate() {
        assert_eq!(*cell, Cell::Tint(index % 3), "100 models: cell {index}");
    }

    // And back: the extra pages stay allocated but nothing reads them.
    let again = draw_grid(&mut eng, HANDLE, JOINTS, 44, &grid, "few-again");
    assert_eq!(again, few);
}

#[test]
fn past_the_joint_cap_the_models_that_do_not_fit_are_not_drawn() {
    const HANDLE: u64 = 0x701A_0002;
    const JOINTS: usize = 512; // two models to a page, 32 to the cap
    const FIT: usize = 32;
    const INSTANCES: usize = 36;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let grid = Grid::new(6, 6);
    cache_square(&mut eng, HANDLE, JOINTS, grid.pitch() * 0.3);
    let cells = draw_grid(&mut eng, HANDLE, JOINTS, INSTANCES, &grid, "cap");
    assert_eq!(eng.renderer.joint_page_count(), 16);
    for (index, cell) in cells.iter().enumerate() {
        let expected = if index < FIT {
            Cell::Tint(index % 3)
        } else {
            // Dropped with a warning: not drawn anywhere, and in particular
            // not on top of another model's cell.
            Cell::Empty
        };
        assert_eq!(*cell, expected, "cap: cell {index}");
    }
}
