//! The joint palette in pages.
//!
//! Every skinning shader declares the palette as one uniform block of 1024
//! matrices: 64 KiB, the largest uniform binding every backend guarantees.
//! That used to be the whole palette, so a frame could pose 1024 joints over
//! all its skinned models together, and the model that did not fit was drawn
//! with another model's matrices.
//!
//! The palette is now a list of such blocks ("pages"). Page 0 is the buffer
//! pair the renderer has always had, bound exactly as before. A frame that
//! stages more than 1024 joints gets further pages, each its own 64 KiB
//! buffer pair (this frame's matrices and the previous frame's) with its own
//! bind group of the unchanged `joint_layout`. A model's joints never
//! straddle two pages, so a skinned draw binds the page its joints are in and
//! hands the shader the offset inside that page. No shader, bind group layout
//! or pipeline changes, which is the point: the shaders are spliced by text
//! into about a hundred variants on nine backends, and a storage buffer is
//! not available on every device tier (`capabilities.rs`: the baseline tier
//! starts from WebGL2-level limits, which allow none).
//!
//! A frame that stays within 1024 joints never creates a second page and
//! takes the same code path as before. Pages are created on first use and
//! kept (128 KiB each).
//!
//! CPU side, a draw keeps one ABSOLUTE offset (`page * 1024 + offset in
//! page`): `joint_page_of` and `joint_offset_in_page` take it apart where the
//! bind group is chosen and where the shader's offset is written.

use super::*;

/// Matrices in one page: what `array<mat4x4<f32>, 1024>` in the shaders holds.
pub(super) const JOINTS_PER_PAGE: usize = 1024;

/// Pages a frame may use: 16384 joints, 2 MiB of buffers if all are in use.
pub(super) const MAX_JOINT_PAGES: usize = 16;

/// One 1024-matrix block beyond the first.
pub(super) struct JointPage {
    buffer: wgpu::Buffer,
    prev_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Where a model's joint group goes in the frame's palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JointPlacement {
    /// At this absolute offset. It is `staged` unless the group would have
    /// straddled a page boundary, in which case it is the next page's start
    /// and the matrices in between are padding.
    At(usize),
    /// More joints in one model than a page holds: it can never be bound.
    ModelTooLarge,
    /// Every page is full.
    PaletteFull,
}

/// Place a group of `group_len` matrices after `staged` already placed ones.
pub(super) fn place_joint_group(staged: usize, group_len: usize) -> JointPlacement {
    if group_len > JOINTS_PER_PAGE {
        return JointPlacement::ModelTooLarge;
    }
    let start = if staged % JOINTS_PER_PAGE + group_len > JOINTS_PER_PAGE {
        (staged / JOINTS_PER_PAGE + 1) * JOINTS_PER_PAGE
    } else {
        staged
    };
    if start + group_len > MAX_JOINT_PAGES * JOINTS_PER_PAGE {
        return JointPlacement::PaletteFull;
    }
    JointPlacement::At(start)
}

/// Page that holds the joints at this absolute offset.
#[inline]
pub(super) fn joint_page_of(joint_offset: f32) -> usize {
    joint_offset.max(0.0) as usize / JOINTS_PER_PAGE
}

/// The offset the shader adds to a vertex's joint indices: the position
/// inside the bound page. Equal to the absolute offset on page 0.
#[inline]
pub(super) fn joint_offset_in_page(joint_offset: f32) -> f32 {
    (joint_offset.max(0.0) as usize % JOINTS_PER_PAGE) as f32
}

/// The joint bind group layout: this frame's palette (binding 0), the
/// previous frame's (1), both uniform blocks, and the morph arena (2: per
/// vertex previous-minus-current morph displacement, `morph_pass.rs`), all
/// read by vertex shaders only. A shader may use any subset.
pub(super) fn create_joint_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let palette = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("joint_layout"),
        entries: &[
            palette(0),
            palette(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// A bind group of `create_joint_layout`.
pub(super) fn joint_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    palette: &wgpu::Buffer,
    prev_palette: &wgpu::Buffer,
    morph_arena: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("joint_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: palette.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: prev_palette.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(morph_arena),
            },
        ],
    })
}

const IDENTITY_JOINT: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl Renderer {
    /// Bind group (of `joint_layout`) for a page; page 0 is the original.
    #[inline]
    pub(super) fn joint_page_bind_group(&self, page: usize) -> &wgpu::BindGroup {
        match page
            .checked_sub(1)
            .and_then(|i| self.joint_extra_pages.get(i))
        {
            Some(extra) => &extra.bind_group,
            None => &self.joint_bind_group,
        }
    }

    /// This frame's matrices for a page, for the passes that build their own
    /// bind group around the palette (the compute skinning of the ray-traced
    /// path).
    #[inline]
    pub(super) fn joint_page_buffer(&self, page: usize) -> &wgpu::Buffer {
        match page
            .checked_sub(1)
            .and_then(|i| self.joint_extra_pages.get(i))
        {
            Some(extra) => &extra.buffer,
            None => &self.joint_buffer,
        }
    }

    /// Before a draw: make `group` hold the page with the joints at
    /// `joint_offset`. `bound` is the page the pass has bound there (0 after
    /// the pass set `joint_bind_group`); nothing is issued while it matches,
    /// so a frame with one page records exactly the commands it always did.
    /// Unskinned draws carry offset 0 and so put page 0 back.
    #[inline]
    pub(super) fn bind_joint_page(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        group: u32,
        joint_offset: f32,
        bound: &mut usize,
    ) {
        let page = joint_page_of(joint_offset);
        if page != *bound {
            pass.set_bind_group(group, self.joint_page_bind_group(page), &[]);
            *bound = page;
        }
    }

    /// Joint offset of an entry in a sorted draw list whose `stable_id` is an
    /// index into `model_draw_commands` (retained scene nodes are numbered
    /// after the cached draws and are never skinned: offset 0).
    #[inline]
    pub(super) fn cached_draw_joint_offset(&self, stable_id: usize) -> f32 {
        self.model_draw_commands
            .get(stable_id)
            .map_or(0.0, |command| command.joint_offset)
    }

    /// Append a staged group to the frame's palette and return its absolute
    /// offset, or `None` (after a one-time warning) when it cannot be placed.
    /// The caller must then not draw the model: there is no slot whose
    /// matrices are its own.
    pub(super) fn place_staged_joint_group(
        &mut self,
        group: &[[[f32; 4]; 4]],
        prev_group: &[[[f32; 4]; 4]],
    ) -> Option<f32> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let staged = self.frame_joint_data.len();
        match place_joint_group(staged, group.len()) {
            JointPlacement::At(start) => {
                // Padding up to a page boundary: identity, never read by a
                // draw (no group's offset points into it).
                self.frame_joint_data.resize(start, IDENTITY_JOINT);
                self.frame_joint_data_prev.resize(start, IDENTITY_JOINT);
                self.frame_joint_data.extend_from_slice(group);
                self.frame_joint_data_prev.extend_from_slice(prev_group);
                Some(start as f32)
            }
            JointPlacement::ModelTooLarge => {
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "bloom: a skinned model has {} joints; one model can have at most {}. It is not drawn. Strip joints from its rig.",
                        group.len(),
                        JOINTS_PER_PAGE
                    );
                }
                None
            }
            JointPlacement::PaletteFull => {
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "bloom: joint palette full: {} joints already staged this frame, a model with {} more does not fit in {}. It is not drawn. Draw fewer skinned models or strip joints from their rigs.",
                        staged,
                        group.len(),
                        MAX_JOINT_PAGES * JOINTS_PER_PAGE
                    );
                }
                None
            }
        }
    }

    /// Upload the frame's palette, page by page, and reset the staging state.
    pub(super) fn flush_joint_matrices(&mut self) {
        // Morph jobs of the draws recorded this frame run first: the passes
        // below draw from their buffers.
        self.encode_morph_pass();
        // Accumulator = all skinned poses staged by skinned drawModel
        // calls this frame, packed consecutively. Each draw carries the
        // offset of its group, so a flat upload per page is all the GPU needs.
        let total = self.frame_joint_data.len();
        let count = total.min(JOINTS_PER_PAGE);
        if count > 0 {
            let mut all_data = vec![[[0.0f32; 4]; 4]; JOINTS_PER_PAGE];
            all_data[..count].copy_from_slice(&self.frame_joint_data[..count]);
            self.queue
                .write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&all_data));
            // PT-7 — previous frame's palette, same offsets.
            let prev_count = self.frame_joint_data_prev.len().min(count);
            for i in 0..count {
                all_data[i] = if i < prev_count {
                    self.frame_joint_data_prev[i]
                } else {
                    self.frame_joint_data[i]
                };
            }
            self.queue
                .write_buffer(&self.joint_prev_buffer, 0, bytemuck::cast_slice(&all_data));
        }
        // Pages past the first: only in a frame that staged more than 1024
        // joints. Only the used part of a page is uploaded; the rest keeps
        // whatever an earlier frame left, and no draw's offset reaches it.
        for page in 1..total.div_ceil(JOINTS_PER_PAGE).min(MAX_JOINT_PAGES) {
            while self.joint_extra_pages.len() < page {
                let extra = self.create_joint_page();
                self.joint_extra_pages.push(extra);
            }
            let start = page * JOINTS_PER_PAGE;
            let end = total.min(start + JOINTS_PER_PAGE);
            let extra = &self.joint_extra_pages[page - 1];
            self.queue.write_buffer(
                &extra.buffer,
                0,
                bytemuck::cast_slice(&self.frame_joint_data[start..end]),
            );
            // `place_staged_joint_group` keeps the two lists the same length.
            let prev = if self.frame_joint_data_prev.len() >= end {
                &self.frame_joint_data_prev[start..end]
            } else {
                &self.frame_joint_data[start..end]
            };
            self.queue
                .write_buffer(&extra.prev_buffer, 0, bytemuck::cast_slice(prev));
        }
        self.frame_joint_data.clear();
        self.frame_joint_data_prev.clear();
        // Any leftover staged poses are stale (no draw consumed them).
        self.pending_skin_groups.clear();
        self.pending_skin_groups_prev.clear();
        self.morph.clear_pending();
    }

    fn create_joint_page(&self) -> JointPage {
        let size = (JOINTS_PER_PAGE * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let make = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let buffer = make("joint_page_buffer");
        let prev_buffer = make("joint_page_prev_buffer");
        let bind_group = joint_bind_group(
            &self.device,
            &self.joint_layout,
            &buffer,
            &prev_buffer,
            self.morph.arena_view(),
        );
        JointPage {
            buffer,
            prev_buffer,
            bind_group,
        }
    }

    /// The morph arena was remade (it grew): every joint bind group shows it.
    pub(super) fn rebuild_joint_bind_groups(&mut self) {
        self.joint_bind_group = joint_bind_group(
            &self.device,
            &self.joint_layout,
            &self.joint_buffer,
            &self.joint_prev_buffer,
            self.morph.arena_view(),
        );
        for page in &mut self.joint_extra_pages {
            page.bind_group = joint_bind_group(
                &self.device,
                &self.joint_layout,
                &page.buffer,
                &page.prev_buffer,
                self.morph.arena_view(),
            );
        }
    }

    /// Pages in existence (1 until a frame needed more). For tests and stats.
    pub fn joint_page_count(&self) -> usize {
        1 + self.joint_extra_pages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_pack_consecutively_inside_a_page() {
        assert_eq!(place_joint_group(0, 67), JointPlacement::At(0));
        assert_eq!(place_joint_group(67, 23), JointPlacement::At(67));
        // Exactly filling the page is not a straddle.
        assert_eq!(place_joint_group(1001, 23), JointPlacement::At(1001));
    }

    #[test]
    fn a_group_that_would_straddle_starts_the_next_page() {
        // The shooter's case: Trooper (67) + 41 creatures (23 each) = 1010;
        // the 42nd creature used to be the first model posed wrongly.
        assert_eq!(place_joint_group(1010, 23), JointPlacement::At(1024));
        assert_eq!(place_joint_group(1024, 23), JointPlacement::At(1024));
        assert_eq!(place_joint_group(2047, 2), JointPlacement::At(2048));
    }

    #[test]
    fn offsets_split_into_page_and_offset_in_page() {
        assert_eq!(joint_page_of(0.0), 0);
        assert_eq!(joint_page_of(1023.0), 0);
        assert_eq!(joint_page_of(1024.0), 1);
        assert_eq!(joint_offset_in_page(1023.0), 1023.0);
        assert_eq!(joint_offset_in_page(1024.0), 0.0);
        assert_eq!(joint_offset_in_page(2048.0 + 46.0), 46.0);
        // Page 0 offsets reach the shader unchanged.
        for offset in [0.0, 67.0, 757.0, 1001.0] {
            assert_eq!(joint_offset_in_page(offset), offset);
        }
    }

    #[test]
    fn every_placed_group_fits_its_page() {
        let mut staged = 0;
        for len in [67usize, 23, 1024, 1, 500, 600, 23, 1000, 30] {
            let JointPlacement::At(start) = place_joint_group(staged, len) else {
                panic!("group of {len} after {staged} was not placed");
            };
            assert!(start >= staged);
            assert!(start % JOINTS_PER_PAGE + len <= JOINTS_PER_PAGE);
            staged = start + len;
        }
    }

    #[test]
    fn past_the_cap_nothing_is_placed() {
        let cap = MAX_JOINT_PAGES * JOINTS_PER_PAGE;
        assert_eq!(
            place_joint_group(cap - 23, 23),
            JointPlacement::At(cap - 23)
        );
        assert_eq!(place_joint_group(cap - 22, 23), JointPlacement::PaletteFull);
        assert_eq!(place_joint_group(cap, 1), JointPlacement::PaletteFull);
        assert_eq!(
            place_joint_group(0, JOINTS_PER_PAGE + 1),
            JointPlacement::ModelTooLarge
        );
    }
}
