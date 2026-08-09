// Bloom global material indirection ABI — version 1.
//
// This header is consumed by GPU-driven opaque passes. The legacy/custom
// material ABI remains version 3 and unchanged: Tier C therefore stays a
// pixel-identical compatibility path.

const BLOOM_RESOURCE_SLOT_MASK: u32 = 0x000fffffu;
const BLOOM_RESOURCE_GENERATION_SHIFT: u32 = 20u;
const BLOOM_GLOBAL_MATERIAL_RECORD_VERSION: u32 = 1u;
const BLOOM_GLOBAL_MATERIAL_VERSION_SHIFT: u32 = 24u;
const BLOOM_GLOBAL_MATERIAL_LOBE_MASK: u32 = 0x00ffffffu;

struct GlobalMaterialRecord {
  // x=generation, y=high-8-bit layered-PBR version + low-24-bit lobe mask,
  // z=user-param byte offset, w=user-param byte size.
  header:          vec4<u32>,
  base_color:      vec4<f32>,
  metal_rough:     vec4<f32>,
  emissive:        vec4<f32>,
  shading_model:   vec4<f32>,
  foliage_params:  vec4<f32>,
  texture_ids_0:   vec4<u32>,
  texture_ids_1:   vec4<u32>,
  texture_ids_2:   vec4<u32>,
  sampler_ids_0:   vec4<u32>,
  sampler_ids_1:   vec4<u32>,
};

struct GlobalMaterialTable {
  records: array<GlobalMaterialRecord>,
};

struct GlobalResourceGenerationTable {
  // x=texture generation, y=sampler generation, z=texture flags,
  // w=texture semantic. flags: bit0=sRGB content, bit1=hardware decode,
  // bit2=HDR-linear.
  entries: array<vec4<u32>>,
};

// The pipeline using this header supplies this persistent global layout as one
// persistent global group. No per-material bind group is created or switched.
@group(2) @binding(0) var<storage, read> global_materials: GlobalMaterialTable;
@group(2) @binding(1) var global_textures: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var global_samplers: binding_array<sampler>;
@group(2) @binding(3) var<storage, read> global_resource_generations: GlobalResourceGenerationTable;

fn bloom_resource_slot(id: u32) -> u32 {
  return id & BLOOM_RESOURCE_SLOT_MASK;
}

fn bloom_resource_generation(id: u32) -> u32 {
  return id >> BLOOM_RESOURCE_GENERATION_SHIFT;
}

// Record zero is the diagnostic fallback. It is white, rough, non-metallic,
// non-emissive, and carries fallback texture/sampler ID zero.
fn bloom_material_record(id: u32) -> GlobalMaterialRecord {
  let slot = bloom_resource_slot(id);
  if (slot == 0u || slot >= arrayLength(&global_materials.records)) {
    return global_materials.records[0u];
  }
  let candidate = global_materials.records[slot];
  if (candidate.header.x != bloom_resource_generation(id)) {
    return global_materials.records[0u];
  }
  return candidate;
}

fn bloom_global_material_record_version(material_record: GlobalMaterialRecord) -> u32 {
  return material_record.header.y >> BLOOM_GLOBAL_MATERIAL_VERSION_SHIFT;
}

fn bloom_global_material_lobe_mask(material_record: GlobalMaterialRecord) -> u32 {
  return material_record.header.y & BLOOM_GLOBAL_MATERIAL_LOBE_MASK;
}

fn bloom_texture_slot(id: u32) -> u32 {
  let slot = bloom_resource_slot(id);
  if (slot == 0u || slot >= arrayLength(&global_resource_generations.entries)) {
    return 0u;
  }
  if (global_resource_generations.entries[slot].x != bloom_resource_generation(id)) {
    return 0u;
  }
  return slot;
}

fn bloom_sampler_slot(id: u32) -> u32 {
  let slot = bloom_resource_slot(id);
  if (slot == 0u || slot >= arrayLength(&global_resource_generations.entries)) {
    return 0u;
  }
  if (global_resource_generations.entries[slot].y != bloom_resource_generation(id)) {
    return 0u;
  }
  return slot;
}

fn bloom_srgb_channel_to_linear(v: f32) -> f32 {
  if (v <= 0.04045) {
    return v / 12.92;
  }
  return pow((v + 0.055) / 1.055, 2.4);
}

fn bloom_decode_registered_color(texture_slot: u32, value: vec4<f32>) -> vec4<f32> {
  let flags = global_resource_generations.entries[texture_slot].z;
  let srgb_content = (flags & 1u) != 0u;
  let hardware_decoded = (flags & 2u) != 0u;
  if (!srgb_content || hardware_decoded) {
    return value;
  }
  return vec4<f32>(
    bloom_srgb_channel_to_linear(value.r),
    bloom_srgb_channel_to_linear(value.g),
    bloom_srgb_channel_to_linear(value.b),
    value.a
  );
}

fn bloom_sample_base_color(material_record: GlobalMaterialRecord, uv: vec2<f32>) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(material_record.texture_ids_0.x);
  let sampler_slot = bloom_sampler_slot(material_record.sampler_ids_0.x);
  let sampled = textureSample(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv
  );
  return bloom_decode_registered_color(texture_slot, sampled) * material_record.base_color;
}

fn bloom_sample_raw(
  texture_id: u32,
  sampler_id: u32,
  uv: vec2<f32>
) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(texture_id);
  let sampler_slot = bloom_sampler_slot(sampler_id);
  return textureSample(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv
  );
}

fn bloom_sample_raw_bias(
  texture_id: u32,
  sampler_id: u32,
  uv: vec2<f32>,
  bias: f32
) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(texture_id);
  let sampler_slot = bloom_sampler_slot(sampler_id);
  return textureSampleBias(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv,
    bias
  );
}

fn bloom_sample_raw_level(
  texture_id: u32,
  sampler_id: u32,
  uv: vec2<f32>,
  level: f32
) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(texture_id);
  let sampler_slot = bloom_sampler_slot(sampler_id);
  return textureSampleLevel(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv,
    level
  );
}

fn bloom_base_color_dimensions(material_record: GlobalMaterialRecord) -> vec2<u32> {
  let texture_slot = bloom_texture_slot(material_record.texture_ids_0.x);
  return textureDimensions(global_textures[texture_slot]);
}

fn bloom_sample_registered_color_bias(
  texture_id: u32,
  sampler_id: u32,
  uv: vec2<f32>,
  bias: f32
) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(texture_id);
  let sampler_slot = bloom_sampler_slot(sampler_id);
  let sampled = textureSampleBias(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv,
    bias
  );
  return bloom_decode_registered_color(texture_slot, sampled);
}

// Slot zero is the white diagnostic texture. For a missing normal map the
// semantic fallback must reproduce the legacy RGBA8 default-normal texel
// exactly: (128, 128, 255, 0). The 128/255 RGB quantization is observable,
// while alpha is LEADR/Toksvig filtered variance and must be zero for a flat
// unfiltered normal. Alpha=1 forces roughness to 1 for every untextured
// GPU-driven material.
fn bloom_sample_normal_raw_bias(
  material_record: GlobalMaterialRecord,
  uv: vec2<f32>,
  bias: f32
) -> vec4<f32> {
  let texture_slot = bloom_texture_slot(material_record.texture_ids_0.y);
  if (texture_slot == 0u) {
    return vec4<f32>(128.0 / 255.0, 128.0 / 255.0, 1.0, 0.0);
  }
  let sampler_slot = bloom_sampler_slot(material_record.sampler_ids_0.y);
  return textureSampleBias(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv,
    bias
  );
}

fn bloom_sample_normal(material_record: GlobalMaterialRecord, uv: vec2<f32>) -> vec3<f32> {
  let texture_slot = bloom_texture_slot(material_record.texture_ids_0.y);
  if (texture_slot == 0u) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  let sampler_slot = bloom_sampler_slot(material_record.sampler_ids_0.y);
  // Normal resources are required to register a linear view. Renormalization
  // keeps filtered mip values on the unit hemisphere.
  let encoded = textureSample(
    global_textures[texture_slot],
    global_samplers[sampler_slot],
    uv
  ).xyz;
  return normalize(encoded * 2.0 - 1.0);
}
