fn main() {
    let data = std::fs::read("/Users/amlug/projects/bloom/engine/examples/renderer-test/assets/DamagedHelmet.glb").unwrap();
    let gltf = gltf::Gltf::from_slice(&data).unwrap();
    let mut buffer_data: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => { if let Some(b) = gltf.blob.as_ref() { buffer_data.push(b.clone()); } }
            _ => { buffer_data.push(Vec::new()); }
        }
    }
    for mesh in gltf.meshes() {
        for (i, prim) in mesh.primitives().enumerate() {
            let reader = prim.reader(|b| buffer_data.get(b.index()).map(|d| d.as_slice()));
            let has_tan = reader.read_tangents().is_some();
            let has_norm = reader.read_normals().is_some();
            let mat = prim.material();
            let has_nmap = mat.normal_texture().is_some();
            let has_mr = mat.pbr_metallic_roughness().metallic_roughness_texture().is_some();
            let has_em = mat.emissive_texture().is_some();
            let has_occl = mat.occlusion_texture().is_some();
            println!("mesh {} prim {}: tangents={} normals={} normal_map={} mr_tex={} em_tex={} occl_tex={}", mesh.index(), i, has_tan, has_norm, has_nmap, has_mr, has_em, has_occl);
        }
    }
}
