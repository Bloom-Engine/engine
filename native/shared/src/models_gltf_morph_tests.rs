use super::super::*;

/// Binary chunk under construction: each push is one buffer view.
#[derive(Default)]
struct Bin {
    data: Vec<u8>,
    views: Vec<String>,
}

impl Bin {
    fn push(&mut self, bytes: &[u8]) -> usize {
        while self.data.len() % 4 != 0 {
            self.data.push(0);
        }
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{}}}"#,
            bytes.len()
        ));
        self.views.len() - 1
    }
    fn f32s(&mut self, values: &[f32]) -> usize {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.push(&bytes)
    }
    fn u16s(&mut self, values: &[u16]) -> usize {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.push(&bytes)
    }
}

fn glb(json: &str, bin: &[u8]) -> Vec<u8> {
    let mut json = json.as_bytes().to_vec();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut bin = bin.to_vec();
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

/// A skinned face quad with "jawOpen" (dense accessor) and "smile" (a
/// SPARSE accessor without a buffer view, as Blender writes shape keys),
/// and an unskinned teeth triangle with "jawOpen" and an all-zero
/// "tongueOut". One animation drives the face's two weights.
fn face_glb() -> Vec<u8> {
    let mut b = Bin::default();
    let mut acc: Vec<String> = Vec::new();
    let mut accessor = |view: usize, ty: &str, ct: u32, count: usize, extra: &str| {
        acc.push(format!(
            r#"{{"bufferView":{view},"componentType":{ct},"count":{count},"type":"{ty}"{extra}}}"#
        ));
        acc.len() - 1
    };
    let face_pos = b.f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    let face_joints = b.u16s(&[0; 16]);
    let face_weights = b.f32s(&[
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ]);
    let face_idx = b.u16s(&[0, 1, 2, 0, 2, 3]);
    let jaw_pos = b.f32s(&[0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let smile_nrm = b.f32s(&[0.0; 12]);
    let smile_idx = b.u16s(&[2]);
    let smile_val = b.f32s(&[0.125, 0.0, 0.0]);
    let teeth_pos = b.f32s(&[0.0, 0.0, 0.1, 1.0, 0.0, 0.1, 0.5, 0.5, 0.1]);
    let teeth_idx = b.u16s(&[0, 1, 2]);
    let teeth_jaw = b.f32s(&[0.0, -0.2, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0]);
    let teeth_tongue = b.f32s(&[0.0; 9]);
    let times = b.f32s(&[0.0, 1.0]);
    let weights = b.f32s(&[0.0, 0.0, 1.0, 0.5]);
    let ibm = b.f32s(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    let a_face_pos = accessor(face_pos, "VEC3", 5126, 4, r#","min":[0,0,0],"max":[1,1,0]"#);
    let a_face_j = accessor(face_joints, "VEC4", 5123, 4, "");
    let a_face_w = accessor(face_weights, "VEC4", 5126, 4, "");
    let a_face_i = accessor(face_idx, "SCALAR", 5123, 6, "");
    let a_jaw = accessor(
        jaw_pos,
        "VEC3",
        5126,
        4,
        r#","min":[0,-0.25,0],"max":[0,0,0]"#,
    );
    let a_smile_n = accessor(smile_nrm, "VEC3", 5126, 4, "");
    // Sparse, no bufferView: zeros with one substituted element.
    acc.push(format!(
        r#"{{"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[0.125,0,0],
            "sparse":{{"count":1,"indices":{{"bufferView":{smile_idx},"componentType":5123}},
                       "values":{{"bufferView":{smile_val}}}}}}}"#
    ));
    let a_smile_p = acc.len() - 1;
    let mut accessor = |view: usize, ty: &str, ct: u32, count: usize, extra: &str| {
        acc.push(format!(
            r#"{{"bufferView":{view},"componentType":{ct},"count":{count},"type":"{ty}"{extra}}}"#
        ));
        acc.len() - 1
    };
    let a_teeth_pos = accessor(
        teeth_pos,
        "VEC3",
        5126,
        3,
        r#","min":[0,0,0.1],"max":[1,0.5,0.1]"#,
    );
    let a_teeth_i = accessor(teeth_idx, "SCALAR", 5123, 3, "");
    let a_teeth_jaw = accessor(
        teeth_jaw,
        "VEC3",
        5126,
        3,
        r#","min":[0,-0.2,0],"max":[0,0,0]"#,
    );
    let a_tongue = accessor(
        teeth_tongue,
        "VEC3",
        5126,
        3,
        r#","min":[0,0,0],"max":[0,0,0]"#,
    );
    let a_times = accessor(times, "SCALAR", 5126, 2, r#","min":[0],"max":[1]"#);
    let a_weights = accessor(weights, "SCALAR", 5126, 4, "");
    let a_ibm = accessor(ibm, "MAT4", 5126, 1, "");

    let json = format!(
        r#"{{
  "asset":{{"version":"2.0"}},
  "scene":0,
  "scenes":[{{"nodes":[0,1,2]}}],
  "nodes":[
    {{"name":"Face","mesh":0,"skin":0}},
    {{"name":"Teeth","mesh":1}},
    {{"name":"Head"}}
  ],
  "skins":[{{"joints":[2],"inverseBindMatrices":{a_ibm}}}],
  "meshes":[
    {{"name":"Face","extras":{{"targetNames":["jawOpen","smile"]}},
      "primitives":[{{"attributes":{{"POSITION":{a_face_pos},"JOINTS_0":{a_face_j},"WEIGHTS_0":{a_face_w}}},
        "indices":{a_face_i},
        "targets":[{{"POSITION":{a_jaw}}},{{"POSITION":{a_smile_p},"NORMAL":{a_smile_n}}}]}}]}},
    {{"name":"Teeth","extras":{{"targetNames":["jawOpen","tongueOut"]}},
      "primitives":[{{"attributes":{{"POSITION":{a_teeth_pos}}},"indices":{a_teeth_i},
        "targets":[{{"POSITION":{a_teeth_jaw}}},{{"POSITION":{a_tongue}}}]}}]}}
  ],
  "animations":[{{"name":"talk",
    "samplers":[{{"input":{a_times},"output":{a_weights},"interpolation":"LINEAR"}}],
    "channels":[{{"sampler":0,"target":{{"node":0,"path":"weights"}}}}]}}],
  "buffers":[{{"byteLength":{len}}}],
  "bufferViews":[{views}],
  "accessors":[{accessors}]
}}"#,
        len = b.data.len(),
        views = b.views.join(","),
        accessors = acc.join(","),
    );
    glb(&json, &b.data)
}

#[test]
fn targets_share_slots_by_name_and_sparse_accessors_are_read() {
    let data = face_glb();
    let model = load_gltf(&data).expect("synthetic morph GLB loads");
    let face = model
        .meshes
        .iter()
        .find(|m| m.vertices.len() == 4)
        .expect("face");
    let teeth = model
        .meshes
        .iter()
        .find(|m| m.vertices.len() == 3)
        .expect("teeth");
    let fm = face.morph.as_ref().expect("face has morphs");
    let tm = teeth.morph.as_ref().expect("teeth have morphs");
    // One name table for the model: jawOpen, smile, tongueOut.
    assert_eq!(
        fm.names.as_slice(),
        &["jawOpen".to_string(), "smile".into(), "tongueOut".into()]
    );
    assert!(Arc::ptr_eq(&fm.names, &tm.names));
    // jawOpen moves face vertex 0; the sparse smile moves vertex 2 only.
    assert_eq!(fm.ranges, vec![0, 1, 1, 2, 2]);
    assert_eq!(fm.vertex_entries(0)[0].slot, 0);
    assert_eq!(fm.vertex_entries(2)[0].slot, 1);
    let jaw = fm.position_offset(0, &[1.0, 0.0, 0.0]);
    assert!((jaw[1] + 0.25).abs() < 1e-3, "{jaw:?}");
    let smile = fm.position_offset(2, &[0.0, 1.0, 0.0]);
    assert!((smile[0] - 0.125).abs() < 1e-3, "{smile:?}");
    // The teeth's jawOpen is slot 0 too; the all-zero tongueOut costs no entry.
    assert_eq!(tm.entries.len(), 2);
    assert!(tm.entries.iter().all(|e| e.slot == 0));
    let teeth_jaw = tm.position_offset(1, &[0.5]);
    assert!((teeth_jaw[1] + 0.1).abs() < 1e-3);
    assert!(!fm.has_tangents && fm.has_normals);
}

#[test]
fn every_loader_reads_the_same_targets() {
    let data = face_glb();
    let plain = load_gltf(&data).unwrap();
    let staged = load_gltf_staged(&data).unwrap().model;
    let count = |m: &ModelData| -> Vec<usize> {
        m.meshes
            .iter()
            .map(|mesh| mesh.morph.as_ref().map_or(0, |t| t.entries.len()))
            .collect()
    };
    assert_eq!(count(&plain), count(&staged));
    assert_eq!(count(&plain).iter().sum::<usize>(), 4);
}

#[test]
fn weights_channels_become_tracks_over_model_slots() {
    let data = face_glb();
    let anim = load_gltf_animation(&data).expect("animation loads");
    let clip = &anim.animations[0];
    assert_eq!(clip.morph_tracks.len(), 1);
    let track = &clip.morph_tracks[0];
    assert_eq!(track.slots, vec![0, 1]);
    assert_eq!(track.times, vec![0.0, 1.0]);
    assert_eq!(track.values, vec![0.0, 0.0, 1.0, 0.5]);
    assert!((clip.duration - 1.0).abs() < 1e-6);

    // Through the mixer, by name, with an explicit weight on top.
    let mut manager = ModelManager::new();
    let model = manager.models.alloc(load_gltf(&data).unwrap());
    let h = manager.animations.alloc(anim);
    assert_eq!(manager.model_morph_count(model), 3);
    assert_eq!(manager.model_morph_index(model, "smile"), 1);
    assert_eq!(manager.model_morph_index(model, "frown"), -1);
    manager.anim_play(h, 0, 0.0, 1.0, false);
    manager.advance_and_update(h, 0.5);
    assert!((manager.anim_morph_weight(h, 0) - 0.5).abs() < 1e-5);
    assert!((manager.anim_morph_weight(h, 1) - 0.25).abs() < 1e-5);
    manager.anim_set_morph_weight(h, 2, 0.4);
    assert!((manager.anim_morph_weight(h, 2) - 0.4).abs() < 1e-6);
}
