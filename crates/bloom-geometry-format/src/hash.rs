use sha2::{Digest, Sha256};

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Hash the complete geometry source closure exactly once for both the cooker
/// and runtime model router. Buffer order is glTF buffer index order.
pub fn geometry_source_sha256(source_bytes: &[u8], buffers: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"bloom-static-geometry-source-v1");
    hasher.update((source_bytes.len() as u64).to_le_bytes());
    hasher.update(source_bytes);
    for (index, buffer) in buffers.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        hasher.update((buffer.len() as u64).to_le_bytes());
        hasher.update(buffer);
    }
    hasher.finalize().into()
}

pub fn hex_hash(hash: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
