//! Shared, strict reader for Bloom's versioned virtual-geometry archive.
//!
//! The offline cooker and every runtime backend use this crate so malformed
//! content cannot be interpreted differently by the producer and consumer.

mod decode;
mod hash;
mod types;
mod validate;
mod vertex;
mod wire;

pub use decode::decode_geometry;
pub use hash::{hex_hash, sha256};
pub use types::{
    ClusterRecord, CompatibilityReason, CompatibilityRecord, GeometryArchive, PageRecord,
    VertexEncoding, CLUSTER_RECORD_BYTES, COMPATIBILITY_RECORD_BYTES, DEFAULT_PAGE_BYTES,
    ENDIAN_TAG, FLAG_ALPHA_MASKED, FLAG_COARSE_ROOT, FLAG_DOUBLE_SIDED, FLOAT32_VERTEX_BYTES,
    HEADER_BYTES, MAGIC, MAX_PAGE_BYTES, MIN_PAGE_BYTES, NO_RELATION, PAGE_RECORD_BYTES,
    QUANTIZED_VERSION, QUANTIZED_VERTEX_BYTES, VERSION,
};
pub use validate::validate_page_budget;
