pub(crate) fn read_u16(bytes: &[u8], offset: usize, label: &str) -> Result<u16, String> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

pub(crate) fn read_i16(bytes: &[u8], offset: usize, label: &str) -> Result<i16, String> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(i16::from_le_bytes(value.try_into().unwrap()))
}

pub(crate) fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

pub(crate) fn read_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

pub(crate) fn read_usize(bytes: &[u8], offset: usize, label: &str) -> Result<usize, String> {
    let value = read_u64(bytes, offset, label)?;
    usize::try_from(value).map_err(|_| format!("{label} exceeds host address space"))
}

pub(crate) fn read_f32(bytes: &[u8], offset: usize, label: &str) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(bytes, offset, label)?))
}

pub(crate) fn read_f32x3(bytes: &[u8], offset: usize, label: &str) -> Result<[f32; 3], String> {
    Ok([
        read_f32(bytes, offset, label)?,
        read_f32(bytes, offset + 4, label)?,
        read_f32(bytes, offset + 8, label)?,
    ])
}

pub(crate) fn read_hash(bytes: &[u8], offset: usize, label: &str) -> Result<[u8; 32], String> {
    bytes
        .get(offset..offset + 32)
        .ok_or_else(|| format!("{label} is truncated"))?
        .try_into()
        .map_err(|_| format!("{label} has invalid length"))
}

pub(crate) fn checked_table_end(
    start: usize,
    count: usize,
    stride: usize,
    label: &str,
) -> Result<usize, String> {
    count
        .checked_mul(stride)
        .and_then(|bytes| start.checked_add(bytes))
        .ok_or_else(|| format!("{label} range overflow"))
}

pub(crate) fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
