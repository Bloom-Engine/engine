/** Numeric finiteness without Number static intrinsics missing in Perry WASM. */
export function isFiniteNumber(value: number): boolean {
  // Finite values subtract to zero; NaN and either infinity subtract to NaN.
  // Keep the type guard so JavaScript callers do not coerce strings or null.
  return typeof value === 'number' && value - value === 0;
}
