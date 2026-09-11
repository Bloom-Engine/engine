import {
  lerp, clamp, remap, easeInQuad, easeOutQuad, easeInOutQuad,
  easeInCubic, easeOutCubic, easeInOutCubic,
} from '../../../src/math/index';

// Compile the public implementation, including its cross-module call paths.
// Scalar JSON avoids Perry WASM's separate object JSON.stringify limitation.
const samples = [-0.25, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.25];
let result = '{"samples":[';
for (let i = 0; i < samples.length; i++) {
  const t = samples[i];
  if (i > 0) result = result + ',';
  result = result + '[' + t + ',' + lerp(-2.5, 4.75, t)
    + ',' + easeInQuad(t) + ',' + easeOutQuad(t) + ',' + easeInOutQuad(t)
    + ',' + easeInCubic(t) + ',' + easeOutCubic(t) + ',' + easeInOutCubic(t)
    + ',' + clamp(t, 0, 1) + ',' + remap(t, 0, 1, -2.5, 4.75) + ']';
}
const nan = 0 / 0;
const infinity = 1 / 0;
const a = lerp(0, 1, nan);
const b = easeInQuad(nan);
const c = easeOutQuad(nan);
const d = easeInOutQuad(nan);
const e = easeInCubic(nan);
const f = easeOutCubic(nan);
const nanPreserved = a !== a && b !== b && c !== c && d !== d && e !== e && f !== f;
const infinityPreserved = lerp(0, 1, infinity) === infinity
  && easeInQuad(infinity) === infinity && easeOutQuad(infinity) === -infinity
  && easeInOutQuad(infinity) === -infinity && easeInCubic(infinity) === infinity
  && easeOutCubic(infinity) === infinity;
const negativeZeroPreserved = 1 / easeOutQuad(-0) === -infinity
  && 1 / easeInCubic(-0) === -infinity;
result = result + '],"nanPreserved":' + nanPreserved
  + ',"infinityPreserved":' + infinityPreserved
  + ',"negativeZeroPreserved":' + negativeZeroPreserved + '}';
console.log('BLOOM_SCALAR_MATH_RESULT:' + result);
