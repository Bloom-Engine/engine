# Complete installed starter browser acceptance

The minimal compiled-game fixture verifies rendering and lifecycle hooks, but
does not exercise the generated starter's asset read, named function callbacks
or text rendering. The new gate creates a project through the installed CLI,
then runs the installed `bloom run --web` command against its unchanged source.
The command compiles game and engine WASM, writes the asset manifest and starts
its development server. The preparer checks served HTML, manifest, asset and
engine bytes, including the WASM MIME type, then stops its owned process tree.
The complete generated website and every file's SHA-256 are retained.

Windows CI uses the existing qualified Perry 0.5.1220 profile and a pinned
[official wasm-pack 0.15.0 archive](https://github.com/wasm-bindgen/wasm-pack/releases/tag/v0.15.0).
Archive SHA-256 is checked before selecting the executable from the tar file;
the tool's version and executable hash are recorded. The build adds no test
markers or alternate logic to the starter source.

Hosted macOS Chrome loads this exact website through its production bootstrap.
The monitor observes actual asset fetch/read, text and rectangle FFI calls and
the real Perry callback dispatcher. It also verifies every clear's opaque-black
RGBA arguments. It requests normal engine stop after eight
successful frames and requires exactly one cleanup. The screenshot must contain
the 800x450 viewport, the square's white interior at its actual interpolated
position, a populated text region and the expected black background. Text
coverage allows glyph antialiasing differences; the real draw call must contain
the exact text read from `assets/welcome.txt`. The existing startup-fault gate
remains required. This new gate does not replace the game's draw calls or load
a fake physics factory.

## Failure found by executing the full starter

The first real generated WASM run fails during named `init` callback dispatch:
Perry exports that void function without a WASM result, but its closure bridge
passes the returned JavaScript `undefined` into an i64 decoder. The decoder then
raises `Cannot mix BigInt and other types`. Native starter rendering passed;
compilation-only web evidence did not expose this startup failure. The raw
failed probe and the observed `undefined` return are retained.

Bloom's production splicer now installs a compatibility bridge that preserves
an actual void return and delegates every boxed value to the existing decoder.
It does not suppress other decoder exceptions. Actual native/WASM lifecycle
checks now cover named functions for all five hooks. A recording-FFI probe runs
the real compiled starter and production scheduler, reads the asset text,
records eight text/rectangle calls and cleans up once with the correction.
That initial probe did not validate color channels, load the engine renderer
or claim browser/network acceptance.

The corrected installed `bloom run --web` command passes locally in 64.531
seconds, including game/engine compilation, serving the four required resources
with exact file bytes and the correct WASM MIME type, and controlled shutdown.
The resulting 11-file website includes the actual production compatibility
bridge. Running that exact generated boot script with the recording FFI also
passes all starter callbacks and one cleanup. Ten Python acceptance-control
tests and the repository contracts pass; the text/image verifier accepts the
previous actual native starter capture. Hosted browser rendering is pending.

The first hosted run, 34571662102, passed all 21 preceding Tests jobs and the
existing browser checks. The complete starter verifier then rejected the
downloaded website before launch: upload-artifact omitted wasm-pack's
`pkg/.gitignore`, while its build receipt correctly included all 11 files.
All ten downloaded file hashes match. The scoped starter artifact upload now
includes hidden files, preserving the complete site and the exact hash check.
That failure remains retained.

The retry, 34574961654, passed those same 21 preceding jobs and both existing
browser checks. The complete site passed its file-hash check, loaded the actual
asset and reached drawing. The full starter then failed its first frame:
`Colors.BLACK` and `Colors.WHITE` reached FFI with undefined RGBA channels, and
WebGPU rejected the non-finite clear alpha. Its error, draw arguments and
failure screenshot are retained. This demonstrates why call counts and a
successful build are insufficient evidence of complete starter execution.

The starter now defines its two colors as explicit typed RGBA literals, matching
the qualified native examples. Its strengthened actual-WASM recording probe
rejects the earlier site and accepts the fresh build's exact black clear and
white text/square arguments in all eight frames, with one cleanup. The hosted
monitor now retains clear arguments as well as text/rectangle arguments; the
verifier rejects missing channels. Fresh installed `bloom run --web` passes in
47.218 seconds with unchanged source, the complete website and exact served
bytes retained. This correction qualifies the starter's color usage; general
imported palette compatibility remains separate. Hosted rendering of the
corrected starter remains pending.

The corrected unmodified starter also compiles and links natively. A bounded
copy on Radeon 760M / DX12 renders the expected 800x450 black background, white
square and asset text (857 lit text pixels), exits after nine frames and runs
cleanup once. The capture SHA-256 is
`d89c3b30d0e472526d6a39869e0c9e3ae3e9c0943f87b2dbf87a34cc91ac0b3f`.
This local native check uses a linked checkout and the SDK's shader dependencies;
it does not qualify a clean Windows installation.

General Perry exception propagation, browser physics acceptance, the complete
canonical runtime matrix, visible native presentation and clean distribution
packaging remain separate. A browser warning about optional physics startup is
retained as observed; the starter itself does not exercise physics.
