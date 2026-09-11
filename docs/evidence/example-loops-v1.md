# Shared example loops and scalar interpolation

Six portable examples used blocking source loops, preventing the browser from
scheduling their frames. They now use the shared `runGame` callback and cleanup:
test3d, dungeon-crawl, isometric-rpg, kart-racer, space-blaster and voxel-sandbox.
The 3D test now closes its window, the space game closes audio during cleanup,
and the voxel game restores the cursor before closing.

## Runtime findings and corrections

Actual native execution exposed failures that compilation did not detect.

- Perry 0.5.1220's integer specialization truncates fractional inputs in
  arithmetic-only number functions. For example, native `lerp(10, 20, 0.25)`
  returned 10, while WASM returned 12.5. Five quadratic/cubic easing functions
  also returned incorrect values. An identity division keeps these public
  functions on the compiler's floating-point path. The generated native trace
  and earlier failed observations are retained. The division preserves the
  mathematical operation and can be removed by LLVM after correct lowering.
- The same specialization loses globals inside arithmetic expressions. The
  native voxel index returned 32 for both `(32, 0, 32)` and `(32, 5, 32)`,
  overwriting unrelated terrain cells. LLVM showed multiplication by zero
  instead of the captured dimension. Passing width/depth explicitly preserves
  the intended packed layout and restores visible terrain.
- Combined interpolation calls and nested camera writes produced undefined
  fields in the native isometric game. Local result variables preserve the
  camera values in both camera examples. Isometric entities now retain map
  coordinates only; screen coordinates are derived when drawing. This avoids
  a separate observed failure where stored derived coordinates became undefined
  during intervening draw loops, hiding the player and NPCs.

The compiler binary and source remain unchanged. These corrections qualify
the stated engine/example paths; general numeric specialization and object
assignment behavior in arbitrary user programs remain compiler limitations.

## Required checks and local observations

`tools/ci/scalar_math_smoke.py` compiles and executes the public math code with
Perry as a native executable and as actual WASM. Eight sample points cover
fractional interpolation, both easing branches, endpoints and extrapolation,
with independent host expectations. NaN, infinity and signed-zero behavior also
pass. The WASM runner uses the production void-return compatibility and no
engine FFI, renderer, browser or network. CI retains this contract separately
from the existing fixed-step lifecycle check.

`tools/ci/native_example_smoke.py` compiles bounded diagnostic copies of all six
examples against the actual checkout. It preserves their update/draw and cleanup
bodies, captures frame eight and stops after readback. All six pass locally on
Radeon 760M DX12 with nine frames and one completed cleanup each. The six
unchanged source entries also compile and splice to WASM with resolved engine
imports; that is compilation evidence, not browser execution.

The native gate requires recognizable game content at the requested viewport,
including the isometric player and terrain, the space game's player, the dungeon
player/floor, a kart and track, the 3D cube and voxel terrain. Applying these
checks to the actual earlier voxel HUD-only and isometric missing-player images
rejects them; all six corrected native captures pass. Synthetic controls also
reject a nonblank HUD, missing player, incomplete readback and duplicate cleanup.
These checks do not replace approved renderer image goldens or prove gameplay.
The all-20 canonical native compile/link gate remains required.

The first diagnostic native runs failed before rendering because DXC/DXIL was
not available to the process. Running the same binaries with the existing SDK
shader-runtime directory succeeded. Logs retain the failure and DLL hashes.
Automatic distribution of shader runtimes remains packaging work; diagnostic
crash dumps are private and excluded from publication.

Full browser execution of these six examples, other canonical runtime entries,
input/gameplay coverage and visible native presentation remain open. Visual
inspection also found missing grid lines in test3d; the cube/content smoke does
not qualify that separate renderer defect. Current-source full strict graphics
qualification remains required before integration. Hosted results for this
candidate are pending; no issue is closed by the local checks.
