// ============================================================
// Bloom Bistro Validation Scene
// ============================================================
// Amazon Lumberyard Bistro — Parisian street corner scene. Different
// profile from Sponza: outdoor lighting dominated by a single sun,
// varied materials (stone, brick, painted wood, glass, fabric awnings,
// metal fixtures, foliage), and long sight lines. A good cross-check
// that the rendering pipeline doesn't over-fit to Sponza's atrium
// geometry and IBL.
//
// Assets aren't shipped with the repo — they total ~1.2 GB. To set
// up this scene, clone zeux/niagara_bistro (MIT-licensed glTF
// conversion of NVIDIA's Bistro) into `assets/`:
//
//   cd examples/bistro
//   git clone https://github.com/zeux/niagara_bistro.git assets
//
// Use `scripts/run-bistro-rich.sh` to build and launch the complete expanded
// scene: all 2,909 authored mesh-node placements backed by shared source
// geometry and the directly supported bistrox material set.

import {
  initWindow, closeWindow, windowShouldClose, beginDrawing, endDrawing, takeScreenshot,
  captureFrameToPng, captureDebugIntermediates,
  setEnvClearFromHdr, setTargetFPS, getDeltaTime, getFPS,
  isKeyDown, isKeyPressed,
  getMouseDeltaX, getMouseDeltaY,
  disableCursor, enableCursor,
  beginMode3D, endMode3D,
  setFog, setSunShafts, setVignette, setChromaticAberration,
  setAutoExposure, setAutoExposureKey, setAutoExposureRate,
  setEnvIntensity, setManualExposure, setTaaEnabled, setBloomEnabled,
  setSsgiEnabled, setSsrEnabled, setSharpenStrength, setShadowsAlwaysFresh, setMotionBlurEnabled,
  setPathTracing,
  setBloomIntensity, setSsaoIntensity, setSsaoRadius,
  setSsgiIntensity, setSsgiRadius, setTonemap, Tonemap,
  setColorSaturation,
  setQualityPreset, setRenderScale, getRenderScale, getPhysicalWidth, getPhysicalHeight,
  getCommandLineArgs, resize,
} from "bloom/core";
import { parseQualityRun, QualityRun } from "bloom/quality";
import { Key } from "bloom/core";
import { drawText } from "bloom/text";
import {
  setAmbientLight, setDirectionalLight, loadModel, setProceduralSky, setSunDirection,
} from "bloom/models";
import {
  enableShadows, addDirectionalLight,
  createSceneNode, attachModelToNode, setSceneNodeTransform,
} from "bloom/scene";
import { clamp, mat4Identity } from "bloom/math";

const SCREEN_W = 800;
const SCREEN_H = 450;
const MOUSE_SENS = 0.003;
const MOVE_SPEED = 5.0;
const SPRINT_MULT = 2.5;

// Auto-capture args (matches the sponza examples' CLI)
const argv: string[] = getCommandLineArgs();
const qualityConfig = parseQualityRun(argv);
let captureFrames = 0;
let capturePath = "";
let diagnosticCaptureFrames = 0;
let frameCount = 0;
let initYaw = 0.0;
let initCamX = -26.43;
let initCamY = 3.16;
let initCamZ = 11.17;
// Use the complete expanded-material scene by default. The sibling
// bistro.gltf relies on MSFT_texture_dds indirection and is retained for
// importer diagnostics; bistrox.gltf carries the same 2,909 placements in
// the directly supported texture/material form used by the quality demo.
let scenePath = "assets/bistrox.gltf";
let taaOverride = -1; // -1 = default, 0 = force off, 1 = force on
let bloomOverride = -1;
let fogOverride = -1;
let sunShaftOverride = -1;
let ssgiOverride = -1;
let ssrOverride = -1;
let motionBlurOverride = 0;
let vsmMotionPath = false;
let motionYaw = 0.0;
let motionFrames = 0;
let shadowsAlwaysFresh = false;
let interactiveQualityPreset = -1;
let interactiveRenderScale = -1.0;
let interactiveSharpenStrength = -1.0;
let godotReference = false;
let referenceManualExposure = 1.20;
let referenceSunIntensity = 2.60;
let referenceSkySunIntensity = 3.00;
let referenceEnvIntensity = 0.70;
let referenceAmbientIntensity = 0.06;
let referenceAutoExposure = true;
let referenceAutoExposureKey = 0.12;
let pathTracingMode = 0;
let fullscreen = false;
let cameraXOverride = false;
let cameraYOverride = false;
let cameraZOverride = false;
let yawOverride = false;
let pitchOverride = false;
let initPitch = 0.0;
for (let i = 1; i < argv.length; i = i + 1) {
  if (argv[i] === "--capture" && i + 2 < argv.length) {
    captureFrames = Math.floor(parseFloat(argv[i + 1]));
    capturePath = argv[i + 2];
  }
  if (argv[i] === "--debug-capture" && i + 1 < argv.length) {
    diagnosticCaptureFrames = Math.floor(parseFloat(argv[i + 1]));
  }
  if (argv[i] === "--yaw" && i + 1 < argv.length) {
    initYaw = parseFloat(argv[i + 1]);
    yawOverride = true;
  }
  if (argv[i] === "--pitch" && i + 1 < argv.length) {
    initPitch = parseFloat(argv[i + 1]);
    pitchOverride = true;
  }
  if (argv[i] === "--camera-x" && i + 1 < argv.length) {
    initCamX = parseFloat(argv[i + 1]);
    cameraXOverride = true;
  }
  if (argv[i] === "--camera-y" && i + 1 < argv.length) {
    initCamY = parseFloat(argv[i + 1]);
    cameraYOverride = true;
  }
  if (argv[i] === "--camera-z" && i + 1 < argv.length) {
    initCamZ = parseFloat(argv[i + 1]);
    cameraZOverride = true;
  }
  if (argv[i] === "--taa" && i + 1 < argv.length) {
    taaOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--bloom" && i + 1 < argv.length) {
    bloomOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--fog" && i + 1 < argv.length) {
    fogOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--sun-shafts" && i + 1 < argv.length) {
    sunShaftOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--ssgi" && i + 1 < argv.length) {
    ssgiOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--ssr" && i + 1 < argv.length) {
    ssrOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--motion-blur" && i + 1 < argv.length) {
    motionBlurOverride = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--scene" && i + 1 < argv.length) {
    scenePath = argv[i + 1];
  }
  if (argv[i] === "--vsm-motion-path") {
    vsmMotionPath = true;
  }
  if (argv[i] === "--motion-yaw" && i + 1 < argv.length) {
    motionYaw = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--motion-frames" && i + 1 < argv.length) {
    motionFrames = Math.max(0, Math.floor(parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--shadows-always-fresh") {
    shadowsAlwaysFresh = true;
  }
  if (argv[i] === "--quality-preset" && i + 1 < argv.length) {
    interactiveQualityPreset = Math.max(0, Math.min(4, Math.floor(parseFloat(argv[i + 1]))));
  }
  if (argv[i] === "--render-scale" && i + 1 < argv.length) {
    interactiveRenderScale = Math.max(0.15, Math.min(1.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--sharpen" && i + 1 < argv.length) {
    interactiveSharpenStrength = Math.max(0.0, Math.min(1.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--godot-reference") {
    godotReference = true;
  }
  if (argv[i] === "--fullscreen") {
    fullscreen = true;
  }
  if (argv[i] === "--manual-exposure" && i + 1 < argv.length) {
    referenceManualExposure = Math.max(0.01, Math.min(10.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--auto-exposure" && i + 1 < argv.length) {
    referenceAutoExposure = parseInt(argv[i + 1]) !== 0;
  }
  if (argv[i] === "--auto-exposure-key" && i + 1 < argv.length) {
    referenceAutoExposureKey = Math.max(0.01, Math.min(1.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--reference-sun" && i + 1 < argv.length) {
    referenceSunIntensity = Math.max(0.0, Math.min(20.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--reference-sky-sun" && i + 1 < argv.length) {
    referenceSkySunIntensity = Math.max(0.0, Math.min(20.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--reference-env" && i + 1 < argv.length) {
    referenceEnvIntensity = Math.max(0.0, Math.min(10.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--reference-ambient" && i + 1 < argv.length) {
    referenceAmbientIntensity = Math.max(0.0, Math.min(10.0, parseFloat(argv[i + 1])));
  }
  if (argv[i] === "--path-tracing" && i + 1 < argv.length) {
    pathTracingMode = Math.max(0, Math.min(2, Math.floor(parseFloat(argv[i + 1]))));
  }
}

// Match the launch pose of Bistro-Demo-Tweaked's Human-For-Scale camera.
// The camera is a 2.1 m spring arm attached to the player transform authored
// in MainScene.tscn. Explicit camera arguments always win.
if (godotReference) {
  if (!cameraXOverride) { initCamX = -3.2720; }
  if (!cameraYOverride) { initCamY = 1.5440; }
  if (!cameraZOverride) { initCamZ = 7.2358; }
  if (!yawOverride) { initYaw = -0.3440; }
  if (interactiveQualityPreset < 0) { interactiveQualityPreset = 4; }
  if (interactiveRenderScale < 0.0) { interactiveRenderScale = 1.0; }
  if (interactiveSharpenStrength < 0.0) { interactiveSharpenStrength = 0.20; }
}

// ---- Init ----
initWindow(SCREEN_W, SCREEN_H, "Bloom Bistro", fullscreen);
if (qualityConfig !== null) resize(SCREEN_W, SCREEN_H, SCREEN_W, SCREEN_H);
setTargetFPS(60);
let qualityRun: QualityRun | null = qualityConfig !== null ? new QualityRun(qualityConfig) : null;
// QualityRun applies these options for deterministic captures. Interactive
// launches used to parse the same flags but silently ignore them, leaving a
// requested Ultra session at the engine's 0.75 balanced default. Apply the
// preset first, then the explicit scale override, matching QualityRun order.
if (qualityRun === null) {
  if (interactiveQualityPreset >= 0) { setQualityPreset(interactiveQualityPreset as any); }
  if (interactiveRenderScale > 0.0) { setRenderScale(interactiveRenderScale); }
  if (interactiveSharpenStrength >= 0.0) { setSharpenStrength(interactiveSharpenStrength); }
  console.error(
    "BLOOM_BISTRO_RENDER output=" + getPhysicalWidth() + "x" + getPhysicalHeight()
      + " render_scale=" + getRenderScale().toFixed(2)
      + " quality_preset=" + interactiveQualityPreset,
  );
}
setEnvClearFromHdr("assets/outdoor.hdr");
enableShadows();
setShadowsAlwaysFresh(shadowsAlwaysFresh);

// Open-air street scene: the sky is the dominant IBL source. 1.2×
// env intensity gives colourful ambient reflection without washing
// out direct sunlight (now at 3.0 below).
if (godotReference) {
  // Bistro-Demo-Tweaked uses Godot's ProceduralSkyMaterial, an ACES output
  // transform, physical auto-exposure, and a 4250 K / 75 klux afternoon sun.
  // Bloom's public light intensity is a relative radiometric multiplier, so
  // preserve the authored sun/sky relationship rather than copying lux as a
  // dimensionless value. The direction is the Godot light's +Z basis: from a
  // shaded surface toward the sun.
  const godotSun = { x: 0.59732, y: 0.79653, z: -0.0935387 };
  setProceduralSky(true, { rayleighDensity: 1.0, mieDensity: 2.0, groundAlbedo: 0.1 });
  setSunDirection(godotSun, referenceSkySunIntensity);
  setEnvIntensity(referenceEnvIntensity);
  setTonemap(Tonemap.ACESFull);
  // The reference camera launches with auto exposure enabled. Godot's 0.22
  // `auto_exposure_scale` is applied on top of a physical camera sensitivity;
  // Bloom's key is an absolute HDR median target, so copying 0.22 directly
  // overexposes this scene by several stops. 0.12 maps the authored outdoor
  // luminance to the same mid-tone range while retaining adaptation between
  // the street and cafe. A fixed exposure remains available through
  // --auto-exposure 0, and --auto-exposure-key supports calibration captures.
  setAutoExposureKey(referenceAutoExposureKey);
  setAutoExposureRate(0.02);
  setAutoExposure(referenceAutoExposure);
  setManualExposure(referenceManualExposure);
  setColorSaturation(1.17);
  // Godot's numeric SSAO scale is not interchangeable with Bloom's. A raw
  // 2.3 copy creates dark silhouette halos (the familiar "cartoon outline"
  // failure) in Bloom. These values match the authored contact-darkening
  // intent without changing large-scale albedo or direct-light contrast.
  setSsaoRadius(0.65);
  setSsaoIntensity(0.85);
  setSsgiRadius(2.3);
  setSsgiIntensity(1.1);
  setBloomIntensity(0.035);
} else {
  setEnvIntensity(1.2);
  setAutoExposure(false);
  setManualExposure(1.0);
}
if (taaOverride === 0) { setTaaEnabled(false); }
if (taaOverride === 1) { setTaaEnabled(true); }
if (bloomOverride === 0) { setBloomEnabled(false); }
if (bloomOverride === 1) { setBloomEnabled(true); }
if (ssgiOverride === 0) { setSsgiEnabled(false); }
if (ssgiOverride === 1) { setSsgiEnabled(true); }
if (ssrOverride === 0) { setSsrEnabled(false); }
if (ssrOverride === 1) { setSsrEnabled(true); }
if (godotReference && ssrOverride < 0) { setSsrEnabled(false); }

// Same-camera image-quality isolation. These toggles deliberately live in
// the validation scene rather than changing renderer policy: they let an
// artist freeze one problematic view and identify whether softness comes
// from temporal reconstruction, indirect light, or reflections.
let diagnosticTaaEnabled = taaOverride !== 0
  && !(taaOverride < 0 && interactiveQualityPreset >= 0 && interactiveQualityPreset < 2);
let diagnosticSsgiEnabled = ssgiOverride === 1
  || (ssgiOverride < 0 && (interactiveQualityPreset < 0 || interactiveQualityPreset >= 3));
let diagnosticSsrEnabled = ssrOverride === 1
  || (ssrOverride < 0 && !godotReference && (interactiveQualityPreset < 0 || interactiveQualityPreset >= 3));
const diagnosticSharpenStrengths = [0.0, 0.25, 0.40, 0.45, 0.85];
const diagnosticSharpenStrength = interactiveSharpenStrength >= 0.0
  ? interactiveSharpenStrength
  : interactiveQualityPreset >= 0
    ? diagnosticSharpenStrengths[interactiveQualityPreset]
    : 0.5;
let diagnosticSharpenEnabled = diagnosticSharpenStrength > 0.0;
let diagnosticCaptureRequested = false;

// Bistro is an image-quality inspection scene. Ultra intentionally offers
// cinematic motion blur, but applying its 8-tap directional filter while the
// user free-looks makes texture detail appear softer than the stationary
// renderer actually is and obscures temporal-reconstruction defects. Keep the
// inspection default optically clean; --motion-blur 1 remains available to
// qualify the effect itself.
setMotionBlurEnabled(motionBlurOverride === 1);

// Keep the reference presentation optically clear by default. The Bistro
// already gets natural distance depth from its atmosphere sky and HDR IBL;
// layering exponential fog and screen-space shafts over it lifts nearby black
// levels and hides the material/texture detail this scene is meant to assess.
// Both effects remain explicit qualification modes via `--fog 1` and
// `--sun-shafts 1`.
setFog(0.92, 0.90, 0.84, fogOverride === 1 ? 0.006 : 0.0, 0.0, 0.05);
setSunShafts(sunShaftOverride === 1 ? 0.25 : 0.0, 0.96, 1.0, 0.94, 0.82);
setVignette(godotReference ? 0.0 : 0.10, 0.30);
setChromaticAberration(godotReference ? 0.0 : 0.0005);

// ---- Load Bistro into scene graph ----
// The rich launcher supplies the complete source exterior here. Each mesh
// entry represents an authored primitive placement while its immutable vertex
// and index payload remains shared with every repeated placement.
const bistro = loadModel(scenePath);
console.error("BLOOM_QUALITY_SCENE bistro_meshes=" + bistro.meshCount);
const identity = mat4Identity();
for (let i = 0; i < bistro.meshCount; i = i + 1) {
  const node = createSceneNode();
  attachModelToNode(node, bistro.handle, i);
  setSceneNodeTransform(node, identity);
}
console.error("BLOOM_QUALITY_SCENE bistro_placements_attached=" + bistro.meshCount);
if (pathTracingMode > 0) {
  setPathTracing(pathTracingMode);
}

// ---- Camera ----
// Matches the preset glTF camera in zeux/niagara_bistro (translation
// -26.43, 3.16, 11.17 aimed toward the bistro façade near the world
// origin). Gives a clean opening frame showing the signature corner
// with the lantern, awning, and cobble street.
let camX = initCamX;
let camY = initCamY;
let camZ = initCamZ;
let camYaw = (yawOverride || godotReference) ? initYaw : -1.17; // ≈ 67° left of -Z
let camPitch = pitchOverride ? initPitch : 0.0;
let cursorLocked = false;
let fixtureFrame = 0;

// ---- Main loop ----
while (!windowShouldClose()) {
  const qualityCapture = qualityRun !== null ? qualityRun.beginFrame() : false;
  const dt = qualityRun !== null ? qualityRun.deltaTime() : getDeltaTime();

  if (cursorLocked) {
    camYaw = camYaw - getMouseDeltaX() * MOUSE_SENS;
    camPitch = camPitch - getMouseDeltaY() * MOUSE_SENS;
    camPitch = clamp(camPitch, -1.4, 1.4);
  }

  const speed = isKeyDown(Key.LEFT_SHIFT) ? MOVE_SPEED * SPRINT_MULT : MOVE_SPEED;
  const fwdX = -Math.sin(camYaw);
  const fwdZ = -Math.cos(camYaw);
  const rightX = Math.cos(camYaw);
  const rightZ = -Math.sin(camYaw);

  if (isKeyDown(Key.W) || isKeyDown(Key.UP))    { camX = camX + fwdX * speed * dt; camZ = camZ + fwdZ * speed * dt; }
  if (isKeyDown(Key.S) || isKeyDown(Key.DOWN))   { camX = camX - fwdX * speed * dt; camZ = camZ - fwdZ * speed * dt; }
  if (isKeyDown(Key.A) || isKeyDown(Key.LEFT))   { camX = camX - rightX * speed * dt; camZ = camZ - rightZ * speed * dt; }
  if (isKeyDown(Key.D) || isKeyDown(Key.RIGHT))  { camX = camX + rightX * speed * dt; camZ = camZ + rightZ * speed * dt; }
  if (isKeyDown(Key.SPACE))        { camY = camY + speed * dt; }
  if (isKeyDown(Key.C))            { camY = camY - speed * dt; }

  if (isKeyPressed(Key.TAB)) {
    cursorLocked = !cursorLocked;
    if (cursorLocked) { disableCursor(); } else { enableCursor(); }
  }
  if (qualityRun === null && isKeyPressed(Key.T)) {
    diagnosticTaaEnabled = !diagnosticTaaEnabled;
    setTaaEnabled(diagnosticTaaEnabled);
  }
  if (qualityRun === null && isKeyPressed(Key.G)) {
    diagnosticSsgiEnabled = !diagnosticSsgiEnabled;
    setSsgiEnabled(diagnosticSsgiEnabled);
  }
  if (qualityRun === null && isKeyPressed(Key.R)) {
    diagnosticSsrEnabled = !diagnosticSsrEnabled;
    setSsrEnabled(diagnosticSsrEnabled);
  }
  // P is deliberately outside the WASD movement cluster. Binding this to S
  // made every fresh backward key press toggle post-process sharpening, so
  // ordinary navigation changed the apparent texture and shadow sharpness.
  if (qualityRun === null && isKeyPressed(Key.P)) {
    diagnosticSharpenEnabled = !diagnosticSharpenEnabled;
    setSharpenStrength(diagnosticSharpenEnabled ? diagnosticSharpenStrength : 0.0);
  }
  if (qualityRun === null && isKeyPressed(Key.N)) {
    diagnosticCaptureRequested = true;
  }

  // Opt-in VSM transition oracle. Move six metres along the exact sun
  // light-plane right vector every 30 frames, returning to the established
  // Bistro camera on frame 240 for a matched static/transition comparison.
  let renderCamX = camX;
  let renderCamZ = camZ;
  let renderYaw = camYaw;
  if (vsmMotionPath) {
    fixtureFrame = fixtureFrame + 1;
    // With --motion-frames, interpolate once from the launch pose to the
    // endpoint so the fixture crosses the same refit/cache boundaries as
    // interactive walking.  The default keeps the established 30-frame
    // square wave used by the original translation-only oracle.
    const pathStep = motionFrames > 0
      ? Math.min(fixtureFrame / motionFrames, 1.0)
      : Math.floor(fixtureFrame / 30) % 2;
    renderCamX = camX + pathStep * 3.748170285;
    renderCamZ = camZ + pathStep * 4.685212856;
    renderYaw = camYaw + pathStep * motionYaw;
  }
  const renderFwdX = -Math.sin(renderYaw);
  const renderFwdZ = -Math.cos(renderYaw);
  const lookX = renderCamX + Math.cos(camPitch) * renderFwdX * 100;
  const lookY = camY + Math.sin(camPitch) * 100;
  const lookZ = renderCamZ + Math.cos(camPitch) * renderFwdZ * 100;

  beginDrawing();

  setAmbientLight(
    godotReference ? { r: 255, g: 245, b: 232, a: 255 } : { r: 150, g: 160, b: 180, a: 255 },
    godotReference ? referenceAmbientIntensity : 0.3,
  );
  // Parisian afternoon sun — warm, angled slightly from the side.
  // 3.0 intensity gives a stronger sun-to-ambient ratio, matching
  // the Cycles reference's dominant directional light (Cycles uses
  // ~5 W/m² sun vs 1.2× HDR env — our ratio was previously too
  // flat, leaving sunlit and shaded surfaces in a narrow tonal band).
  setDirectionalLight(
    godotReference ? { x: 0.59732, y: 0.79653, z: -0.0935387 } : { x: -0.5, y: 0.75, z: 0.4 },
    godotReference ? { r: 255, g: 212, b: 177, a: 255 } : { r: 255, g: 240, b: 220, a: 255 },
    godotReference ? referenceSunIntensity : 3.0,
  );
  // Tiny fill from below — same trick as Sponza, keeps overhangs
  // and awnings from bottoming out when SSGI misses them.
  if (!godotReference) {
    addDirectionalLight(0.0, -1.0, 0.0, 0.5, 0.55, 0.7, 0.4);
  }

  beginMode3D({
    position: { x: renderCamX, y: camY, z: renderCamZ },
    target: { x: lookX, y: lookY, z: lookZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 60,
    projection: "perspective",
  });

  endMode3D();

  if (qualityRun === null) {
    drawText("Bloom Bistro", 10, 10, 20, { r: 255, g: 255, b: 255, a: 255 });
    const fps = getFPS();
    const ms = fps > 0.0 ? 1000.0 / fps : 0.0;
    const fpsColor = fps >= 55.0
      ? { r: 120, g: 230, b: 120, a: 255 }
      : fps >= 30.0
        ? { r: 230, g: 220, b: 120, a: 255 }
        : { r: 230, g: 120, b: 120, a: 255 };
    const fpsText = `FPS ${Math.round(fps)}  (${ms.toFixed(1)} ms)`;
    drawText(fpsText, 10, 35, 16, fpsColor);
    drawText("WASD move / Mouse look / Tab cursor / N capture", 10, SCREEN_H - 48, 14, { r: 180, g: 180, b: 180, a: 255 });
    const diagnosticText = `T TAA ${diagnosticTaaEnabled ? "on" : "off"} / G SSGI ${diagnosticSsgiEnabled ? "on" : "off"} / R SSR ${diagnosticSsrEnabled ? "on" : "off"} / P sharpen ${diagnosticSharpenEnabled ? "on" : "off"}`;
    drawText(diagnosticText, 10, SCREEN_H - 26, 13, { r: 180, g: 210, b: 240, a: 255 });
  }

  if (qualityRun === null && (captureFrames > 0 || diagnosticCaptureFrames > 0)) {
    frameCount = frameCount + 1;
    if (diagnosticCaptureFrames > 0 && frameCount === diagnosticCaptureFrames) {
      diagnosticCaptureRequested = true;
    }
  }

  if (qualityCapture && qualityRun !== null) {
    qualityRun.requestCapture();
  } else if (diagnosticCaptureRequested) {
    console.error(
      "BLOOM_BISTRO_CAPTURE_CAMERA"
        + " x=" + renderCamX.toFixed(6)
        + " y=" + camY.toFixed(6)
        + " z=" + renderCamZ.toFixed(6)
        + " yaw=" + renderYaw.toFixed(6)
        + " pitch=" + camPitch.toFixed(6),
    );
    captureDebugIntermediates("/tmp/bloom-bistro-ssgi-diagnostic");
    captureFrameToPng("/tmp/bloom-bistro-ssgi-diagnostic/frame.png");
    diagnosticCaptureRequested = false;
  } else if (qualityRun === null && captureFrames > 0) {
    if (frameCount === captureFrames) { takeScreenshot(capturePath); }
    if (frameCount > captureFrames) { endDrawing(); break; }
  }

  endDrawing();
  if (qualityRun !== null && qualityRun.endFrame()) break;
}

closeWindow();
