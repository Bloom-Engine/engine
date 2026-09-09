/**
 * Deterministic qualification-window helper used by Bloom's versioned scene
 * corpus. It is inert unless an example receives `--quality-run`.
 *
 * The helper intentionally captures on a frame after measurement. Screenshot
 * readback is synchronous on native platforms; including it in the measured
 * window would turn a correctness artifact into a fake CPU regression.
 */

import {
  captureDebugIntermediates,
  captureFrameToPng,
  getTime,
  isFrameCaptureReady,
  setPresentMode,
  setProfilerEnabled,
  setQualityPreset,
  setRenderScale,
  setTargetFPS,
  writeQualityTelemetry,
} from "../core/index";

export interface QualityRunConfig {
  warmupFrames: number;
  measuredFrames: number;
  fixedTimestep: number;
  outputPath: string;
  telemetryPath: string;
  intermediatesPath: string;
  qualityPreset: number;
  renderScale: number;
}

function finitePositive(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Parse the shared CLI contract:
 *   --quality-run WARMUP MEASURED FIXED_DT OUTPUT_PNG TELEMETRY_JSON INTERMEDIATES_DIR
 */
export function parseQualityRun(argv: string[]): QualityRunConfig | null {
  let qualityPreset = 3;
  let renderScale = 1.0;
  for (let i = 1; i < argv.length; i = i + 1) {
    if (argv[i] === "--quality-preset" && i + 1 < argv.length) {
      qualityPreset = Math.max(0, Math.min(4, Math.floor(parseFloat(argv[i + 1]))));
    } else if (argv[i] === "--render-scale" && i + 1 < argv.length) {
      renderScale = Math.max(0.5, Math.min(1.0, parseFloat(argv[i + 1])));
    }
  }
  for (let i = 1; i < argv.length; i = i + 1) {
    if (argv[i] === "--quality-run" && i + 5 < argv.length) {
      return {
        warmupFrames: Math.max(1, Math.floor(parseFloat(argv[i + 1]))),
        measuredFrames: Math.max(1, Math.floor(parseFloat(argv[i + 2]))),
        fixedTimestep: finitePositive(parseFloat(argv[i + 3]), 1 / 60),
        outputPath: argv[i + 4],
        telemetryPath: argv[i + 5],
        intermediatesPath: i + 6 < argv.length ? argv[i + 6] : "",
        qualityPreset,
        renderScale,
      };
    }
  }
  return null;
}

export class QualityRun {
  readonly config: QualityRunConfig;
  private frame = 0;
  private measurementStarted = false;
  private measurementFinished = false;
  private captureRequested = false;
  private screenshotSubmitted = false;
  private intermediatesSubmitted = false;
  private measurementStartSeconds = 0;
  private measurementWallMs = 0;
  private telemetryWritten = false;

  constructor(config: QualityRunConfig) {
    this.config = config;
    // AutoNoVsync is explicitly distinct from the normal FIFO default.
    setPresentMode(3);
    setTargetFPS(0);
    setQualityPreset(config.qualityPreset as any);
    setRenderScale(config.renderScale);
    setProfilerEnabled(false);
  }

  /** Fixed simulation delta for deterministic camera/animation sequences. */
  deltaTime(): number {
    return this.config.fixedTimestep;
  }

  /**
   * Call immediately before beginDrawing(). Returns true only for the extra
   * post-measurement frame that should be captured.
   */
  beginFrame(): boolean {
    if (!this.measurementStarted && this.frame >= this.config.warmupFrames) {
      setProfilerEnabled(true);
      this.measurementStarted = true;
      this.measurementStartSeconds = getTime();
    }
    if (this.measurementFinished && !this.captureRequested) {
      this.captureRequested = true;
      return true;
    }
    return false;
  }

  /** Call before endDrawing() on the frame for which beginFrame() was true. */
  requestCapture(): void {
    if (this.captureRequested && !this.screenshotSubmitted) {
      if (!this.intermediatesSubmitted && this.config.intermediatesPath.length > 0) {
        this.intermediatesSubmitted = captureDebugIntermediates(
          this.config.intermediatesPath,
        );
      }
      this.screenshotSubmitted = captureFrameToPng(this.config.outputPath);
    }
  }

  /**
   * Call immediately after endDrawing(). Returns true when the PNG and
   * telemetry have both been requested/written and the example should exit.
   */
  endFrame(): boolean {
    if (this.captureRequested) {
      // If the caller-side capture branch was lost by the native TypeScript
      // lowering, queue the readback here and render one additional frame.
      // A request made before beginDrawing() is consumed by that frame, so
      // this fallback is deterministic and still outside the timed window.
      if (!this.screenshotSubmitted) {
        if (!this.intermediatesSubmitted && this.config.intermediatesPath.length > 0) {
          this.intermediatesSubmitted = captureDebugIntermediates(
            this.config.intermediatesPath,
          );
        }
        this.screenshotSubmitted = captureFrameToPng(this.config.outputPath);
        return false;
      }
      if (!isFrameCaptureReady()) return false;
      console.error("BLOOM_QUALITY_DONE " + this.config.telemetryPath);
      return true;
    }

    this.frame = this.frame + 1;
    if (
      this.measurementStarted
      && !this.measurementFinished
      && this.frame >= this.config.warmupFrames + this.config.measuredFrames
    ) {
      this.measurementWallMs = (getTime() - this.measurementStartSeconds) * 1000;
      this.telemetryWritten = writeQualityTelemetry(
        this.config.telemetryPath,
        this.config.warmupFrames,
        this.config.measuredFrames,
        this.config.fixedTimestep,
        this.config.qualityPreset,
        this.config.renderScale,
        this.measurementWallMs,
      );
      if (!this.telemetryWritten) {
        console.error("BLOOM_QUALITY_ERROR telemetry-write-failed " + this.config.telemetryPath);
      }
      // Preserve the snapshot already serialized above but remove all
      // profiler work from the following screenshot frame.
      setProfilerEnabled(false);
      this.measurementFinished = true;
    }
    return false;
  }
}
