import { isFiniteNumber } from './numbers';

/** Bounded fixed-step timing. All time values are in seconds. */
export class FixedStepClock {
  readonly stepSeconds: number;
  readonly maxSteps: number;
  readonly maxFrameSeconds: number;
  readonly valid: boolean;
  steps = 0;
  ticks = 0;
  alpha = 0;
  deltaSeconds = 0;
  droppedSeconds = 0;
  private remainder = 0;

  constructor(stepSeconds: number = 1 / 60, maxSteps: number = 8, maxFrameSeconds: number = 0.25) {
    this.stepSeconds = stepSeconds;
    this.maxSteps = maxSteps;
    this.maxFrameSeconds = maxFrameSeconds;
    this.valid = isFiniteNumber(stepSeconds) && stepSeconds > 0 &&
      isFiniteNumber(maxSteps) && maxSteps >= 1 && maxSteps <= 1000 && Math.floor(maxSteps) === maxSteps &&
      isFiniteNumber(maxFrameSeconds) && maxFrameSeconds > 0;
  }

  /** Returns false for invalid configuration or delta; no simulation time advances. */
  advance(deltaSeconds: number): boolean {
    this.steps = 0;
    this.deltaSeconds = 0;
    if (!this.valid || !isFiniteNumber(deltaSeconds) || deltaSeconds < 0) return false;
    const accepted = Math.min(deltaSeconds, this.maxFrameSeconds);
    const combined = this.remainder + accepted;
    if (!isFiniteNumber(combined)) return false;
    this.deltaSeconds = accepted;
    this.droppedSeconds = Math.min(1.7976931348623157e308, this.droppedSeconds + (deltaSeconds - accepted));
    this.remainder = combined;
    // A tiny tolerance prevents an exact tick partition (e.g. 3 * 0.01) from
    // missing its boundary solely because binary floating point rounded down.
    const tolerance = this.stepSeconds * 1e-9;
    while (this.steps < this.maxSteps && this.stepSeconds - this.remainder <= tolerance) {
      this.remainder = Math.max(0, this.remainder - this.stepSeconds);
      this.steps = this.steps + 1;
    }
    if (this.stepSeconds - this.remainder <= tolerance) {
      const remainder = this.remainder % this.stepSeconds;
      this.droppedSeconds = Math.min(1.7976931348623157e308, this.droppedSeconds + (this.remainder - remainder));
      this.remainder = remainder;
      if (this.stepSeconds - this.remainder <= tolerance) {
        this.droppedSeconds = Math.min(1.7976931348623157e308, this.droppedSeconds + this.remainder);
        this.remainder = 0;
      }
    }
    this.alpha = this.remainder / this.stepSeconds;
    this.ticks = this.ticks + this.steps;
    return true;
  }
}
