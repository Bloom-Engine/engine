import { FixedStepClock } from './fixed_step';

export interface GameLifecycle {
  init?: () => void;
  /** One simulation tick, numbered from 1. Use the supplied constant delta. */
  fixedUpdate?: (dt: number, tick: number) => void;
  /** Once per rendered frame, after fixed updates; delta is clamped. */
  update?: (dt: number) => void;
  /** Blend previous/current simulation state with alpha in [0, 1). */
  draw: (alpha: number) => void;
  cleanup?: () => void;
}

export interface GameLoopOptions {
  fixedStepSeconds?: number;
  maxFixedSteps?: number;
  maxFrameSeconds?: number;
}

/** Internal lifecycle state; the platform adapter owns begin/end drawing. */
export class GameLifecycleDriver {
  readonly clock: FixedStepClock;
  private game: GameLifecycle;
  private initialized = false;
  private disposed = false;

  constructor(game: GameLifecycle, options?: GameLoopOptions) {
    this.game = game;
    let step = 1 / 60;
    let maxSteps = 8;
    let maxFrame = 0.25;
    if (options !== undefined) {
      if (options.fixedStepSeconds !== undefined) step = options.fixedStepSeconds;
      if (options.maxFixedSteps !== undefined) maxSteps = options.maxFixedSteps;
      if (options.maxFrameSeconds !== undefined) maxFrame = options.maxFrameSeconds;
    }
    this.clock = new FixedStepClock(step, maxSteps, maxFrame);
  }

  initialize(): boolean {
    if (!this.clock.valid || this.disposed || this.initialized) return false;
    this.initialized = true;
    // Perry WASM routes obj.callback() through named method dispatch. Reading
    // the function first uses closure dispatch on both supported targets.
    const init = this.game.init;
    if (init !== undefined) init();
    return true;
  }

  frame(dt: number, shouldStop: () => boolean): boolean {
    if (!this.initialized || this.disposed || shouldStop()) return false;
    if (!this.clock.advance(dt)) return false;
    const firstTick = this.clock.ticks - this.clock.steps + 1;
    const fixedUpdate = this.game.fixedUpdate;
    for (let i = 0; i < this.clock.steps; i++) {
      if (fixedUpdate !== undefined) fixedUpdate(this.clock.stepSeconds, firstTick + i);
      if (shouldStop()) return false;
    }
    const update = this.game.update;
    if (update !== undefined) update(this.clock.deltaSeconds);
    if (shouldStop()) return false;
    const draw = this.game.draw;
    draw(this.clock.alpha);
    return !shouldStop();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const cleanup = this.game.cleanup;
    if (this.initialized && cleanup !== undefined) cleanup();
  }
}
