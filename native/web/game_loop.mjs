/** Schedule game frames and dispose their resources after the final frame. */
export function createGameLoop({ requestFrame, cancelFrame, beginFrame, endFrame,
  update, cleanup, onError }) {
  let current = null;

  function finish(session) {
    if (session.finished) return;
    session.finished = true;
    if (current === session) current = null;
    if (session.cleanup !== null) {
      try { cleanup(session.cleanup); }
      catch (error) { onError(error); }
    }
  }

  function stop() {
    const session = current;
    if (session === null) return;
    session.running = false;
    if (session.request !== null) {
      cancelFrame(session.request);
      session.request = null;
    }
    if (!session.inFrame) finish(session);
  }

  function schedule(session) {
    session.request = requestFrame(() => {
      if (current !== session || !session.running) return;
      session.request = null;
      session.inFrame = true;
      let began = false;
      try {
        beginFrame();
        began = true;
        update(session.callback);
      } catch (error) {
        session.running = false;
        onError(error);
      } finally {
        if (began) {
          try { endFrame(); }
          catch (error) { session.running = false; onError(error); }
        }
        session.inFrame = false;
        if (session.running) schedule(session);
        else finish(session);
      }
    });
  }

  return {
    get running() { return current !== null && current.running; },
    start(callback, cleanupCallback = null) {
      if (current !== null) {
        throw new Error('Bloom game loop is already active; stop it before starting another.');
      }
      const session = { callback, cleanup: cleanupCallback, running: true,
        inFrame: false, finished: false, request: null };
      current = session;
      schedule(session);
    },
    stop,
  };
}
