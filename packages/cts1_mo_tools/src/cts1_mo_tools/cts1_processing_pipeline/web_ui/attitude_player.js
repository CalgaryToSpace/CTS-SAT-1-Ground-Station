// Client-side playback for `attitude_view.AttitudePlayer`.
//
// The server sends every frame of the playback window once, when playback
// starts (`cts1Attitude.play`), and this loop then re-poses the scene and
// rewrites the readouts locally -- no server round trip per frame. That
// keeps playback smooth even when individual requests to the server fail
// (e.g. a rate-limiting proxy). The server is told where playback stopped
// (`pause`'s return value, or the "ended" event) and re-syncs its own state
// from there.
//
// The pose math mirrors `attitude_view.py` (`scene_from_body`,
// `_align_y_to`, `_rate_arrow_length`, `AttitudeView.update`); keep them in
// step.

(() => {
  if (window.cts1Attitude) return;

  const RATE_HEAD_LENGTH = 0.3;
  const DEG = Math.PI / 180;

  const matmul = (a, b) =>
    [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));

  const rotX = (a) => {
    const c = Math.cos(a), s = Math.sin(a);
    return [[1, 0, 0], [0, c, -s], [0, s, c]];
  };
  const rotY = (a) => {
    const c = Math.cos(a), s = Math.sin(a);
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  };
  const rotZ = (a) => {
    const c = Math.cos(a), s = Math.sin(a);
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  };
  const ORBIT_TO_SCENE = [[1, 0, 0], [0, -1, 0], [0, 0, -1]];

  const sceneFromBody = (roll, pitch, yaw) =>
    matmul(ORBIT_TO_SCENE, matmul(rotZ(yaw * DEG), matmul(rotY(pitch * DEG), rotX(roll * DEG))));

  const alignYTo = (d) => {
    const n = Math.hypot(...d);
    const y = d.map((c) => c / n);
    const ref = Math.abs(y[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const dot = ref[0] * y[0] + ref[1] * y[1] + ref[2] * y[2];
    let x = ref.map((r, i) => r - dot * y[i]);
    const xn = Math.hypot(...x);
    x = x.map((c) => c / xn);
    const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
    return [0, 1, 2].map((i) => [x[i], y[i], z[i]]);
  };

  const rateArrowLength = (norm) => Math.min(3.6, 2.2 + 0.9 * Math.log10(1 + norm));

  const fmt = (v, unit) => (v == null ? "?" : (v < 0 ? "-" : "+") + Math.abs(v).toFixed(2) + unit);

  // Upper bound for `t[i] <= x` (like Python's `bisect_right(...) - 1`).
  const lastAtOrBefore = (times, x) => {
    let lo = 0, hi = times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo - 1;
  };

  class Player {
    constructor(ids) {
      this.ids = ids;
      this.frames = null;
      this.index = 0;
      this.clock = null;
      this.timer = null;
      this.pacing = "Per beacon";
      this.speed = 2;
      this.skipGaps = true;
      this.maxHoldSec = 1.5;
      this.realTimeTickSec = 0.1;
      this.lastTickWall = 0;
      this.frameShownWall = 0;
    }

    get playing() {
      return this.timer !== null;
    }

    get count() {
      return this.frames ? this.frames.t.length : 0;
    }

    configure({ pacing, speed, skipGaps, maxHoldSec, realTimeTickSec }) {
      this.pacing = pacing;
      this.speed = speed;
      this.skipGaps = skipGaps;
      this.maxHoldSec = maxHoldSec;
      this.realTimeTickSec = realTimeTickSec;
      if (this.playing) this.startTimer();
    }

    play(frames, state) {
      this.frames = frames;
      this.configure(state);
      this.clock = state.clock;
      this.show(state.index);
      this.lastTickWall = performance.now() / 1000;
      this.frameShownWall = this.lastTickWall;
      this.startTimer();
    }

    startTimer() {
      if (this.timer !== null) clearInterval(this.timer);
      const ms = this.pacing === "Per beacon" ? 1000 / this.speed : 1000 * this.realTimeTickSec;
      this.timer = setInterval(() => this.tick(), ms);
    }

    stop() {
      if (this.timer !== null) clearInterval(this.timer);
      this.timer = null;
      return { index: this.index, clock: this.clock };
    }

    seek(index) {
      if (!this.frames) return;
      this.show(index);
    }

    tick() {
      const n = this.count;
      if (this.index >= n - 1) {
        emitEvent(this.ids.ended_event, this.stop());
        return;
      }
      if (this.pacing === "Per beacon") {
        this.show(this.index + 1);
        return;
      }
      const t = this.frames.t;
      const now = performance.now() / 1000;
      const elapsed = now - this.lastTickWall;
      this.lastTickWall = now;
      if (this.clock == null) this.clock = t[this.index];
      this.clock += elapsed * this.speed;
      if (this.skipGaps && now - this.frameShownWall >= this.maxHoldSec) {
        this.clock = Math.max(this.clock, t[this.index + 1]);
      }
      const index = lastAtOrBefore(t, this.clock);
      if (index !== this.index) this.show(index);
    }

    // Move to frame `index` (keeping the real-time clock inside its span, as
    // `AttitudePlayer._on_frame` does) and render it.
    show(index) {
      const t = this.frames.t;
      index = Math.max(0, Math.min(index, t.length - 1));
      this.index = index;
      const start = t[index];
      const end = index + 1 < t.length ? t[index + 1] : null;
      if (this.clock == null || !(start <= this.clock && (end == null || this.clock < end))) {
        this.clock = start;
      }
      this.frameShownWall = performance.now() / 1000;
      this.render();
    }

    render() {
      const ids = this.ids;
      const f = this.frames;
      const i = this.index;
      const scene = getElement(ids.scene);
      const store = mounted_app.elements;
      if (!scene || !store) return;

      const roll = f.roll[i], pitch = f.pitch[i], yaw = f.yaw[i];
      const rates = [f.wx[i], f.wy[i], f.wz[i]];
      const hasAttitude = roll != null && pitch != null && yaw != null;
      scene.rotate(ids.body, hasAttitude ? sceneFromBody(roll, pitch, yaw) : sceneFromBody(0, 0, 0));

      const texts = {};
      ["x", "y", "z"].forEach((axis, k) => {
        texts[ids.axis_labels[axis]] = `${axis.toUpperCase()}  ω${axis} ${fmt(rates[k], "°/s")}`;
      });
      const haveRates = rates.every((r) => r != null);
      const norm = haveRates ? Math.hypot(...rates) : 0;
      const showRate = norm > 0.01;
      if (showRate) {
        const length = rateArrowLength(norm);
        const shaft = length - RATE_HEAD_LENGTH;
        scene.rotate(ids.rate_arrow, alignYTo(rates));
        scene.scale(ids.shaft, 1, shaft, 1);
        scene.move(ids.head, 0, shaft + RATE_HEAD_LENGTH / 2, 0);
        scene.move(ids.rate_label, ...rates.map((c) => (c / norm) * (length + 0.25)));
        texts[ids.rate_label] = `ω ${norm.toFixed(2)}°/s`;
      }
      scene.visible(ids.rate_arrow, showRate);
      scene.visible(ids.rate_label, showRate);
      for (const [id, text] of Object.entries(texts)) {
        const obj = scene.objects?.get(id);
        if (obj?.element && obj.element.textContent !== text) obj.element.textContent = text;
      }

      // NiceGUI elements: edit the client-side element store, so Vue
      // re-renders them just as it would for a server update.
      const setText = (id, text) => {
        const el = store[id];
        if (el && el.text !== text) el.text = text;
      };
      const values = {
        Roll: fmt(roll, "°"),
        Pitch: fmt(pitch, "°"),
        Yaw: fmt(yaw, "°"),
        "ωx": fmt(rates[0], " °/s"),
        "ωy": fmt(rates[1], " °/s"),
        "ωz": fmt(rates[2], " °/s"),
        "Estimation mode": f.est[i],
        "Control mode": f.ctrl[i],
      };
      for (const [name, value] of Object.entries(values)) setText(ids.readout[name], value);
      setText(ids.caption, f.caption[i]);
      setText(ids.position, `Frame ${i + 1} of ${this.count}`);

      const noAttitude = store[ids.no_attitude];
      if (noAttitude) {
        const cls = (noAttitude.class ?? []).filter((c) => c !== "hidden");
        if (hasAttitude) cls.push("hidden");
        noAttitude.class = cls;
      }
      const slider = store[ids.slider];
      if (slider?.props) slider.props["model-value"] = i;
    }
  }

  const players = {};
  const get = (ids) => (players[ids.scene] ??= new Player(ids));

  window.cts1Attitude = {
    play(ids, frames, state) {
      get(ids).play(frames, state);
    },
    // Stop and report where playback got to (null if it never started).
    pause(sceneId) {
      const p = players[sceneId];
      return p && p.frames ? p.stop() : null;
    },
    seek(sceneId, index) {
      players[sceneId]?.seek(index);
    },
    configure(sceneId, state) {
      players[sceneId]?.configure(state);
    },
  };
})();
