// Microphone -> 16 kHz mono -> clips cut at pauses (the dialect model was trained on 1-20 s
// utterances and Gemma 4 takes at most 30 s of audio), each handed to onSegment as it ends.

const RATE = 16000;
const FRAME = 320; // 20 ms
const PRE = 10; // keep 200 ms before speech starts
const SILENCE = 35; // a 700 ms pause ends a clip
const TAIL = 12; // keep 240 ms after the last voiced frame
const MAX = 750; // force a cut after 15 s, at the quietest moment of the last 4 s
const MIN_VOICED = 12; // ignore blips shorter than about 250 ms

class Downsampler {
  constructor(inRate) { this.ratio = inRate / RATE; this.pos = 0; this.acc = 0; this.cnt = 0; }
  push(x) {
    const out = new Float32Array(Math.ceil(x.length / this.ratio) + 1);
    let k = 0;
    for (let i = 0; i < x.length; i++) {
      this.acc += x[i]; this.cnt++; this.pos += 1;
      if (this.pos >= this.ratio) { out[k++] = this.acc / this.cnt; this.acc = 0; this.cnt = 0; this.pos -= this.ratio; }
    }
    return out.subarray(0, k);
  }
}

class Growable {
  constructor() { this.a = new Float32Array(RATE * 60); this.n = 0; }
  push(x) {
    if (this.n + x.length > this.a.length) {
      const b = new Float32Array(Math.max(this.a.length * 2, this.n + x.length));
      b.set(this.a.subarray(0, this.n));
      this.a = b;
    }
    this.a.set(x, this.n);
    this.n += x.length;
  }
  slice(s, e) { return this.a.slice(Math.max(0, s), Math.min(this.n, e)); }
}

export function supported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && (window.AudioContext || window.webkitAudioContext));
}

export class Recorder {
  constructor({ onSegment = () => {}, onLevel = () => {} } = {}) {
    this.onSegment = onSegment;
    this.onLevel = onLevel;
    this.seq = 0;
  }

  async start() {
    // Create the context inside the click (Safari only lets a user gesture start audio), then ask for the mic.
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx();
    const resumed = this.ctx.resume();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e) {
      this.ctx.close();
      throw e;
    }
    await resumed;
    this.buf = new Growable();
    this.down = new Downsampler(this.ctx.sampleRate);
    this.rms = [];
    this.noise = null;
    this.vadPos = 0;
    this.segStart = null;
    this.lastVoice = 0;
    this.voiced = 0;
    this.running = true;
    let src;
    try {
      src = this.ctx.createMediaStreamSource(this.stream);
    } catch {
      // older Firefox: the context must run at the microphone's own sample rate
      const rate = this.stream.getAudioTracks()[0].getSettings().sampleRate || 48000;
      await this.ctx.close();
      this.ctx = new Ctx({ sampleRate: rate });
      await this.ctx.resume();
      this.down = new Downsampler(this.ctx.sampleRate);
      src = this.ctx.createMediaStreamSource(this.stream);
    }
    this.mute = this.ctx.createGain();
    this.mute.gain.value = 0;
    this.mute.connect(this.ctx.destination);
    try {
      await this.ctx.audioWorklet.addModule("/js/pcm-worklet.js");
      this.node = new AudioWorkletNode(this.ctx, "pcm-tap");
      this.node.port.onmessage = (e) => this._push(e.data);
    } catch {
      this.node = this.ctx.createScriptProcessor(4096, 1, 1);
      this.node.onaudioprocess = (e) => this._push(new Float32Array(e.inputBuffer.getChannelData(0)));
    }
    src.connect(this.node);
    this.node.connect(this.mute);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    src.connect(this.analyser);
    const data = new Float32Array(this.analyser.fftSize);
    const tick = () => {
      if (!this.running) return;
      this.analyser.getFloatTimeDomainData(data);
      let s = 0;
      for (const v of data) s += v * v;
      this.onLevel(Math.sqrt(s / data.length));
      this.raf = requestAnimationFrame(tick);
    };
    tick();
    this.t0 = performance.now();
  }

  get seconds() { return this.buf ? this.buf.n / RATE : 0; }

  _push(x) {
    if (!this.running) return;
    this.buf.push(this.down.push(x));
    while (this.vadPos + FRAME <= this.buf.n) {
      const f = this.buf.a.subarray(this.vadPos, this.vadPos + FRAME);
      let s = 0;
      for (const v of f) s += v * v;
      this._frame(Math.sqrt(s / FRAME));
      this.vadPos += FRAME;
    }
  }

  _frame(rms) {
    const i = this.rms.length;
    this.rms.push(rms);
    if (this.noise === null) {
      if (i >= 14) this.noise = Math.max(1e-4, [...this.rms].sort((a, b) => a - b)[7]);
      return;
    }
    const speech = rms > Math.max(this.noise * 2.8, 0.006);
    if (!speech) this.noise = 0.97 * this.noise + 0.03 * rms;
    if (this.segStart === null) {
      if (speech) {
        this.segStart = Math.max(0, i - PRE) * FRAME;
        this.lastVoice = i;
        this.voiced = 1;
      }
      return;
    }
    if (speech) { this.voiced++; this.lastVoice = i; }
    if (i - this.lastVoice >= SILENCE) {
      this._emit(this.segStart, (this.lastVoice + TAIL) * FRAME);
      this.segStart = null;
    } else if (i - this.segStart / FRAME >= MAX) {
      let best = i - 10, bestV = Infinity;
      for (let k = Math.max(this.segStart / FRAME + 50, i - 200); k <= i - 10; k++) {
        let v = 0;
        for (let j = k - 4; j <= k + 4; j++) v += this.rms[j] || 0;
        if (v < bestV) { bestV = v; best = k; }
      }
      this._emit(this.segStart, best * FRAME);
      this.segStart = best * FRAME;
      this.voiced = MIN_VOICED;
    }
  }

  _emit(start, end) {
    const voiced = this.voiced;
    this.voiced = 0;
    if (voiced < MIN_VOICED || end - start < RATE * 0.5) return;
    this.onSegment({ seq: this.seq++, samples: this.buf.slice(start, end), start: start / RATE, end: end / RATE });
  }

  async stop() {
    if (!this.running) return;
    this.running = false;
    cancelAnimationFrame(this.raf);
    if (this.segStart !== null) {
      this._emit(this.segStart, Math.min(this.buf.n, (this.lastVoice + TAIL) * FRAME));
      this.segStart = null;
    }
    try { this.node.disconnect(); } catch { /* already gone */ }
    for (const t of this.stream.getTracks()) t.stop();
    try { await this.ctx.close(); } catch { /* closed */ }
  }

  // The user cancelled: stop like stop(), but hand nothing more to onSegment.
  async cancel() {
    this.onSegment = () => {};
    await this.stop();
  }
}

export function encodeWav(samples, rate = RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
