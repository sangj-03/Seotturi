"""Recording-condition augmentation for the 4-region classifier (region4.py).

What the site does to audio, and what tells the AI Hub sets apart besides the accent, measured on the
served model (one-off probe scripts, not included here; 2026-10-09): the background alone names the region well
above chance, near-silent gaps between words (browser noise suppression) and a denoiser break the model,
and so does the level. Each training copy therefore gets a random chain of:

  crop        a 1.5-4 s piece (short answers)
  bg          the background of a clip from ANOTHER set mixed in at 10-25 dB SNR (pink noise if none)
  eq          spectral tilt and a low-pass (other microphones)
  opus        an Opus round trip (uploaded m4a/webm recordings)
  denoise     spectral gating (stand-in for browser noise suppression; evaluation uses ffmpeg afftdn,
              which training never sees)
  gate        quiet frames turned down 20-50 dB (what aggressive suppression leaves between words)
  speech_only the non-speech runs cut out
  level       the voiced RMS set to -35 ... -12 dBFS (always; the site scales clips up to peak 0.9)

A copy's parameters are drawn once in region4.py and stored in its row ("aug"), so features.py can be
resumed and the copies rebuilt exactly.
"""

import subprocess
import wave

import numpy as np

SR, FR = 16000, 320  # 20 ms frames


def _load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def _speech_frames(x):
    n = len(x) // FR
    rms = np.sqrt((x[:n * FR].reshape(n, FR) ** 2).mean(1) + 1e-12)
    return n, rms, rms > max(np.percentile(rms, 15) * 3, 0.004)


def _runs(mask):
    """[start, end) of the True runs of a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _join(pieces):
    fade = np.linspace(0, 1, 160, dtype=np.float32)
    out = []
    for p in pieces:
        p = p.copy()
        if len(p) > 320:
            p[:160] *= fade
            p[-160:] *= fade[::-1]
        out.append(p)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def background(x, min_run=5, pad=5):
    """Non-speech runs (>= 100 ms, >= 100 ms away from speech) joined; None if under 0.5 s."""
    if len(x) < 10 * FR:
        return None
    n, _, sp = _speech_frames(x)
    near = np.convolve(sp.astype(int), np.ones(2 * pad + 1, int), "same") > 0
    y = _join([x[a * FR:b * FR] for a, b in _runs(~near) if b - a >= min_run])
    return y if len(y) >= SR // 2 else None


def speech_only(x, pad=3):
    if len(x) < 10 * FR:
        return x
    n, _, sp = _speech_frames(x)
    keep = np.convolve(sp.astype(int), np.ones(2 * pad + 1, int), "same") > 0
    y = _join([x[a * FR:b * FR] for a, b in _runs(keep)])
    return y if len(y) >= SR // 2 else x


def gate(x, db, k):
    if len(x) < 10 * FR:
        return x
    n, rms, _ = _speech_frames(x)
    f = x[:n * FR].reshape(n, FR).copy()
    f[rms <= max(np.percentile(rms, 15) * k, 0.004)] *= 10 ** (db / 20)
    return np.concatenate([f.reshape(-1), x[n * FR:]])


def _stft(x, n=512, hop=128):
    win = np.hanning(n).astype(np.float32)
    x = np.pad(x, (n // 2, n // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    return np.fft.rfft(frames, axis=1), win


def _istft(S, win, length, n=512, hop=128):
    frames = np.fft.irfft(S, n, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n, np.float32)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + n] += f
        norm[i * hop:i * hop + n] += win ** 2
    return (out / np.maximum(norm, 1e-6))[n // 2:n // 2 + length]


def denoise(x, alpha, floor):
    """Spectral gating: noise profile from the quietest 10 % of frames, gain smoothed over time."""
    if len(x) < 2048:
        return x
    S, win = _stft(x)
    mag = np.abs(S)
    energy = mag.sum(1)
    noise = mag[energy <= np.percentile(energy, 10)].mean(0) + 1e-8
    g = np.clip(1 - alpha * noise / (mag + 1e-8), floor, 1.0)
    g = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, "same"), 0, g)
    return _istft(S * g, win, len(x)).astype(np.float32)


def eq(x, tilt, lowpass):
    """Spectral tilt (dB per octave above 1 kHz) and a low-pass, applied in one FFT."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    gain_db = tilt * np.log2(np.maximum(f, 125) / 1000)
    gain = 10 ** (gain_db / 20) * (1 / (1 + (f / lowpass) ** 8))
    return np.fft.irfft(X * gain, len(x)).astype(np.float32)


def opus(x, kbps):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg", "pipe:1"], input=pcm, capture_output=True)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
                         input=enc.stdout, capture_output=True)
    y = np.frombuffer(dec.stdout, "<i2").astype(np.float32) / 32768
    return y[:len(x)] if len(y) >= SR // 2 else x


def pink(n, rng):
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 20)
    y = np.fft.irfft(w / np.sqrt(f), n).astype(np.float32)
    return y / (np.sqrt((y ** 2).mean()) + 1e-9)


def mix_background(x, path, snr, seed):
    bg = background(_load(path)) if path else None
    if bg is None:
        bg = pink(len(x), np.random.default_rng(seed))
    bg = np.tile(bg, int(np.ceil(len(x) / len(bg))))[:len(x)]
    _, rms, sp = _speech_frames(x)
    s = np.sqrt((rms[sp] ** 2).mean()) if sp.any() else np.sqrt((x ** 2).mean())
    b = np.sqrt((bg ** 2).mean()) + 1e-9
    return x + bg * (s / b) * 10 ** (-snr / 20)


def level(x, db):
    """Voiced RMS (louder half of the frames) to `db` dBFS."""
    if len(x) < 2 * FR:
        return x
    n = len(x) // FR
    e = np.sqrt((x[:n * FR].reshape(n, FR) ** 2).mean(1) + 1e-12)
    v = e[e >= np.percentile(e, 50)]
    return np.clip(x * 10 ** (db / 20) / (np.sqrt((v ** 2).mean()) + 1e-9), -1, 1).astype(np.float32)


def draw(rng, bg_paths):
    """Random parameters of one copy (at least one change besides the level)."""
    while True:
        s = {"level": round(float(rng.uniform(-35, -12)), 1)}
        if rng.random() < 0.3:
            s["crop"] = [round(float(rng.random()), 3), round(float(rng.uniform(1.5, 4.0)), 2)]
        if rng.random() < 0.3:
            s["bg"] = {"path": bg_paths[rng.integers(len(bg_paths))], "snr": round(float(rng.uniform(10, 25)), 1),
                       "seed": int(rng.integers(1 << 30))}
        if rng.random() < 0.3:
            s["eq"] = {"tilt": round(float(rng.uniform(-4, 4)), 2), "lowpass": int(rng.uniform(3500, 8000))}
        if rng.random() < 0.15:
            s["opus"] = int(rng.choice([12, 16, 24, 32]))
        if rng.random() < 0.35:
            s["denoise"] = {"alpha": round(float(rng.uniform(1.0, 2.5)), 2), "floor": round(float(rng.uniform(0.03, 0.3)), 3)}
        if rng.random() < 0.3:
            s["gate"] = {"db": round(float(rng.uniform(-50, -20)), 1), "k": round(float(rng.uniform(2, 4)), 2)}
        if rng.random() < 0.2:
            s["speech_only"] = True
        if len(s) > 1:
            return s


def apply(x, s):
    """One copy: crop -> background -> microphone -> codec -> suppression -> level (the browser's order)."""
    if "crop" in s:
        start, sec = s["crop"]
        n = int(sec * SR)
        if len(x) > n:
            i = int(start * (len(x) - n))
            x = x[i:i + n]
    if "bg" in s:
        x = mix_background(x, s["bg"]["path"], s["bg"]["snr"], s["bg"]["seed"])
    if "eq" in s:
        x = eq(x, s["eq"]["tilt"], s["eq"]["lowpass"])
    if "opus" in s:
        x = opus(x, s["opus"])
    if "denoise" in s:
        x = denoise(x, s["denoise"]["alpha"], s["denoise"]["floor"])
    if "gate" in s:
        x = gate(x, s["gate"]["db"], s["gate"]["k"])
    if s.get("speech_only"):
        x = speech_only(x)
    return level(x, s["level"])
