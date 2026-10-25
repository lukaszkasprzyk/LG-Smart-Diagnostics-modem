#!/usr/bin/env python3
"""
LG Smart Diagnostics — signal visualisation + protocol parameter analysis.

Generates a 4-panel figure:
  1. Spectrogram (full recording)
  2. Per-tone energy over time
  3. Detected symbol timeline
  4. Tone frequency spectrum

Also prints a technical parameter report (symbol rate, tones, CRC, etc.)

Usage:
    python visualise_lg.py example_CF_fridge_GBB92MCB2P.m4a
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy import signal
import soundfile as sf

# Reuse decode_lg helpers
sys.path.insert(0, os.path.dirname(__file__))
from decode_lg import (
    load_audio, detect_tones, demodulate, symbols_to_bytes,
    crc8, find_crc_frames, NOMINAL_TONES, SYMBOL_RATE, BANDPASS_BW,
)

# ---------------------------------------------------------------------------
# Palette — 4 categorical hues, colorblind-safe (OKLab-verified)
# ---------------------------------------------------------------------------
TONE_COLORS = ["#3B82F6", "#F59E0B", "#10B981", "#EF4444"]  # blue, amber, green, red
TONE_LABELS = ["F6 ~1397 Hz", "G6 ~1568 Hz", "A6 ~1760 Hz", "C7 ~2093 Hz"]
SYM_CMAP   = matplotlib.colors.ListedColormap(TONE_COLORS)


def _bandpass(center, bw, fs, order=4):
    lo = max((center - bw) / (fs / 2), 1e-4)
    hi = min((center + bw) / (fs / 2), 1 - 1e-4)
    return signal.butter(order, [lo, hi], btype="band")


def compute_tone_energies(audio, fs, tones):
    energies = []
    for t in tones:
        b, a = _bandpass(t, BANDPASS_BW, fs)
        filtered = signal.lfilter(b, a, audio)
        energies.append(filtered ** 2)
    return np.array(energies)


def smooth(x, win):
    return np.convolve(x, np.ones(win) / win, mode="same")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_signal(audio, fs, tones, symbols, onset_s, end_s, out_path):
    sym_len   = int(round(fs / SYMBOL_RATE))
    energies  = compute_tone_energies(audio, fs, tones)
    t_axis    = np.arange(len(audio)) / fs
    duration  = len(audio) / fs

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [2.5, 1.8, 0.6, 1.8]},
    )
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#1F2937")
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    def title(ax, text):
        ax.set_title(text, color="#F9FAFB", fontsize=9, loc="left", pad=4)

    def xlabel(ax, text):
        ax.set_xlabel(text, color="#6B7280", fontsize=8)

    def ylabel(ax, text):
        ax.set_ylabel(text, color="#6B7280", fontsize=8)

    # ── 1. Spectrogram ─────────────────────────────────────────────────────
    ax = axes[0]
    f_sg, t_sg, Sxx = signal.spectrogram(
        audio, fs, nperseg=1024, noverlap=768,
        window="hann", scaling="spectrum",
    )
    mask = (f_sg >= 1000) & (f_sg <= 2500)
    db = 10 * np.log10(Sxx[mask] + 1e-12)
    vmin, vmax = np.percentile(db, [5, 99])
    ax.pcolormesh(t_sg, f_sg[mask], db, cmap="magma", vmin=vmin, vmax=vmax,
                  shading="auto", rasterized=True)
    ax.axvline(onset_s, color="#6EE7B7", lw=1, ls="--", alpha=0.8, label="signal start")
    ax.axvline(end_s,   color="#FCA5A5", lw=1, ls="--", alpha=0.8, label="signal end")
    for tone in tones:
        ax.axhline(tone, color="#FBBF24", lw=0.5, alpha=0.5)
    ax.legend(fontsize=7, framealpha=0.3, facecolor="#111827", labelcolor="#E5E7EB",
              loc="upper right")
    title(ax, "Spectrogram (1000–2500 Hz)")
    ylabel(ax, "Frequency (Hz)")
    ax.set_xlim(0, duration)
    ax.yaxis.set_major_locator(MultipleLocator(200))

    # ── 2. Per-tone energy ─────────────────────────────────────────────────
    ax = axes[1]
    win = sym_len // 4
    t_en = t_axis
    for i, (e, col, lbl) in enumerate(zip(energies, TONE_COLORS, TONE_LABELS)):
        e_smooth = smooth(e, win)
        e_norm   = e_smooth / (e_smooth.max() + 1e-12)
        ax.plot(t_en, e_norm + i * 1.15, color=col, lw=0.8, alpha=0.9, label=lbl)
    ax.axvline(onset_s, color="#6EE7B7", lw=1, ls="--", alpha=0.6)
    ax.axvline(end_s,   color="#FCA5A5", lw=1, ls="--", alpha=0.6)
    ax.legend(fontsize=7, framealpha=0.3, facecolor="#111827", labelcolor="#E5E7EB",
              ncol=4, loc="upper right")
    title(ax, "Per-tone energy (normalised, offset for clarity)")
    ylabel(ax, "Energy (a.u.)")
    ax.set_xlim(0, duration)
    ax.set_yticks([])

    # ── 3. Symbol timeline ─────────────────────────────────────────────────
    ax = axes[2]
    sym_array = np.array(symbols, dtype=float).reshape(1, -1)
    t_sym_start = onset_s
    t_sym_end   = onset_s + len(symbols) * (1 / SYMBOL_RATE)
    ax.imshow(
        sym_array, aspect="auto", cmap=SYM_CMAP, vmin=0, vmax=3,
        extent=[t_sym_start, t_sym_end, 0, 1],
        interpolation="nearest",
    )

    # Frame structure annotation (from LG patent AU2010269313B2)
    # 16 sym preamble + 32 sym header (FEC 1/2) + rest = payload
    sym_t = 1 / SYMBOL_RATE
    t_pre_end     = onset_s + 16 * sym_t
    t_header_end  = onset_s + 48 * sym_t
    regions = [
        (onset_s,      t_pre_end,    "#6EE7B7", "Preamble\n16 sym"),
        (t_pre_end,    t_header_end, "#FCD34D", "Header\n32 sym (FEC 1/2)"),
        (t_header_end, t_sym_end,    "#93C5FD", f"Payload\n{len(symbols)-48} sym"),
    ]
    for x0, x1, col, label in regions:
        ax.axvspan(x0, x1, ymin=0, ymax=1, alpha=0.22, color=col, zorder=2)
        ax.axvline(x0, color=col, lw=1, alpha=0.7, zorder=3)
        mid = (x0 + x1) / 2
        ax.text(mid, 1.08, label, color=col, fontsize=6.5, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(), zorder=4)

    ax.set_xlim(0, duration)
    ax.set_yticks([])
    title(ax, f"Symbol timeline  ({len(symbols)} symbols,  {SYMBOL_RATE:.2f} sym/s)")
    patches = [mpatches.Patch(color=TONE_COLORS[i], label=f"sym {i}") for i in range(4)]
    ax.legend(handles=patches, fontsize=7, framealpha=0.3, facecolor="#111827",
              labelcolor="#E5E7EB", ncol=4, loc="upper right")

    # ── 4. FFT of active region ────────────────────────────────────────────
    ax = axes[3]
    seg = audio[int(onset_s * fs): int(end_s * fs)]
    N   = len(seg)
    fft = np.abs(np.fft.rfft(seg, n=N)) / N
    fft_f = np.fft.rfftfreq(N, 1 / fs)
    mask2 = (fft_f >= 1100) & (fft_f <= 2400)
    ax.fill_between(fft_f[mask2], fft[mask2], color="#374151", linewidth=0)
    ax.plot(fft_f[mask2], fft[mask2], color="#6B7280", lw=0.6)
    for tone, col, lbl in zip(tones, TONE_COLORS, TONE_LABELS):
        idx  = np.argmin(np.abs(fft_f - tone))
        peak = fft[idx]
        ax.axvline(tone, color=col, lw=1, alpha=0.8)
        ax.annotate(f"{tone:.0f} Hz", xy=(tone, peak),
                    xytext=(tone + 10, peak * 1.1),
                    color=col, fontsize=7.5, va="bottom")
    title(ax, "Frequency spectrum of active signal region")
    xlabel(ax, "Frequency (Hz)")
    ylabel(ax, "Magnitude")
    ax.set_xlim(1100, 2400)

    # ── shared x label ─────────────────────────────────────────────────────
    for ax in axes[:3]:
        ax.set_xlabel("")
    axes[2].set_xlabel("Time (s)", color="#6B7280", fontsize=8)

    fig.suptitle("LG Smart Diagnostics — signal analysis", color="#F9FAFB",
                 fontsize=11, y=0.995)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Figure saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Technical parameter report
# ---------------------------------------------------------------------------

def print_tech_report(audio, fs, tones, symbols, onset_s, end_s):
    duration  = len(audio) / fs
    sym_len   = int(round(fs / SYMBOL_RATE))
    data      = symbols_to_bytes(symbols)
    frames    = find_crc_frames(data)

    # Measured symbol rate from actual timing
    measured_rate = len(symbols) / (end_s - onset_s) if end_s > onset_s else 0

    # Tone spacing
    spacings = [tones[i+1] - tones[i] for i in range(len(tones)-1)]

    # Symbol distribution
    from collections import Counter
    dist = Counter(symbols)
    entropy = -sum((c/len(symbols)) * np.log2(c/len(symbols))
                   for c in dist.values() if c > 0)

    print()
    print("=" * 58)
    print("  TECHNICAL PARAMETER REPORT")
    print("=" * 58)
    print(f"  Recording duration    : {duration:.2f} s")
    print(f"  Signal window         : {onset_s:.2f} s – {end_s:.2f} s")
    print(f"  Signal duration       : {end_s - onset_s:.2f} s")
    print()
    print("  MODULATION")
    print(f"  Type                  : 4-MFSK (Four-tone Frequency Shift Keying)")
    print(f"  Grey coding           : Yes (00/01/11/10 low→high)")
    print(f"  Bits per symbol       : 2")
    print()
    print("  TONES (auto-detected)")
    for i, (t, nom) in enumerate(zip(tones, NOMINAL_TONES)):
        offset = t - nom
        print(f"  Symbol {i} (nominal {nom:.0f} Hz): {t:.1f} Hz  (offset {offset:+.1f} Hz)")
    print(f"  Tone spacings         : {[f'{s:.0f}' for s in spacings]} Hz")
    print()
    print("  TIMING")
    print(f"  Nominal symbol rate   : {SYMBOL_RATE:.2f} sym/s")
    print(f"  Measured symbol rate  : {measured_rate:.2f} sym/s")
    print(f"  Symbol period         : {1000/SYMBOL_RATE:.2f} ms  ({sym_len} samples @ {fs} Hz)")
    print(f"  Total symbols decoded : {len(symbols)}")
    print(f"  Total bits            : {len(symbols)*2}")
    print(f"  Total bytes           : {len(data)}")
    print()
    print("  SYMBOL STATISTICS")
    for s in range(4):
        pct = 100 * dist[s] / len(symbols)
        print(f"  Symbol {s}              : {dist[s]:4d} ({pct:.1f}%)")
    print(f"  Entropy               : {entropy:.2f} bit/symbol  (max 2.00)")
    print()
    print("  FRAMING / CRC")
    if frames:
        print(f"  CRC-8 polynomial      : 0x07  (init 0xFF)")
        print(f"  CRC verified frames   : {len(frames)}")
        for f in frames:
            print(f"    offset={f['offset']:2d} len={f['length']:2d}  "
                  f"payload={f['payload'].hex(' ')}  crc=0x{f['crc']:02x}")
    else:
        print("  CRC-8                 : no verified frames found")
    print()
    print("  RAW PAYLOAD")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        h = " ".join(f"{b:02x}" for b in chunk)
        a = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}:  {h:<48}  {a}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualise LG Smart Diagnostics signal and report protocol parameters."
    )
    ap.add_argument("file", help="Audio recording (.wav, .m4a, …)")
    ap.add_argument("--out", default="", help="Output PNG path (default: <file>.png)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    out_path = args.out or os.path.splitext(args.file)[0] + "_analysis.png"

    print(f"Loading: {args.file}")
    audio, fs = load_audio(args.file)
    print(f"Duration: {len(audio)/fs:.2f} s  Sample rate: {fs} Hz")

    tones = detect_tones(audio, fs)
    print(f"Auto-detected tones: {[f'{t:.1f}' for t in tones]} Hz")

    symbols, onset_s, end_s = demodulate(audio, fs, tones)
    print(f"Signal: {onset_s:.2f} s – {end_s:.2f} s  ({len(symbols)} symbols)")

    print_tech_report(audio, fs, tones, symbols, onset_s, end_s)
    plot_signal(audio, fs, tones, symbols, onset_s, end_s, out_path)


if __name__ == "__main__":
    main()
