#!/usr/bin/env python3
"""
LG Smart Diagnostics acoustic decoder

Supports two LG Smart Diagnosis modulations:
  - 4-MFSK  ~1400–2100 Hz  refrigerators (newer models)
  - BFSK    ~2600/2840 Hz  washing machines, dishwashers (older models)

Symbol rate: ~23.89 sym/s (~41.9 ms per symbol) for both variants.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

import numpy as np
from scipy import signal
import soundfile as sf


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

NOMINAL_TONES_4MFSK = [1396.91, 1567.98, 1760.00, 2093.00]  # refrigerators
NOMINAL_TONES_BFSK  = [2600.00, 2840.00]                     # washers / dishwashers
NOMINAL_TONES       = NOMINAL_TONES_4MFSK                    # backward compat alias

SYMBOL_RATE  = 23.89   # sym/s
BANDPASS_BW  = 80      # Hz half-width per filter
ONSET_THRESH = 0.05    # fraction of peak energy to mark signal start
GREY = {0: 0b00, 1: 0b01, 2: 0b11, 3: 0b10}


# ---------------------------------------------------------------------------
# LG error code database
# Source: LG support pages + service documentation
# Format: display_code -> (short, description, suggested_action)
# ---------------------------------------------------------------------------

ERROR_CODES = {
    # --- Refrigerator: Fans ---
    "CF":   ("Condenser fan",        "Condenser fan feedback signal <65 s during operation",
             "Check fan at rear/bottom for obstruction or moisture; may need service"),
    "FF":   ("Freezer fan",          "Freezer evaporator fan not running",
             "Unplug 24 h with doors open to defrost; if persists, fan motor needs replacement"),
    "rF":   ("Fridge fan",           "Refrigerator evaporator fan abnormality",
             "Unplug 24 h with doors open to defrost; if persists, fan motor needs replacement"),
    "IF":   ("Ice maker fan",        "Ice maker fan issue",
             "Manual defrost (unplug 8 h, doors open); if persists, service required"),
    "F1":   ("Ice maker fan",        "Ice maker fan issue (alternate code)",
             "Manual defrost (unplug 8 h, doors open); if persists, service required"),
    # --- Refrigerator: Temperature sensors ---
    "FS":   ("Freezer sensor",       "Freezer temperature sensor open or short",              "Service required"),
    "rS":   ("Fridge sensor",        "Refrigerator temperature sensor open or short",         "Service required"),
    "dS":   ("Defrost sensor",       "Defrost sensor open or short (freezer)",                "Service required"),
    "rT":   ("Room temp sensor",     "Room temperature sensor open or short",                  "Service required"),
    "SS":   ("Pantry sensor",        "Pantry/convertible zone sensor open or short",           "Service required"),
    "HS":   ("Humidity sensor",      "Humidity sensor open or short",                          "Service required"),
    "EI":   ("Ice maker sensor",     "Ice maker temperature sensor failure",                   "Service required"),
    # --- Refrigerator: Defrost ---
    "dH":   ("Defrost timeout",      "Unit spent >1 h attempting defrost",
             "Unplug 2 min to reset; if recurring, defrost heater or thermal fuse may be faulty"),
    # --- Refrigerator: Ice maker ---
    "IS":   ("Ice maker motor",      "Ice maker fan motor issue or sensor disconnection",      "Service required"),
    "iT":   ("Ice maker defect",     "Ice maker defect",                                       "Service required"),
    "gF":   ("Flow sensor",          "Flow sensor error or low water pressure",                "Check water supply; service if flow sensor faulty"),
    # --- Refrigerator: Communication ---
    "CO":   ("Control board comms",  "Main board ↔ display communication failure",            "Service required"),
    "C1":   ("Display comms",        "LCD display ↔ dispenser communication failure",          "Service required"),
    "Od":   ("WiFi modem",           "WiFi modem communication error",                         "Service required"),
    # --- Refrigerator: Door / motor ---
    "Ad":   ("Door motor",           "Door motor did not reach position within 5 s",           "Service required"),
    "dr":   ("Door motor",           "Door motor positioning failure (same as Ad)",             "Service required"),
    "AS":   ("Door reed switch",     "Auto-close door motor reed switch open during operation","Service required"),
    "67":   ("Door gap",             "Door gap or obstruction preventing proper closure",
             "Check door seal and hinge; clear any obstructions"),
    # --- Refrigerator: Compressor / refrigerant ---
    "22":   ("Compressor relay",     "Compressor relay problem",                               "Service required"),
    "33":   ("Compressor relay",     "Compressor relay problem",                               "Service required"),
    "CE":   ("Refrigerant leak (low)","Low-side refrigerant leak (evaporator)",                "Service required"),
    "CH":   ("Refrigerant leak (hi)", "High-side refrigerant leak (condenser)",                "Service required"),
    # --- Refrigerator: Mode indicators ---
    "OF":   ("Demo mode",            "Demo/Display mode active — cooling disabled",
             "Hold Fridge + Ice Plus 3 s to exit Demo mode"),
    "88":   ("Self-test OK",         "Normal operation confirmed",                             "No action needed"),
    "Sb":   ("Sabbath mode",         "Sabbath mode active — alarms and dispenser disabled",    "Auto-disables after 96 h"),
    # --- Washer / Dishwasher error codes ---
    "dE":   ("Door",                 "Door not properly closed or door latch fault",
             "Check door seal and latch; restart cycle"),
    "dE1":  ("Door (alt)",           "Door switch fault (front-load)",                         "Service required"),
    "dE2":  ("Door (alt)",           "Door switch fault (top-load)",                           "Service required"),
    "LE":   ("Motor lock",           "Motor cannot rotate — overload or blocked drum",
             "Remove excess load; check for foreign objects in drum"),
    "UE":   ("Unbalanced",           "Load unbalanced — machine cannot spin",
             "Redistribute laundry evenly; reduce load size"),
    "OE":   ("Drain",                "Water not draining within expected time",
             "Check drain hose for kink or blockage; clean pump filter"),
    "IE":   ("Inlet",                "Water not filling within expected time",
             "Check water supply valves are open; check inlet filter screens"),
    "FE":   ("Overfill",             "Water level sensor detects overfill",                    "Service required"),
    "PE":   ("Water level sensor",   "Water level/pressure sensor fault",                      "Service required"),
    "tE":   ("Thermistor",           "Water temperature sensor (thermistor) fault",             "Service required"),
    "tE1":  ("Heater sensor",        "Heater temperature sensor fault",                         "Service required"),
    "HE":   ("Heater",               "Heater fault — water not reaching target temperature",    "Service required"),
    "HE1":  ("Heater (alt)",         "Heater element or relay fault",                           "Service required"),
    "CE1":  ("Current sensor",       "Motor current sensor fault",                              "Service required"),
    "PF":   ("Power failure",        "Power outage detected mid-cycle",
             "Restart cycle; if recurring check power supply stability"),
    "E6":   ("Clutch",               "Clutch fault (top-load)",                                 "Service required"),
    "nE":   ("Noise sensor",         "Anti-vibration noise sensor fault",                       "Service required"),
    "Cd":   ("Cool down",            "Cool-down phase active (not an error — dryer code)",      "Wait for cycle to finish"),
    "SU":   ("Excess suds",          "Too much detergent — suds reduction cycle added",
             "Reduce detergent amount; use HE-rated detergent"),
    "CL":   ("Child lock",           "Child lock active — not an error",
             "Hold designated button(s) 3 s to toggle"),
    "tC":   ("Tub clean reminder",   "Time to run tub clean cycle",
             "Run tub clean cycle with washing machine cleaner"),
    "AE":   ("Gear fault",           "Gears for drum rotation cannot operate (top-load)",       "Service required"),
    "SE":   ("WashTower sensor",     "Sensor fault — WashTower model",                          "Service required"),
    "dH":   ("Dryer heat",           "Dryer heating fault — WashTower model",                   "Service required"),
}

NIBBLE_CHAR = {**{i: str(i) for i in range(10)},
               10: 'A', 11: 'b', 12: 'C', 13: 'd', 14: 'E', 15: 'F'}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float64, converting via ffmpeg if needed."""
    if not path.lower().endswith(".wav"):
        path = _convert_to_wav(path, target_sr)
        audio, fs = sf.read(path)
        os.unlink(path)
    else:
        audio, fs = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return audio, fs


def _convert_to_wav(path: str, target_sr: int) -> str:
    ext = os.path.splitext(path)[1] or ".audio"
    tmp_in  = tempfile.mktemp(suffix=ext)
    tmp_out = tempfile.mktemp(suffix=".wav")
    shutil.copy2(path, tmp_in)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ac", "1", "-ar", str(target_sr), tmp_out],
            capture_output=True,
        )
    finally:
        os.unlink(tmp_in)
    if result.returncode != 0:
        sys.exit(f"ffmpeg error: {result.stderr.decode()}")
    return tmp_out


# ---------------------------------------------------------------------------
# Tone detection + modulation auto-detect
# ---------------------------------------------------------------------------

def detect_tones(audio: np.ndarray, fs: int, search_range=(1200, 3200)) -> list[float]:
    """
    Auto-detect carrier frequencies from the spectrum.
    Returns 2 tones (BFSK) or 4 tones (4-MFSK) depending on what's found.
    Falls back to 4-MFSK nominal tones if detection fails.
    """
    fft   = np.abs(np.fft.rfft(audio, n=len(audio)))
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    mask  = (freqs >= search_range[0]) & (freqs <= search_range[1])
    lf, lm = freqs[mask], fft[mask]

    min_dist = int(50 / (fs / len(audio)))
    peaks, _ = signal.find_peaks(lm, distance=max(1, min_dist), height=lm.max() * 0.05)
    if len(peaks) < 2:
        return NOMINAL_TONES_4MFSK[:]

    # Heuristic: if all strong peaks are above 2200 Hz → likely BFSK washer/dishwasher
    top4_idx  = peaks[np.argsort(lm[peaks])[-4:]]
    top4_freq = sorted([float(lf[p]) for p in top4_idx])
    all_high  = all(f > 2200 for f in top4_freq)

    if all_high:
        # BFSK: take the 2 highest-energy peaks
        top2_idx  = peaks[np.argsort(lm[peaks])[-2:]]
        top2_freq = sorted([float(lf[p]) for p in top2_idx])
        return top2_freq

    # 4-MFSK: take top 4
    if len(peaks) >= 4:
        return [float(lf[p]) for p in sorted(top4_idx)]

    return NOMINAL_TONES_4MFSK[:]


def is_bfsk(tones: list[float]) -> bool:
    return len(tones) == 2


# ---------------------------------------------------------------------------
# Demodulator
# ---------------------------------------------------------------------------

def _bandpass(center: float, bw: float, fs: int, order: int = 4):
    lo = max((center - bw) / (fs / 2), 1e-4)
    hi = min((center + bw) / (fs / 2), 1 - 1e-4)
    return signal.butter(order, [lo, hi], btype="band")


def demodulate(audio: np.ndarray, fs: int, tones: list[float],
               onset_threshold: float = ONSET_THRESH) -> tuple[list[int], float, float]:
    """Return (symbols, onset_sec, end_sec) for the active signal region."""
    sym_len = int(round(fs / SYMBOL_RATE))

    filters  = [_bandpass(t, BANDPASS_BW, fs) for t in tones]
    energies = np.array([signal.lfilter(b, a, audio) ** 2 for b, a in filters])

    # Onset detection
    total    = energies.sum(axis=0)
    smoothed = np.convolve(total, np.ones(sym_len) / sym_len, mode="same")
    thresh   = onset_threshold * smoothed.max()
    active   = np.where(smoothed > thresh)[0]
    if len(active) == 0:
        sys.exit("No LG Smart Diagnostics signal detected in recording.")

    onset = int(active[0])
    end   = int(active[-1])

    symbols: list[int] = []
    pos = onset + sym_len // 2
    while pos + sym_len // 2 <= end:
        s   = max(0, pos - sym_len // 2)
        e   = min(len(audio), pos + sym_len // 2)
        sym = int(np.argmax(energies[:, s:e].mean(axis=1)))
        symbols.append(sym)
        pos += sym_len

    return symbols, onset / fs, end / fs


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def symbols_to_bytes(symbols: list[int]) -> bytes:
    """4-MFSK: 2 bits per symbol via Grey code."""
    bits: list[int] = []
    for s in symbols:
        g = GREY[s]
        bits.append((g >> 1) & 1)
        bits.append(g & 1)
    raw = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        raw.append(byte)
    return bytes(raw)


def symbols_to_bytes_bfsk(symbols: list[int]) -> bytes:
    """BFSK: 1 bit per symbol, direct mapping."""
    raw = bytearray()
    for i in range(0, len(symbols) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (symbols[i + j] & 1)
        raw.append(byte)
    return bytes(raw)


def crc8(data: bytes, poly: int = 0x07, init: int = 0xFF) -> int:
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def find_crc_frames(data: bytes) -> list[dict]:
    """Scan for CRC-8 verified frames in the byte stream."""
    frames = []
    for start in range(min(32, len(data) - 3)):
        for length in range(2, min(24, len(data) - start - 1)):
            payload = data[start:start + length]
            calc    = crc8(payload)
            nxt     = data[start + length]
            if calc == nxt and nxt not in (0x00, 0xFF, 0xAA, 0x55):
                frames.append({"offset": start, "length": length,
                                "payload": payload, "crc": nxt})
    return frames


def _byte_to_display(b: int) -> str:
    return NIBBLE_CHAR[(b >> 4) & 0xF] + NIBBLE_CHAR[b & 0xF]


def scan_error_codes(data: bytes) -> list[dict]:
    """
    Try every byte-alignment offset and look for known LG error code strings
    in consecutive nibble pairs.  Returns a deduplicated list sorted by
    number of offsets the code appears in (most frequent first).

    Note: codes derived purely from 0xFF fill bytes (FF, CE, EF …) are
    suppressed when the byte stream contains >30 % 0xFF bytes, to avoid
    false positives from padding-heavy payloads.
    """
    found: dict[str, int] = {}
    bits = ''.join(f'{b:08b}' for b in data)

    ff_fraction = data.count(0xFF) / max(len(data), 1)

    for offset in range(16):
        shifted = bits[offset:]
        sb = [int(shifted[i:i+8], 2) for i in range(0, len(shifted) - 7, 8)]
        for i in range(len(sb)):
            b = sb[i]
            if b in (0x00, 0xFF, 0xAA, 0x55):
                continue
            display = _byte_to_display(b)
            if display in ERROR_CODES:
                found[display] = found.get(display, 0) + 1

    # Suppress codes that are artefacts of 0xFF-heavy streams
    FF_ARTEFACT_CODES = {"FF", "CE", "EF", "FE", "88"}
    results = []
    for code, hits in found.items():
        if ff_fraction > 0.30 and code in FF_ARTEFACT_CODES:
            continue
        info = ERROR_CODES.get(code)
        if info:
            results.append({"code": code, "hits": hits,
                             **dict(zip(("short", "desc", "action"), info))})

    return sorted(results, key=lambda x: x["hits"], reverse=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def hexdump(data: bytes) -> str:
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        h = " ".join(f"{b:02x}" for b in chunk)
        a = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {i:04x}:  {h:<48}  {a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Decode LG Smart Diagnostics acoustic signal and report error codes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decode_lg.py recording.m4a
  python decode_lg.py recording.wav --verbose
  python decode_lg.py recording.m4a --auto-tune
""",
    )
    ap.add_argument("file",        help="Audio recording (.wav, .m4a, .mp3, …)")
    ap.add_argument("--auto-tune", action="store_true",
                    help="Auto-detect carrier frequencies from the recording")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print symbol stream, bit stream, and hex dump")
    ap.add_argument("--onset-threshold", type=float, default=ONSET_THRESH, metavar="F",
                    help=f"Fraction of peak energy for onset detection (default {ONSET_THRESH})")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    print(f"Loading: {args.file}")
    audio, fs = load_audio(args.file)
    print(f"Duration: {len(audio)/fs:.2f} s  Sample rate: {fs} Hz")

    # Tone selection + modulation detection
    if args.auto_tune:
        tones = detect_tones(audio, fs)
        print(f"Auto-detected tones: {[f'{t:.1f}' for t in tones]} Hz")
    else:
        tones = NOMINAL_TONES_4MFSK[:]
        print(f"Using nominal tones: {[f'{t:.2f}' for t in tones]} Hz")

    bfsk = is_bfsk(tones)
    print(f"Modulation        : {'BFSK (washer/dishwasher)' if bfsk else '4-MFSK (refrigerator)'}")

    # Demodulate
    symbols, onset_s, end_s = demodulate(audio, fs, tones, onset_threshold=args.onset_threshold)
    duration_s = len(audio) / fs
    signal_s   = end_s - onset_s

    print(f"\nSignal window : {onset_s:.2f} s – {end_s:.2f} s  (duration {signal_s:.2f} s)")
    if onset_s < 0.3:
        print("  WARNING: signal starts at the very beginning of recording — preamble may be cut off")
    if duration_s - end_s < 0.3:
        print("  WARNING: signal ends at the very end of recording — tail may be cut off")

    print(f"Symbols       : {len(symbols)}  ({len(symbols)/SYMBOL_RATE:.1f} s @ {SYMBOL_RATE:.2f} sym/s)")
    print(f"Tones (actual): {[f'{t:.1f}' for t in tones]} Hz")
    print(f"Symbol length : {int(round(fs / SYMBOL_RATE))} samples ({1000/SYMBOL_RATE:.1f} ms)")

    if args.verbose:
        print(f"\nSymbols: {''.join(str(s) for s in symbols)}")

    # Decode to bytes
    data = symbols_to_bytes_bfsk(symbols) if bfsk else symbols_to_bytes(symbols)

    if args.verbose:
        print(f"\nHex dump ({len(data)} bytes):\n{hexdump(data)}")

    # CRC-verified frames
    frames = find_crc_frames(data)
    if frames:
        print(f"\nCRC-8 verified frame(s) found:")
        for f in frames:
            print(f"  offset={f['offset']:2d} len={f['length']:2d}  "
                  f"payload={f['payload'].hex(' ')}  crc=0x{f['crc']:02x}")

    # Error code detection
    codes = scan_error_codes(data)
    if not codes:
        print("\nNo known LG error codes detected in signal.")
    else:
        # Show codes with at least 3 hits to filter noise (arbitrary threshold)
        significant = [c for c in codes if c["hits"] >= 3]
        noise       = [c for c in codes if c["hits"] < 3]

        print(f"\n{'='*60}")
        if significant:
            print("DETECTED ERROR CODE(S):")
            for c in significant:
                print(f"\n  [{c['code']}]  {c['short']}")
                print(f"  Description : {c['desc']}")
                print(f"  Action      : {c['action']}")
                print(f"  Confidence  : {c['hits']} / 16 bit offsets")
        else:
            print("No high-confidence error codes found.")
            print("Low-confidence candidates:", ", ".join(c['code'] for c in noise))
        print('='*60)


if __name__ == "__main__":
    main()
