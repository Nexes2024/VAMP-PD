"""
MDS-UPDRS Part 3 – Hand Movements Feature Extractor
====================================================
Processes every matching video under a BIDS-style dataset tree and writes
one CSV row per video with all UPDRS-aligned features.

Filename convention expected:
    sub-XXX_task-handmovementsL_acq-gopro_video.mp4   (left hand)
    sub-XXX_task-handmovementsR_acq-gopro_video.mp4   (right hand)

Usage:
    python extract_hand_movement_features.py --input_dir "./videos/" --hand right
    python extract_hand_movement_features.py --input_dir "./videos/" --hand left
    python extract_hand_movement_features.py --input_dir "./videos/" --hand right --cpu --conf 0.2

Output CSV columns (one row per video):
    Identity  : video, subject, rel_path, hand, status
    Recording : duration_s, fps, total_frames, detected_frames, detection_rate
    Speed     : cycle_count, cycles_per_second,
                ICI_mean, ICI_std, ICI_CV, interval_slope
    Amplitude : amp_mean, amp_std, amp_CV, amp_min,
                open_peak_mean, open_peak_std,
                fist_valley_mean, fist_valley_std,
                amp_norm_mean, amp_norm_std,
                amp_slope, half_ratio, end_drop, decrement_onset
    Pauses    : pause_count, max_pause_s, pause_fraction,
                flat_fraction, missed_cycles
    Raw signal: O_mean, O_std, O_min, O_max, O_range,
                hand_size_proxy_mean

Openness signal definition
--------------------------
  Five fingertips: thumb(T), index(I), middle(M), ring(R), pinky(P)
  C_tip(t)  = centroid of the 5 tips  (wrist-anchored coordinates)
  O(t)      = mean_i ‖tip_i(t) − C_tip(t)‖          (spread)
  O_norm(t) = O(t) / hand_size(t)
  hand_size(t) = ‖middle_MCP(t) − wrist(t)‖
"""

import os
import csv
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from mmpose.apis import MMPoseInferencer

# ── HAND KEYPOINT INDICES ────────────────────────────────────────────────────
#
# Wholebody 133-kpt layout:
#   Body 0-16 | Foot 17-22 | Face 23-90 | L-Hand 91-111 | R-Hand 112-132
#
# Within each 21-kpt hand block:
#   [0]  Wrist
#   [1]  Thumb CMC  [2]  Thumb MCP  [3]  Thumb IP   [4]  Thumb TIP
#   [5]  Index MCP  [6]  Index PIP  [7]  Index DIP  [8]  Index TIP
#   [9]  Mid   MCP [10]  Mid   PIP [11]  Mid   DIP [12]  Mid   TIP
#  [13]  Ring  MCP [14]  Ring  PIP [15]  Ring  DIP [16]  Ring  TIP
#  [17]  Pinky MCP [18]  Pinky PIP [19]  Pinky DIP [20]  Pinky TIP

def _build_hand_config(base):
    return {
        "wrist":       base + 0,
        "thumb_tip":   base + 4,
        "index_mcp":   base + 5,
        "index_tip":   base + 8,
        "middle_mcp":  base + 9,
        "middle_tip":  base + 12,
        "ring_mcp":    base + 13,
        "ring_tip":    base + 16,
        "pinky_mcp":   base + 17,
        "pinky_tip":   base + 20,
    }

HAND_CONFIG = {
    "right": {"indices": _build_hand_config(112), "label": "R", "keyword": "task-handmovementsR"},
    "left":  {"indices": _build_hand_config(91),  "label": "L", "keyword": "task-handmovementsL"},
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── SIGNAL HELPERS ────────────────────────────────────────────────────────────

def smooth(x, wlen=11):
    if len(x) < wlen:
        return x.copy()
    return savgol_filter(x, window_length=wlen, polyorder=3)


def detect_peaks_valleys(signal_sm, min_dist_frames=8):
    """
    Adaptive prominence = 10% of signal range.
    Peaks  = hand fully open.
    Valleys = hand closed / fist.
    Returns (peaks, valleys) as frame-index arrays.
    """
    sig_range  = float(np.max(signal_sm) - np.min(signal_sm))
    prominence = max(0.01, 0.1 * sig_range)   # floor prevents zero

    peaks,  _ = find_peaks( signal_sm, distance=min_dist_frames, prominence=prominence)
    valleys, _ = find_peaks(-signal_sm, distance=min_dist_frames, prominence=prominence)
    return peaks, valleys


def safe_nan(x):
    return float("nan") if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 6)


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(O_raw, O_norm_raw, hand_size_raw, fps):
    """
    O_raw       : raw openness signal  (pixels)
    O_norm_raw  : openness normalised by hand size
    hand_size_raw: wrist–middle_MCP distance per frame
    fps         : frames per second

    Returns dict of all features.
    """
    feat = {}
    n        = len(O_raw)
    duration = n / fps

    # ── Raw signal stats ──────────────────────────────────────────────────────
    feat["O_mean"]             = round(float(np.mean(O_raw)),  3)
    feat["O_std"]              = round(float(np.std(O_raw)),   3)
    feat["O_min"]              = round(float(np.min(O_raw)),   3)
    feat["O_max"]              = round(float(np.max(O_raw)),   3)
    feat["O_range"]            = round(float(np.max(O_raw) - np.min(O_raw)), 3)
    feat["hand_size_proxy_mean"] = round(float(np.mean(hand_size_raw)), 3)
    feat["duration_s"]         = round(duration, 3)
    feat["total_frames"]       = n

    NAN_KEYS = [
        "cycle_count", "cycles_per_second",
        "ICI_mean", "ICI_std", "ICI_CV", "interval_slope",
        "amp_mean", "amp_std", "amp_CV", "amp_min",
        "open_peak_mean", "open_peak_std",
        "fist_valley_mean", "fist_valley_std",
        "amp_norm_mean", "amp_norm_std",
        "amp_slope", "half_ratio", "end_drop", "decrement_onset",
        "pause_count", "max_pause_s", "pause_fraction",
        "flat_fraction", "missed_cycles",
    ]

    # ── Smooth ────────────────────────────────────────────────────────────────
    O_sm      = smooth(O_raw)
    O_norm_sm = smooth(O_norm_raw)

    peaks, valleys = detect_peaks_valleys(O_sm)

    # Need at least 2 of each for meaningful features
    if len(peaks) < 2 or len(valleys) < 2:
        for k in NAN_KEYS:
            feat[k] = float("nan")
        return feat

    # ── Speed ─────────────────────────────────────────────────────────────────
    # A "cycle" = one open + one close.
    # We define cycle times as peak locations (one peak = one open phase).
    cycle_times  = peaks                        # frame indices of open peaks
    cycle_count  = len(cycle_times)
    cycles_per_s = cycle_count / duration

    ICI = np.diff(cycle_times) / fps            # inter-cycle intervals (s)
    ICI_mean  = float(np.mean(ICI))
    ICI_std   = float(np.std(ICI))
    ICI_CV    = ICI_std / ICI_mean if ICI_mean > 0 else float("nan")
    int_slope = float(np.polyfit(np.arange(len(ICI)), ICI, 1)[0]) if len(ICI) >= 3 else float("nan")

    feat["cycle_count"]      = cycle_count
    feat["cycles_per_second"] = round(cycles_per_s, 4)
    feat["ICI_mean"]         = round(ICI_mean,  4)
    feat["ICI_std"]          = round(ICI_std,   4)
    feat["ICI_CV"]           = round(ICI_CV,    4) if not np.isnan(ICI_CV) else float("nan")
    feat["interval_slope"]   = round(int_slope, 6) if not np.isnan(int_slope) else float("nan")

    # ── Amplitude per cycle ───────────────────────────────────────────────────
    # For cycle i: A_i = O_sm[peak_i] − nearest valley's O_sm
    # open_peak  = O_sm at the peak   (how fully open)
    # fist_valley= O_sm at the valley (how tight the fist)
    open_peaks_vals  = []
    fist_valley_vals = []
    amps             = []
    amps_norm        = []   # same but on the normalised signal

    for pk in peaks:
        open_peaks_vals.append(float(O_sm[pk]))

        # Find closest valley (before or after) to this peak
        diffs = valleys - pk
        nearest_valley = valleys[np.argmin(np.abs(diffs))]
        fist_val = float(O_sm[nearest_valley])
        fist_valley_vals.append(fist_val)

        amp = float(O_sm[pk]) - fist_val
        amps.append(amp)

        # Normalised amplitude using O_norm
        amp_norm = float(O_norm_sm[pk]) - float(O_norm_sm[nearest_valley])
        amps_norm.append(amp_norm)

    amps      = np.array(amps)
    amps_norm = np.array(amps_norm)

    amp_mean = float(np.mean(amps))
    amp_std  = float(np.std(amps))
    amp_CV   = amp_std / amp_mean if amp_mean > 0 else float("nan")

    # Decrement features
    amp_slope = float(np.polyfit(np.arange(len(amps)), amps, 1)[0]) if len(amps) >= 3 else float("nan")

    mid = len(amps) // 2
    half_ratio = (float(np.mean(amps[:mid])) / float(np.mean(amps[mid:]))
                  if mid > 0 and float(np.mean(amps[mid:])) > 0
                  else float("nan"))

    end_drop = ((float(np.mean(amps[:3])) - float(np.mean(amps[-3:]))) /
                float(np.mean(amps[:3]))
                if len(amps) >= 6 and float(np.mean(amps[:3])) > 0
                else float("nan"))

    # decrement_onset: first cycle where amplitude drops below 75% of initial amp
    decrement_onset = float("nan")
    if len(amps) >= 4:
        threshold = 0.75 * float(np.mean(amps[:3]))
        below = np.where(amps < threshold)[0]
        if len(below) > 0:
            decrement_onset = int(below[0]) + 1   # 1-indexed cycle number

    feat["amp_mean"]         = round(amp_mean, 3)
    feat["amp_std"]          = round(amp_std,  3)
    feat["amp_CV"]           = round(amp_CV,   4) if not np.isnan(amp_CV) else float("nan")
    feat["amp_min"]          = round(float(np.min(amps)), 3)
    feat["open_peak_mean"]   = round(float(np.mean(open_peaks_vals)),  3)
    feat["open_peak_std"]    = round(float(np.std(open_peaks_vals)),   3)
    feat["fist_valley_mean"] = round(float(np.mean(fist_valley_vals)), 3)
    feat["fist_valley_std"]  = round(float(np.std(fist_valley_vals)),  3)
    feat["amp_norm_mean"]    = round(float(np.mean(amps_norm)), 4)
    feat["amp_norm_std"]     = round(float(np.std(amps_norm)),  4)
    feat["amp_slope"]        = round(amp_slope, 6) if not np.isnan(amp_slope) else float("nan")
    feat["half_ratio"]       = round(half_ratio, 4) if not np.isnan(half_ratio) else float("nan")
    feat["end_drop"]         = round(end_drop,   4) if not np.isnan(end_drop)   else float("nan")
    feat["decrement_onset"]  = safe_nan(decrement_onset)

    # ── Pauses / hesitations ──────────────────────────────────────────────────
    pause_threshold = 2.0 * float(np.median(ICI))
    pauses          = ICI[ICI > pause_threshold]
    pause_count     = int(len(pauses))
    max_pause       = float(np.max(pauses)) if pause_count > 0 else 0.0
    pause_fraction  = float(np.sum(pauses)) / duration if duration > 0 else 0.0

    # flat_fraction: frames where |ΔO| < 5% of O_range
    delta_thr    = 0.05 * feat["O_range"] if feat["O_range"] > 0 else 0.5
    delta_O      = np.abs(np.diff(O_sm))
    flat_fraction = float(np.sum(delta_O < delta_thr)) / len(delta_O) if len(delta_O) > 0 else float("nan")

    # missed_cycles: expected ≈ 10 (standard UPDRS instruction)
    EXPECTED_CYCLES = 10
    missed_cycles = max(0, EXPECTED_CYCLES - cycle_count)

    feat["pause_count"]    = pause_count
    feat["max_pause_s"]    = round(max_pause,     4)
    feat["pause_fraction"] = round(pause_fraction, 4)
    feat["flat_fraction"]  = round(flat_fraction,  4)
    feat["missed_cycles"]  = missed_cycles

    return feat


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def process_one_video(video_path, inferencer, hand_indices, conf_thr, fps_override=None):
    """
    Run MMPose on every frame, compute the openness signal O(t).

    Returns:
        fps         : float
        detected    : int   (frames where all needed kpts were confident)
        O_raw       : np.ndarray  (openness, pixels, wrist-anchored)
        O_norm_raw  : np.ndarray  (openness normalised by hand size)
        hand_size   : np.ndarray  (wrist–middle_MCP distance per frame)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    IDX_WRIST      = hand_indices["wrist"]
    IDX_THUMB_TIP  = hand_indices["thumb_tip"]
    IDX_INDEX_TIP  = hand_indices["index_tip"]
    IDX_MIDDLE_TIP = hand_indices["middle_tip"]
    IDX_RING_TIP   = hand_indices["ring_tip"]
    IDX_PINKY_TIP  = hand_indices["pinky_tip"]
    IDX_MIDDLE_MCP = hand_indices["middle_mcp"]   # hand-size anchor

    TIP_INDICES = [IDX_THUMB_TIP, IDX_INDEX_TIP, IDX_MIDDLE_TIP,
                   IDX_RING_TIP,  IDX_PINKY_TIP]
    REQUIRED    = TIP_INDICES + [IDX_WRIST, IDX_MIDDLE_MCP]

    O_raw      = []
    O_norm_raw = []
    hand_size  = []
    detected   = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_gen = inferencer(frame, show=False, return_vis=False,
                                return_datasamples=False)
        results    = next(result_gen)

        preds     = results.get("predictions", [])
        instances = preds[0] if preds and isinstance(preds[0], list) else preds

        got_kpt = False
        if instances:
            best      = max(instances, key=lambda x: np.mean(x["keypoint_scores"]))
            keypoints = np.array(best["keypoints"])
            scores    = np.array(best["keypoint_scores"])

            # All required keypoints must be confident
            all_conf = all(
                idx < len(scores) and scores[idx] >= conf_thr
                for idx in REQUIRED
            )

            if all_conf:
                W = keypoints[IDX_WRIST]   # anchor

                # Wrist-anchored fingertips
                tips_anchored = np.array([
                    keypoints[idx] - W for idx in TIP_INDICES
                ])  # shape (5, 2)

                # Centroid of anchored tips
                C = tips_anchored.mean(axis=0)

                # Openness = mean distance of tips from their centroid
                O = float(np.mean(np.linalg.norm(tips_anchored - C, axis=1)))

                # Hand size proxy = wrist → middle MCP (anchored)
                hs = float(np.linalg.norm(keypoints[IDX_MIDDLE_MCP] - W))
                hs = hs if hs > 1e-3 else 1.0   # guard divide-by-zero

                O_raw.append(O)
                O_norm_raw.append(O / hs)
                hand_size.append(hs)

                detected += 1
                got_kpt = True

        # Forward-fill if detection failed (keeps signal continuous)
        if not got_kpt and len(O_raw) > 0:
            O_raw.append(O_raw[-1])
            O_norm_raw.append(O_norm_raw[-1])
            hand_size.append(hand_size[-1])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"    [{frame_idx}/{total}]", end="\r", flush=True)

    cap.release()
    print()
    return fps, detected, np.array(O_raw), np.array(O_norm_raw), np.array(hand_size)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Hand Movements – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root (will be searched recursively)")
    parser.add_argument("--output",    default="updrs_handmovement_features.csv",
                        help="Output CSV path")
    parser.add_argument("--hand",      default="right", choices=["left", "right"],
                        help="Which hand to analyse  (default: right)")
    parser.add_argument("--conf",      default=0.3,  type=float,
                        help="Keypoint confidence threshold  (default: 0.3)")
    parser.add_argument("--fps",       default=None, type=float,
                        help="Override FPS if video metadata is wrong")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    cfg         = HAND_CONFIG[args.hand]
    hand_indices = cfg["indices"]
    hand_keyword = cfg["keyword"]   # e.g. "task-handmovementsR"

    # ── Recursive video discovery ─────────────────────────────────────────────
    video_files        = []
    skipped_wrong_task = 0
    skipped_wrong_hand = 0

    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            fname_lower = fname.lower()
            if "handmovements" not in fname_lower:
                skipped_wrong_task += 1
                continue
            if hand_keyword.lower() not in fname_lower:
                skipped_wrong_hand += 1
                continue
            video_files.append(os.path.join(root, fname))

    # ── Detect subject folders with no matched video (missing data) ───────────
    all_subject_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    subjects_with_video = {
        os.path.basename(os.path.dirname(vp)) for vp in video_files
    }
    missing_subjects = [d for d in all_subject_dirs if d not in subjects_with_video]

    if not video_files:
        print(f"[ERROR] No '{hand_keyword}' videos found under: {args.input_dir}")
        print(f"        {skipped_wrong_hand} handmovement video(s) skipped (wrong hand)")
        print(f"        {skipped_wrong_task} non-handmovement video(s) ignored")
        return

    print(f"[INFO] Dataset root  : {args.input_dir}")
    print(f"[INFO] Hand          : {args.hand.upper()}  (matching '{hand_keyword}')")
    print(f"[INFO] Videos matched: {len(video_files)}")
    print(f"         skipped – wrong hand : {skipped_wrong_hand}")
    print(f"         skipped – other task : {skipped_wrong_task}")
    if missing_subjects:
        print(f"[WARN] {len(missing_subjects)} subject folder(s) with no matching video:")
        for ms in missing_subjects:
            print(f"  - {ms}")
    print(f"[INFO] Output        : {args.output}\n")
    for vp in video_files:
        print(f"  + {os.path.relpath(vp, args.input_dir)}")
    print()

    # ── Load model once ───────────────────────────────────────────────────────
    device = "cpu" if args.cpu else "cuda:0"
    print(f"[INFO] Loading MMPose wholebody model on {device} …\n")
    inferencer = MMPoseInferencer(pose2d="wholebody", device=device)

    # ── CSV column order ──────────────────────────────────────────────────────
    COLUMNS = [
        # Identity
        "video", "subject", "rel_path", "hand", "status",
        # Recording
        "duration_s", "fps", "total_frames", "detected_frames", "detection_rate",
        # Speed
        "cycle_count", "cycles_per_second",
        "ICI_mean", "ICI_std", "ICI_CV", "interval_slope",
        # Amplitude
        "amp_mean", "amp_std", "amp_CV", "amp_min",
        "open_peak_mean", "open_peak_std",
        "fist_valley_mean", "fist_valley_std",
        "amp_norm_mean", "amp_norm_std",
        "amp_slope", "half_ratio", "end_drop", "decrement_onset",
        # Pauses
        "pause_count", "max_pause_s", "pause_fraction",
        "flat_fraction", "missed_cycles",
        # Raw signal
        "O_mean", "O_std", "O_min", "O_max", "O_range",
        "hand_size_proxy_mean",
    ]

    rows = []

    for i, vpath in enumerate(video_files):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))
        print(f"[{i+1}/{len(video_files)}] {subject}/{vname}")

        try:
            fps, detected, O_raw, O_norm_raw, hs_raw = process_one_video(
                vpath, inferencer, hand_indices, args.conf, args.fps
            )

            if len(O_raw) < 10:
                print(f"  [WARN] Too few detections ({detected}), skipping features.")
                row = {c: float("nan") for c in COLUMNS}
                row["video"]    = vname
                row["subject"]  = subject
                row["rel_path"] = rel_path
                row["hand"]     = args.hand
                row["status"]   = "too_few_detections"
                rows.append(row)
                continue

            feat = compute_features(O_raw, O_norm_raw, hs_raw, fps)

            feat["video"]          = vname
            feat["subject"]        = subject
            feat["rel_path"]       = rel_path
            feat["hand"]           = args.hand
            feat["status"]         = "ok"
            feat["fps"]            = round(fps, 2)
            feat["detected_frames"] = detected
            feat["detection_rate"] = round(detected / len(O_raw), 4) if len(O_raw) > 0 else 0.0

            rows.append(feat)

            print(f"  cycles={feat['cycle_count']}  "
                  f"rate={feat.get('cycles_per_second','nan')}  "
                  f"amp_mean={feat.get('amp_mean','nan')}  "
                  f"ICI_CV={feat.get('ICI_CV','nan')}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            row = {c: float("nan") for c in COLUMNS}
            row["video"]    = vname
            row["subject"]  = subject
            row["rel_path"] = rel_path
            row["hand"]     = args.hand
            row["status"]   = f"error: {e}"
            rows.append(row)

    # ── Append missing-subject rows ───────────────────────────────────────────
    for ms in missing_subjects:
        row = {c: float("nan") for c in COLUMNS}
        row["subject"] = ms
        row["hand"]    = args.hand
        row["status"]  = "missing_video"
        rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    n_ok      = sum(1 for r in rows if r.get("status") == "ok")
    n_missing = sum(1 for r in rows if r.get("status") == "missing_video")
    n_err     = sum(1 for r in rows if str(r.get("status", "")).startswith("error"))
    n_few     = sum(1 for r in rows if r.get("status") == "too_few_detections")

    print(f"\n[DONE] Features saved → {args.output}")
    print(f"       Total rows    : {len(rows)}")
    print(f"       OK            : {n_ok}")
    print(f"       Missing video : {n_missing}")
    print(f"       Too few kpts  : {n_few}")
    print(f"       Errors        : {n_err}")


if __name__ == "__main__":
    main()