"""
MDS-UPDRS Part 3 – Finger Tapping Feature Extractor  [v1]

Processes every video in an input folder and writes one CSV row per video
with all UPDRS-aligned features.

Usage:
    python extract_finger_tapping_features.py --input_dir "./videos/" --hand right
    python extract_finger_tapping_features.py --input_dir "./videos/" --hand left --output "results_left.csv"
    python extract_finger_tapping_features.py --input_dir "./videos/" --hand right --conf 0.2 --cpu

Output columns (one row per video):
    video, hand, duration_s, fps, total_frames, detected_frames,
    detection_rate,
    -- Speed --
    tap_count, taps_per_second, ITI_mean, ITI_std, ITI_CV, ITI_slope,
    -- Amplitude --
    amp_mean, amp_std, amp_CV, amp_max, amp_median,
    amp_norm_mean, amp_norm_std,        (normalised by hand size proxy)
    amp_slope, half_ratio, end_drop,
    -- Pauses / Hesitations --
    pause_count, max_pause_s, pause_fraction,
    flat_fraction,
    -- Raw signal stats --
    dist_mean, dist_std, dist_min, dist_max, dist_range,
    hand_size_proxy_mean
"""

import os
import csv
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from mmpose.apis import MMPoseInferencer

# ── HAND DEFINITIONS ──────────────────────────────────────────────────────────
HAND_CONFIG = {
    "right": {"wrist": 112, "thumb_tip": 116, "index_tip": 120, "label": "R"},
    "left":  {"wrist":  91, "thumb_tip":  95, "index_tip":  99, "label": "L"},
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── SIGNAL PROCESSING HELPERS ────────────────────────────────────────────────

def smooth(x, wlen=11):
    if len(x) < wlen:
        return x.copy()
    return savgol_filter(x, window_length=wlen, polyorder=3)


def detect_taps(d_smooth, fps, min_dist_frames=8, prominence=None):
    """
    Taps = local MINIMA of d(t)  (fingers closest / touching).

    prominence: if None (default) it is set adaptively to
        0.1 * (max - min) of d_smooth, so it scales with each patient's
        actual motion range instead of using a fixed pixel threshold.
    Returns array of frame indices.
    """
    if prominence is None:
        signal_range = float(np.max(d_smooth) - np.min(d_smooth))
        prominence   = 0.1 * signal_range      # adaptive: 10% of the motion range

    valleys, _ = find_peaks(
        -d_smooth,
        distance=min_dist_frames,
        prominence=prominence,
    )
    return valleys


def compute_features(frames_arr, d_raw, d_wrist_proxy, fps, hand_label):
    """
    Given the raw distance signal d_raw and wrist-to-index proxy,
    compute all UPDRS features. Returns a dict.
    """
    feat = {}
    n = len(d_raw)
    duration = n / fps

    feat["duration_s"]       = round(duration, 3)
    feat["total_frames"]     = n
    feat["dist_mean"]        = round(float(np.mean(d_raw)), 3)
    feat["dist_std"]         = round(float(np.std(d_raw)),  3)
    feat["dist_min"]         = round(float(np.min(d_raw)),  3)
    feat["dist_max"]         = round(float(np.max(d_raw)),  3)
    feat["dist_range"]       = round(float(np.max(d_raw) - np.min(d_raw)), 3)
    feat["hand_size_proxy_mean"] = round(float(np.mean(d_wrist_proxy)), 3)

    # ── Smooth ────────────────────────────────────────────────────────────────
    d_sm = smooth(d_raw)

    # ── Tap detection (minima) ────────────────────────────────────────────────
    taps = detect_taps(d_sm, fps)
    tap_count = len(taps)
    feat["tap_count"] = tap_count

    if tap_count < 2:
        # Not enough taps – fill with NaN
        for k in ["taps_per_second","ITI_mean","ITI_std","ITI_CV","ITI_slope",
                  "amp_mean","amp_std","amp_CV","amp_max","amp_median",
                  "amp_norm_mean","amp_norm_std","amp_slope","half_ratio","end_drop",
                  "pause_count","max_pause_s","pause_fraction","flat_fraction"]:
            feat[k] = float("nan")
        return feat

    taps_per_sec = tap_count / duration
    feat["taps_per_second"] = round(taps_per_sec, 4)

    # ── ITI (inter-tap intervals) ─────────────────────────────────────────────
    ITI = np.diff(taps) / fps          # seconds between consecutive taps
    ITI_mean = float(np.mean(ITI))
    ITI_std  = float(np.std(ITI))
    ITI_CV   = ITI_std / ITI_mean if ITI_mean > 0 else float("nan")

    # ITI slope: positive → slowing down over time
    tap_idx  = np.arange(len(ITI))
    if len(ITI) >= 3:
        iti_slope = float(np.polyfit(tap_idx, ITI, 1)[0])
    else:
        iti_slope = float("nan")

    feat["ITI_mean"]  = round(ITI_mean,  4)
    feat["ITI_std"]   = round(ITI_std,   4)
    feat["ITI_CV"]    = round(ITI_CV,    4)
    feat["ITI_slope"] = round(iti_slope, 6) if not np.isnan(iti_slope) else float("nan")

    # ── Amplitude per cycle ───────────────────────────────────────────────────
    # Between each pair of consecutive taps, find the peak of d(t)
    amps      = []
    amps_norm = []
    proxy_mean = float(np.mean(d_wrist_proxy)) if len(d_wrist_proxy) > 0 else 1.0

    for i in range(len(taps) - 1):
        seg = d_sm[taps[i]:taps[i+1]]
        if len(seg) == 0:
            continue
        peak_val = float(np.max(seg))
        amps.append(peak_val)
        # Normalise by hand-size proxy so it's comparable across patients
        amps_norm.append(peak_val / proxy_mean if proxy_mean > 0 else float("nan"))

    amps      = np.array(amps)
    amps_norm = np.array(amps_norm)

    if len(amps) == 0:
        for k in ["amp_mean","amp_std","amp_CV","amp_max","amp_median",
                  "amp_norm_mean","amp_norm_std","amp_slope","half_ratio","end_drop"]:
            feat[k] = float("nan")
    else:
        amp_mean   = float(np.mean(amps))
        amp_std    = float(np.std(amps))
        amp_CV     = amp_std / amp_mean if amp_mean > 0 else float("nan")
        amp_slope  = float(np.polyfit(np.arange(len(amps)), amps, 1)[0]) if len(amps)>=3 else float("nan")

        # half_ratio: mean of first half / mean of second half
        mid = len(amps) // 2
        half_ratio = (float(np.mean(amps[:mid])) / float(np.mean(amps[mid:]))) \
                     if mid > 0 and np.mean(amps[mid:]) > 0 else float("nan")

        # end_drop: (mean first 3 − mean last 3) / mean first 3
        if len(amps) >= 6:
            end_drop = (float(np.mean(amps[:3])) - float(np.mean(amps[-3:]))) / \
                       float(np.mean(amps[:3])) if np.mean(amps[:3]) > 0 else float("nan")
        else:
            end_drop = float("nan")

        feat["amp_mean"]      = round(amp_mean, 3)
        feat["amp_std"]       = round(amp_std,  3)
        feat["amp_CV"]        = round(amp_CV,   4)
        feat["amp_max"]       = round(float(np.max(amps)),    3)
        feat["amp_median"]    = round(float(np.median(amps)), 3)
        feat["amp_norm_mean"] = round(float(np.mean(amps_norm)), 4) if not np.any(np.isnan(amps_norm)) else float("nan")
        feat["amp_norm_std"]  = round(float(np.std(amps_norm)),  4) if not np.any(np.isnan(amps_norm)) else float("nan")
        feat["amp_slope"]     = round(amp_slope,  6) if not np.isnan(amp_slope) else float("nan")
        feat["half_ratio"]    = round(half_ratio, 4) if not np.isnan(half_ratio) else float("nan")
        feat["end_drop"]      = round(end_drop,   4) if not np.isnan(end_drop)   else float("nan")

    # ── Pauses / Hesitations ──────────────────────────────────────────────────
    pause_threshold = 2.0 * float(np.median(ITI))   # > 2× median ITI = pause
    pauses          = ITI[ITI > pause_threshold]
    pause_count     = int(len(pauses))
    max_pause       = float(np.max(pauses)) if pause_count > 0 else 0.0
    pause_fraction  = float(np.sum(pauses)) / duration if duration > 0 else 0.0

    feat["pause_count"]     = pause_count
    feat["max_pause_s"]     = round(max_pause,     4)
    feat["pause_fraction"]  = round(pause_fraction, 4)

    # Flat fraction: frames where |Δd| < 5% of dist_range
    delta_threshold = 0.05 * feat["dist_range"] if feat["dist_range"] > 0 else 0.5
    delta_d = np.abs(np.diff(d_sm))
    flat_fraction = float(np.sum(delta_d < delta_threshold)) / len(delta_d) if len(delta_d) > 0 else float("nan")
    feat["flat_fraction"] = round(flat_fraction, 4)

    return feat


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def process_one_video(video_path, inferencer, cfg, conf_thr, fps_override=None):
    """
    Run MMPose on every frame of video_path.
    Returns (fps, detected_frames, d_raw, d_wrist_proxy) as numpy arrays.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    IDX_WRIST     = cfg["wrist"]
    IDX_THUMB_TIP = cfg["thumb_tip"]
    IDX_INDEX_TIP = cfg["index_tip"]

    d_raw          = []
    d_wrist_proxy  = []
    detected       = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_gen = inferencer(frame, show=False, return_vis=False, return_datasamples=False)
        results    = next(result_gen)

        preds     = results.get("predictions", [])
        instances = preds[0] if preds and isinstance(preds[0], list) else preds

        got_kpt = False
        if instances:
            best      = max(instances, key=lambda x: np.mean(x["keypoint_scores"]))
            keypoints = np.array(best["keypoints"])
            scores    = np.array(best["keypoint_scores"])

            if (IDX_WRIST     < len(scores) and scores[IDX_WRIST]     >= conf_thr and
                IDX_THUMB_TIP < len(scores) and scores[IDX_THUMB_TIP] >= conf_thr and
                IDX_INDEX_TIP < len(scores) and scores[IDX_INDEX_TIP] >= conf_thr):

                W = keypoints[IDX_WRIST]
                T = keypoints[IDX_THUMB_TIP] - W     # wrist-anchored
                I = keypoints[IDX_INDEX_TIP]  - W

                dist_pinch = float(np.linalg.norm(I - T))
                dist_proxy = float(np.linalg.norm(keypoints[IDX_INDEX_TIP] - keypoints[IDX_WRIST]))

                d_raw.append(dist_pinch)
                d_wrist_proxy.append(dist_proxy)
                detected += 1
                got_kpt = True

        if not got_kpt and len(d_raw) > 0:
            # Fill forward so signal stays continuous
            d_raw.append(d_raw[-1])
            d_wrist_proxy.append(d_wrist_proxy[-1])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"    [{frame_idx}/{total}]", end="\r", flush=True)

    cap.release()
    print()
    return fps, detected, np.array(d_raw), np.array(d_wrist_proxy)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Finger Tapping – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing input videos")
    parser.add_argument("--output",    default="updrs_features.csv",
                        help="Output CSV path  (default: updrs_features.csv)")
    parser.add_argument("--hand",      default="right", choices=["left", "right"],
                        help="Which hand to analyse  (default: right)")
    parser.add_argument("--conf",      default=0.3,  type=float,
                        help="Keypoint confidence threshold  (default: 0.3)")
    parser.add_argument("--fps",       default=None, type=float,
                        help="Override FPS (useful if metadata is wrong)")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    cfg = HAND_CONFIG[args.hand]

    # ── Keyword that must appear in the filename ───────────────────────────────
    # left  → task-fingertappingL
    # right → task-fingertappingR
    hand_keyword = f"task-fingertapping{cfg['label']}"   # e.g. "task-fingertappingR"

    # ── Recursive walk through the whole dataset tree ─────────────────────────
    video_files        = []
    skipped_wrong_hand = 0
    skipped_no_tap     = 0

    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()          # alphabetical, reproducible ordering
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue

            fname_lower = fname.lower()

            # Must be a fingertapping video at all
            if "fingertapping" not in fname_lower:
                skipped_no_tap += 1
                continue

            # Must match the requested hand (case-insensitive)
            if hand_keyword.lower() not in fname_lower:
                skipped_wrong_hand += 1
                continue

            video_files.append(os.path.join(root, fname))

    if not video_files:
        print(f"[ERROR] No '{hand_keyword}' videos found under: {args.input_dir}")
        print(f"        {skipped_wrong_hand} fingertapping video(s) skipped (wrong hand)")
        print(f"        {skipped_no_tap} non-fingertapping video(s) ignored")
        return

    print(f"[INFO] Dataset root  : {args.input_dir}")
    print(f"[INFO] Hand          : {args.hand.upper()}  (matching '{hand_keyword}')")
    print(f"[INFO] Videos matched: {len(video_files)}")
    print(f"         skipped – wrong hand : {skipped_wrong_hand}")
    print(f"         skipped – not tapping: {skipped_no_tap}")
    print(f"[INFO] Output        : {args.output}\n")
    for vp in video_files:
        print(f"  + {os.path.relpath(vp, args.input_dir)}")
    print()

    # Load model once
    device = "cpu" if args.cpu else "cuda:0"
    print(f"[INFO] Loading MMPose wholebody model on {device} …")
    inferencer = MMPoseInferencer(pose2d="wholebody", device=device)

    # CSV columns (ordered)
    COLUMNS = [
        "video", "subject", "rel_path", "hand", "duration_s", "fps", "total_frames",
        "detected_frames", "detection_rate",
        # Speed
        "tap_count", "taps_per_second",
        "ITI_mean", "ITI_std", "ITI_CV", "ITI_slope",
        # Amplitude
        "amp_mean", "amp_std", "amp_CV", "amp_max", "amp_median",
        "amp_norm_mean", "amp_norm_std",
        "amp_slope", "half_ratio", "end_drop",
        # Pauses
        "pause_count", "max_pause_s", "pause_fraction", "flat_fraction",
        # Raw signal
        "dist_mean", "dist_std", "dist_min", "dist_max", "dist_range",
        "hand_size_proxy_mean",
    ]

    rows = []

    for i, vpath in enumerate(video_files):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))   # e.g. sub-PD001
        print(f"[{i+1}/{len(video_files)}] {subject}/{vname}")

        try:
            fps, detected, d_raw, d_proxy = process_one_video(
                vpath, inferencer, cfg, args.conf, args.fps
            )

            if len(d_raw) < 10:
                print(f"  [WARN] Too few detections ({detected}), skipping.")
                row = {c: float("nan") for c in COLUMNS}
                row["video"] = vname
                row["hand"]  = args.hand
                rows.append(row)
                continue

            feat = compute_features(
                np.arange(len(d_raw)), d_raw, d_proxy, fps, cfg["label"]
            )

            feat["video"]           = vname
            feat["subject"]         = subject
            feat["rel_path"]        = rel_path
            feat["hand"]            = args.hand
            feat["fps"]             = round(fps, 2)
            feat["detected_frames"] = detected
            feat["detection_rate"]  = round(detected / len(d_raw), 4) if len(d_raw) > 0 else 0

            rows.append(feat)

            print(f"  taps={feat['tap_count']}  "
                  f"rate={feat.get('taps_per_second', 'nan')}  "
                  f"amp_mean={feat.get('amp_mean', 'nan')}  "
                  f"ITI_CV={feat.get('ITI_CV', 'nan')}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            row = {c: float("nan") for c in COLUMNS}
            row["video"]    = vname
            row["subject"]  = subject
            row["rel_path"] = rel_path
            row["hand"]     = args.hand
            rows.append(row)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Features saved → {args.output}")
    print(f"       {len(rows)} rows  ×  {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
