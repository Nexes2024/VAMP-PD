"""
MDS-UPDRS Part 3 – Pronation–Supination Feature Extractor
==========================================================
Processes every matching video under a BIDS-style dataset tree and writes
one CSV row per video with all UPDRS-aligned features.

Filename convention:
    sub-XXX_task-pronationsupinationR_acq-gopro_video.mp4
    sub-XXX_task-pronationsupinationL_acq-gopro_video.mp4

Usage:
    python extract_pronation_supination_features.py --input_dir "./videos/" --hand right
    python extract_pronation_supination_features.py --input_dir "./videos/" --hand left
    python extract_pronation_supination_features.py --input_dir "./videos/" --hand right --cpu --conf 0.2

Signal definition
-----------------
  Wrist-anchored:
      Ĩ(t) = I(t) − W(t)      (index  tip, wrist-anchored)
      P̃(t) = P(t) − W(t)      (pinky  tip, wrist-anchored)
      M̃(t) = M(t) − W(t)      (middle tip, wrist-anchored)

  Hand axis vector:
      v(t) = Ĩ(t) − P̃(t)

  Rotation angle (unwrapped, radians):
      θ(t) = unwrap( atan2(v_y(t), v_x(t)) )

  Cycles: peaks and troughs of θ(t)
  Per-cycle amplitude: A_k = |θ(peak_k) − θ(nearest_trough_k)|

Output CSV columns:
    Identity   : video, subject, rel_path, hand, status
    Recording  : duration_s, fps, total_frames, detected_frames, detection_rate
    Speed      : cycle_count, cycles_per_second,
                 ICI_mean, ICI_std, ICI_CV, ICI_slope
    Amplitude  : amp_mean, amp_std, amp_CV, amp_min,
                 theta_range
                 (amplitude is in radians – already scale-invariant, no pixel normalisation)
    Decrement  : amp_slope, half_ratio, end_drop, decrement_onset
    Pauses     : pause_count, max_pause_s, pause_fraction, flat_fraction
    Raw signal : theta_mean, theta_std, theta_min, theta_max,
                 forearm_len_mean  (QC only – not used in feature computation)
"""

import os
import csv
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from mmpose.apis import MMPoseInferencer

# ── KEYPOINT INDEX MAP ────────────────────────────────────────────────────────
#
# Wholebody 133-kpt layout:
#   Body 0-16 (COCO) | Foot 17-22 | Face 23-90
#   L-Hand 91-111    | R-Hand 112-132
#
# COCO body indices:
#   7 = left_elbow    8 = right_elbow
#   9 = left_wrist   10 = right_wrist
#
# Hand block offsets:
#   +0  = wrist   +8  = index tip   +12 = middle tip   +20 = pinky tip

HAND_CONFIG = {
    "right": {
        "label":        "R",
        "keyword":      "task-pronationsupinationR",
        "elbow":        8,
        "body_wrist":   10,
        "hand_wrist":   112,   # 112 + 0
        "index_tip":    120,   # 112 + 8
        "middle_tip":   124,   # 112 + 12
        "pinky_tip":    132,   # 112 + 20
    },
    "left": {
        "label":        "L",
        "keyword":      "task-pronationsupinationL",
        "elbow":        7,
        "body_wrist":   9,
        "hand_wrist":   91,    # 91 + 0
        "index_tip":    99,    # 91 + 8
        "middle_tip":   103,   # 91 + 12
        "pinky_tip":    111,   # 91 + 20
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── SIGNAL HELPERS ────────────────────────────────────────────────────────────

def smooth(x, wlen=11):
    if len(x) < wlen:
        return x.copy()
    return savgol_filter(x, window_length=wlen, polyorder=3)


def detect_peaks_valleys(signal_sm, min_dist_frames=8):
    """
    Adaptive prominence = 10 % of signal range (same rule as other tasks).
    Returns (peaks, valleys) as frame-index arrays.
    """
    sig_range  = float(np.max(signal_sm) - np.min(signal_sm))
    prominence = max(0.05, 0.10 * sig_range)   # floor at 0.05 rad

    peaks,  _ = find_peaks( signal_sm, distance=min_dist_frames, prominence=prominence)
    valleys, _ = find_peaks(-signal_sm, distance=min_dist_frames, prominence=prominence)
    return peaks, valleys


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(theta_raw, forearm_len_raw, fps):
    """
    theta_raw      : unwrapped rotation angle array  (radians)
    forearm_len_raw: ‖elbow − wrist‖ per frame       (pixels, for normalisation)
    fps            : frames per second

    Returns dict of all features.
    """
    feat = {}
    n        = len(theta_raw)
    duration = n / fps

    # ── Raw signal stats ──────────────────────────────────────────────────────
    feat["duration_s"]        = round(duration, 3)
    feat["total_frames"]      = n
    feat["theta_mean"]        = round(float(np.mean(theta_raw)),  4)
    feat["theta_std"]         = round(float(np.std(theta_raw)),   4)
    feat["theta_min"]         = round(float(np.min(theta_raw)),   4)
    feat["theta_max"]         = round(float(np.max(theta_raw)),   4)
    feat["forearm_len_mean"]  = round(float(np.mean(forearm_len_raw)), 3)

    NAN_KEYS = [
        "cycle_count", "cycles_per_second",
        "ICI_mean", "ICI_std", "ICI_CV", "ICI_slope",
        "amp_mean", "amp_std", "amp_CV", "amp_min",
        "theta_range",
        "amp_slope", "half_ratio", "end_drop", "decrement_onset",
        "pause_count", "max_pause_s", "pause_fraction", "flat_fraction",
    ]

    # ── Smooth ────────────────────────────────────────────────────────────────
    theta_sm = smooth(theta_raw)

    # ── Cycle detection ───────────────────────────────────────────────────────
    peaks, valleys = detect_peaks_valleys(theta_sm)

    if len(peaks) < 2 or len(valleys) < 1:
        for k in NAN_KEYS:
            feat[k] = float("nan")
        return feat

    # ── Speed / Rhythm ────────────────────────────────────────────────────────
    cycle_count  = len(peaks)
    cycles_per_s = cycle_count / duration

    ICI      = np.diff(peaks) / fps          # inter-cycle intervals  (s)
    ICI_mean = float(np.mean(ICI))
    ICI_std  = float(np.std(ICI))
    ICI_CV   = ICI_std / ICI_mean if ICI_mean > 0 else float("nan")
    ICI_slope = float(np.polyfit(np.arange(len(ICI)), ICI, 1)[0]) \
                if len(ICI) >= 3 else float("nan")

    feat["cycle_count"]       = cycle_count
    feat["cycles_per_second"] = round(cycles_per_s, 4)
    feat["ICI_mean"]          = round(ICI_mean,   4)
    feat["ICI_std"]           = round(ICI_std,    4)
    feat["ICI_CV"]            = round(ICI_CV,     4) if not np.isnan(ICI_CV)   else float("nan")
    feat["ICI_slope"]         = round(ICI_slope,  6) if not np.isnan(ICI_slope) else float("nan")

    # ── Per-cycle amplitude  A_k = |θ(peak_k) − θ(nearest_preceding_valley_k)| ─
    # We use the nearest PRECEDING valley (last valley before the peak).
    # This gives a consistent half-cycle measurement: the trough the hand
    # came from before reaching this peak, avoiding cross-pairing on noisy signals.
    amps = []

    for pk in peaks:
        # All valleys that come strictly before this peak
        preceding = valleys[valleys < pk]
        if len(preceding) == 0:
            # No preceding valley: fall back to nearest following valley
            following = valleys[valleys > pk]
            if len(following) == 0:
                continue          # no valley at all – skip this cycle
            ref_v = following[0]
        else:
            ref_v = preceding[-1]   # closest preceding valley

        A_k = abs(float(theta_sm[pk]) - float(theta_sm[ref_v]))
        amps.append(A_k)

    if len(amps) == 0:
        for k in NAN_KEYS:
            feat[k] = float("nan")
        return feat

    amps     = np.array(amps)
    amp_mean = float(np.mean(amps))
    amp_std  = float(np.std(amps))
    amp_CV   = amp_std / amp_mean if amp_mean > 0 else float("nan")

    # NOTE: amplitude is in radians (already scale-invariant).
    # We do NOT normalise by forearm length (rad/pixel has no interpretation).
    # forearm_len_mean is retained in the CSV as a QC column: large variance
    # across subjects indicates inconsistent camera distance or elbow dropout.
    theta_range = float(np.max(theta_sm) - np.min(theta_sm))

    feat["amp_mean"]    = round(amp_mean,   4)
    feat["amp_std"]     = round(amp_std,    4)
    feat["amp_CV"]      = round(amp_CV,     4) if not np.isnan(amp_CV) else float("nan")
    feat["amp_min"]     = round(float(np.min(amps)), 4)
    feat["theta_range"] = round(theta_range, 4)

    # ── Decrement ─────────────────────────────────────────────────────────────
    K = len(amps)

    amp_slope = float(np.polyfit(np.arange(K), amps, 1)[0]) if K >= 3 else float("nan")

    mid        = K // 2
    half_ratio = (float(np.mean(amps[:mid])) / float(np.mean(amps[mid:]))
                  if mid > 0 and float(np.mean(amps[mid:])) > 0
                  else float("nan"))

    end_drop = ((float(np.mean(amps[:3])) - float(np.mean(amps[-3:]))) /
                float(np.mean(amps[:3]))
                if K >= 6 and float(np.mean(amps[:3])) > 0
                else float("nan"))

    # decrement_onset: first cycle k where A_k < 0.75 * mean(A_1:3)
    decrement_onset = float("nan")
    if K >= 4:
        threshold = 0.75 * float(np.mean(amps[:3]))
        below = np.where(amps < threshold)[0]
        if len(below) > 0:
            decrement_onset = int(below[0]) + 1   # 1-indexed

    feat["amp_slope"]       = round(amp_slope, 6)       if not np.isnan(amp_slope)       else float("nan")
    feat["half_ratio"]      = round(half_ratio, 4)      if not np.isnan(half_ratio)      else float("nan")
    feat["end_drop"]        = round(end_drop, 4)        if not np.isnan(end_drop)        else float("nan")
    feat["decrement_onset"] = decrement_onset

    # ── Pauses / Hesitations ──────────────────────────────────────────────────
    pause_thr      = 2.0 * float(np.median(ICI))
    pauses         = ICI[ICI > pause_thr]
    pause_count    = int(len(pauses))
    max_pause      = float(np.max(pauses)) if pause_count > 0 else 0.0
    pause_fraction = float(np.sum(pauses)) / duration if duration > 0 else 0.0

    # flat_fraction: frames where |Δθ| < 5 % of theta_range
    delta_thr      = 0.05 * theta_range if theta_range > 0 else 0.05
    delta_theta    = np.abs(np.diff(theta_sm))
    flat_fraction  = float(np.sum(delta_theta < delta_thr)) / len(delta_theta) \
                     if len(delta_theta) > 0 else float("nan")

    feat["pause_count"]    = pause_count
    feat["max_pause_s"]    = round(max_pause,      4)
    feat["pause_fraction"] = round(pause_fraction, 4)
    feat["flat_fraction"]  = round(flat_fraction,  4)

    return feat


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def process_one_video(video_path, inferencer, cfg, conf_thr, fps_override=None):
    """
    Run MMPose on every frame, compute θ(t) and forearm length.

    Returns:
        fps            : float
        detected       : int
        theta_raw      : np.ndarray  (radians, unwrapped)
        forearm_len_raw: np.ndarray  (pixels)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    IDX_ELBOW       = cfg["elbow"]
    IDX_BODY_WRIST  = cfg["body_wrist"]
    IDX_HAND_WRIST  = cfg["hand_wrist"]
    IDX_INDEX_TIP   = cfg["index_tip"]
    # IDX_MIDDLE_TIP is defined in HAND_CONFIG but not used in the core θ(t)
    # signal — the axis is index↔pinky only. Kept in config for the test
    # visualisation script; not needed here.
    IDX_PINKY_TIP   = cfg["pinky_tip"]

    theta_list      = []
    forearm_list    = []
    detected        = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_gen = inferencer(frame, show=False,
                                return_vis=False, return_datasamples=False)
        results    = next(result_gen)

        preds     = results.get("predictions", [])
        instances = preds[0] if preds and isinstance(preds[0], list) else preds

        got_kpt = False
        if instances:
            best      = max(instances, key=lambda x: np.mean(x["keypoint_scores"]))
            keypoints = np.array(best["keypoints"])
            scores    = np.array(best["keypoint_scores"])

            # Wrist: prefer hand_wrist, fall back to body_wrist
            if scores[IDX_HAND_WRIST] >= conf_thr:
                W = keypoints[IDX_HAND_WRIST]
            elif scores[IDX_BODY_WRIST] >= conf_thr:
                W = keypoints[IDX_BODY_WRIST]
            else:
                W = None

            # Require index tip + pinky tip + wrist (for the core signal)
            # Middle tip is used for anchor quality check but not strictly required
            need_ok = (
                W is not None and
                scores[IDX_INDEX_TIP] >= conf_thr and
                scores[IDX_PINKY_TIP] >= conf_thr
            )

            if need_ok:
                I_t = keypoints[IDX_INDEX_TIP] - W   # wrist-anchored index tip
                P_t = keypoints[IDX_PINKY_TIP]  - W  # wrist-anchored pinky tip

                # Hand axis vector v = Ĩ − P̃
                v   = I_t - P_t
                theta_list.append(np.arctan2(v[1], v[0]))

                # Forearm length  (elbow → wrist)
                if scores[IDX_ELBOW] >= conf_thr:
                    fl = float(np.linalg.norm(keypoints[IDX_ELBOW] - W))
                elif len(forearm_list) > 0:
                    fl = forearm_list[-1]   # forward-fill if elbow lost
                else:
                    fl = 1.0

                forearm_list.append(fl)
                detected += 1
                got_kpt = True

        # Forward-fill so signal stays continuous
        if not got_kpt and len(theta_list) > 0:
            theta_list.append(theta_list[-1])
            forearm_list.append(forearm_list[-1])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"    [{frame_idx}/{total}]", end="\r", flush=True)

    cap.release()
    print()

    if len(theta_list) == 0:
        return fps, 0, np.array([]), np.array([])

    # Unwrap the angle time-series to remove ±π discontinuities
    theta_unwrapped = np.unwrap(np.array(theta_list))
    return fps, detected, theta_unwrapped, np.array(forearm_list)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Pronation–Supination – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root (searched recursively)")
    parser.add_argument("--output",    default="updrs_pronation_features.csv",
                        help="Output CSV path  (default: updrs_pronation_features.csv)")
    parser.add_argument("--hand",      default="right", choices=["left", "right"],
                        help="Which hand to analyse  (default: right)")
    parser.add_argument("--conf",      default=0.3,  type=float,
                        help="Keypoint confidence threshold  (default: 0.3)")
    parser.add_argument("--fps",       default=None, type=float,
                        help="Override FPS if video metadata is wrong")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    cfg          = HAND_CONFIG[args.hand]
    hand_keyword = cfg["keyword"]   # e.g. "task-pronationsupinationR"

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
            if "pronationsupination" not in fname_lower:
                skipped_wrong_task += 1
                continue
            if hand_keyword.lower() not in fname_lower:
                skipped_wrong_hand += 1
                continue
            video_files.append(os.path.join(root, fname))

    # ── Detect subject folders with no matched video ──────────────────────────
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
        print(f"        {skipped_wrong_hand} pronation video(s) skipped (wrong hand)")
        print(f"        {skipped_wrong_task} non-pronation video(s) ignored")
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
        # Speed / Rhythm
        "cycle_count", "cycles_per_second",
        "ICI_mean", "ICI_std", "ICI_CV", "ICI_slope",
        # Amplitude
        "amp_mean", "amp_std", "amp_CV", "amp_min",
        "theta_range",
        # Decrement
        "amp_slope", "half_ratio", "end_drop", "decrement_onset",
        # Pauses
        "pause_count", "max_pause_s", "pause_fraction", "flat_fraction",
        # Raw signal
        "theta_mean", "theta_std", "theta_min", "theta_max",
        "forearm_len_mean",
    ]

    rows = []

    for i, vpath in enumerate(video_files):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))
        print(f"[{i+1}/{len(video_files)}] {subject}/{vname}")

        try:
            fps, detected, theta_raw, forearm_raw = process_one_video(
                vpath, inferencer, cfg, args.conf, args.fps
            )

            if len(theta_raw) < 10:
                print(f"  [WARN] Too few detections ({detected}), skipping features.")
                row = {c: float("nan") for c in COLUMNS}
                row.update({"video": vname, "subject": subject,
                            "rel_path": rel_path, "hand": args.hand,
                            "status": "too_few_detections"})
                rows.append(row)
                continue

            feat = compute_features(theta_raw, forearm_raw, fps)

            feat["video"]           = vname
            feat["subject"]         = subject
            feat["rel_path"]        = rel_path
            feat["hand"]            = args.hand
            feat["status"]          = "ok"
            feat["fps"]             = round(fps, 2)
            feat["detected_frames"] = detected
            feat["detection_rate"]  = round(detected / len(theta_raw), 4)

            rows.append(feat)

            print(f"  cycles={feat['cycle_count']}  "
                  f"rate={feat.get('cycles_per_second','nan')}  "
                  f"amp_mean={feat.get('amp_mean','nan')} rad  "
                  f"ICI_CV={feat.get('ICI_CV','nan')}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            row = {c: float("nan") for c in COLUMNS}
            row.update({"video": vname, "subject": subject,
                        "rel_path": rel_path, "hand": args.hand,
                        "status": f"error: {e}"})
            rows.append(row)

    # ── Append missing-subject rows ───────────────────────────────────────────
    for ms in missing_subjects:
        row = {c: float("nan") for c in COLUMNS}
        row.update({"subject": ms, "hand": args.hand, "status": "missing_video"})
        rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    n_ok      = sum(1 for r in rows if r.get("status") == "ok")
    n_missing = sum(1 for r in rows if r.get("status") == "missing_video")
    n_err     = sum(1 for r in rows if str(r.get("status","")).startswith("error"))
    n_few     = sum(1 for r in rows if r.get("status") == "too_few_detections")

    print(f"\n[DONE] Features saved → {args.output}")
    print(f"       Total rows    : {len(rows)}")
    print(f"       OK            : {n_ok}")
    print(f"       Missing video : {n_missing}")
    print(f"       Too few kpts  : {n_few}")
    print(f"       Errors        : {n_err}")


if __name__ == "__main__":
    main()