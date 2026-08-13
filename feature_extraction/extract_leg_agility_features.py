"""
MDS-UPDRS Part 3 – Leg Agility Feature Extractor
=================================================
Processes every matching video under a BIDS-style dataset tree and writes
one CSV row per video.

Filename convention:
    sub-XXX_task-legagilityR_acq-gopro_video.mp4
    sub-XXX_task-legagilityL_acq-gopro_video.mp4

Usage:
    python extract_leg_agility_features.py --input_dir "./videos/" --leg right
    python extract_leg_agility_features.py --input_dir "./videos/" --leg left
    python extract_leg_agility_features.py --input_dir "./videos/" --leg right --cpu --conf 0.2

Signal definition
-----------------
  Let H_t, K_t, A_t in R2 be hip / knee / ankle at frame t.

  Two signals computed per frame:
    x_vert(t) = H_t,y - A_t,y        vertical only (positive = leg raised)
    x_full(t) = ||A_t - H_t||         Euclidean hip-to-ankle distance

  x_vert is used for all feature computation (stable under lateral movement).
  x_full is stored as a QC / alternative column.

  Peaks of x_vert(t) at frames {p_i} = one lift/stomp cycle each.
  IRI_i = (p_{i+1} - p_i) / fps

  Per-cycle amplitude:
    A_i = x_vert(p_i) - min( x_vert[p_i : p_{i+1}] )

Output CSV columns:
    Identity   : video, subject, rel_path, leg, status
    Recording  : duration_s, fps, total_frames, detected_frames, detection_rate
    Speed      : rep_count, reps_per_second, IRI_mean, IRI_CV
    Amplitude  : amp_mean, amp_CV
    Hesitations: pause_count, max_pause_s
    Decrement  : amp_slope
    Raw signal : x_vert_mean, x_vert_std, x_vert_min, x_vert_max, x_vert_range,
                 x_full_mean, x_full_range
"""

import os
import csv
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from mmpose.apis import MMPoseInferencer

# ── KEYPOINT INDEX MAP ────────────────────────────────────────────────────────
#
# All keypoints from the COCO body block (indices 0-16):
#   11 = left_hip     12 = right_hip
#   13 = left_knee    14 = right_knee
#   15 = left_ankle   16 = right_ankle

LEG_CONFIG = {
    "right": {
        "label":   "R",
        "keyword": "task-legagilityR",
        "hip":     12,
        "knee":    14,
        "ankle":   16,
    },
    "left": {
        "label":   "L",
        "keyword": "task-legagilityL",
        "hip":     11,
        "knee":    13,
        "ankle":   15,
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── SIGNAL HELPERS ────────────────────────────────────────────────────────────

def smooth(x, wlen=11):
    if len(x) < wlen:
        return x.copy()
    return savgol_filter(x, window_length=wlen, polyorder=3)


def detect_peaks_signal(x_sm, min_dist_frames=8):
    """
    Peaks of x_sm = leg-lifted positions.
    Adaptive prominence = 10% of signal range.
    """
    sig_range  = float(np.max(x_sm) - np.min(x_sm))
    prominence = max(1.0, 0.10 * sig_range)
    peaks, _   = find_peaks(x_sm, distance=min_dist_frames,
                             prominence=prominence)
    return peaks


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(x_vert_raw, x_full_raw, fps):
    """
    x_vert_raw : H_y - A_y per frame  (positive = leg raised)
    x_full_raw : ||A - H|| per frame  (Euclidean distance)
    fps        : frames per second
    """
    feat = {}
    n        = len(x_vert_raw)
    duration = n / fps

    # ── Raw signal stats ──────────────────────────────────────────────────────
    feat["duration_s"]   = round(duration, 3)
    feat["total_frames"] = n
    feat["x_vert_mean"]  = round(float(np.mean(x_vert_raw)),  3)
    feat["x_vert_std"]   = round(float(np.std(x_vert_raw)),   3)
    feat["x_vert_min"]   = round(float(np.min(x_vert_raw)),   3)
    feat["x_vert_max"]   = round(float(np.max(x_vert_raw)),   3)
    feat["x_vert_range"] = round(float(np.max(x_vert_raw) - np.min(x_vert_raw)), 3)
    feat["x_full_mean"]  = round(float(np.mean(x_full_raw)),  3)
    feat["x_full_range"] = round(float(np.max(x_full_raw) - np.min(x_full_raw)), 3)

    NAN_KEYS = [
        "rep_count", "reps_per_second", "IRI_mean", "IRI_CV",
        "amp_mean", "amp_CV",
        "pause_count", "max_pause_s",
        "amp_slope",
    ]

    # ── Smooth + detect peaks ─────────────────────────────────────────────────
    x_sm  = smooth(x_vert_raw)
    peaks = detect_peaks_signal(x_sm)

    if len(peaks) < 2:
        for k in NAN_KEYS:
            feat[k] = float("nan")
        return feat

    # ── Speed / Rhythm ────────────────────────────────────────────────────────
    rep_count    = len(peaks)
    reps_per_sec = rep_count / duration

    IRI      = np.diff(peaks) / fps
    IRI_mean = float(np.mean(IRI))
    IRI_CV   = float(np.std(IRI)) / IRI_mean if IRI_mean > 0 else float("nan")

    feat["rep_count"]       = rep_count
    feat["reps_per_second"] = round(reps_per_sec, 4)
    feat["IRI_mean"]        = round(IRI_mean, 4)
    feat["IRI_CV"]          = round(IRI_CV,   4) if not np.isnan(IRI_CV) else float("nan")

    # ── Per-cycle amplitude  A_i = x(p_i) - min(x[p_i : p_{i+1}]) ───────────
    amps = []
    for i in range(len(peaks) - 1):
        seg = x_sm[peaks[i]: peaks[i + 1]]
        A_i = float(x_sm[peaks[i]]) - float(np.min(seg))
        amps.append(A_i)

    amps      = np.array(amps)
    amp_mean  = float(np.mean(amps))
    amp_CV    = float(np.std(amps)) / amp_mean if amp_mean > 0 else float("nan")
    amp_slope = float(np.polyfit(np.arange(len(amps)), amps, 1)[0]) \
                if len(amps) >= 3 else float("nan")

    feat["amp_mean"]  = round(amp_mean,  3)
    feat["amp_CV"]    = round(amp_CV,    4) if not np.isnan(amp_CV)    else float("nan")
    feat["amp_slope"] = round(amp_slope, 6) if not np.isnan(amp_slope) else float("nan")

    # ── Hesitations / Halts ───────────────────────────────────────────────────
    pause_thr   = 2.0 * float(np.median(IRI))
    pauses      = IRI[IRI > pause_thr]
    pause_count = int(len(pauses))
    max_pause   = float(np.max(pauses)) if pause_count > 0 else 0.0

    feat["pause_count"] = pause_count
    feat["max_pause_s"] = round(max_pause, 4)

    return feat


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def process_one_video(video_path, inferencer, cfg, conf_thr, fps_override=None):
    """
    Run MMPose on every frame, compute x_vert(t) and x_full(t).

    Returns:
        fps        : float
        detected   : int
        x_vert_raw : np.ndarray  H_y - A_y  (pixels)
        x_full_raw : np.ndarray  ||A - H||  (pixels)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    IDX_HIP   = cfg["hip"]
    IDX_ANKLE = cfg["ankle"]

    x_vert_list = []
    x_full_list = []
    detected    = 0

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

            hip_ok   = scores[IDX_HIP]   >= conf_thr
            ankle_ok = scores[IDX_ANKLE] >= conf_thr

            if hip_ok and ankle_ok:
                H = keypoints[IDX_HIP]
                A = keypoints[IDX_ANKLE]

                x_vert_list.append(float(H[1] - A[1]))          # vertical only
                x_full_list.append(float(np.linalg.norm(A - H))) # Euclidean
                detected += 1
                got_kpt = True

        if not got_kpt and len(x_vert_list) > 0:
            x_vert_list.append(x_vert_list[-1])
            x_full_list.append(x_full_list[-1])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"    [{frame_idx}/{total}]", end="\r", flush=True)

    cap.release()
    print()
    return fps, detected, np.array(x_vert_list), np.array(x_full_list)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Leg Agility – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root (searched recursively)")
    parser.add_argument("--output",    default="updrs_legagility_features.csv",
                        help="Output CSV path  (default: updrs_legagility_features.csv)")
    parser.add_argument("--leg",       default="right", choices=["left", "right"],
                        help="Which leg to analyse  (default: right)")
    parser.add_argument("--conf",      default=0.3,  type=float,
                        help="Keypoint confidence threshold  (default: 0.3)")
    parser.add_argument("--fps",       default=None, type=float,
                        help="Override FPS if video metadata is wrong")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    cfg         = LEG_CONFIG[args.leg]
    leg_keyword = cfg["keyword"]

    # ── Recursive video discovery ─────────────────────────────────────────────
    video_files        = []
    skipped_wrong_task = 0
    skipped_wrong_leg  = 0

    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            fname_lower = fname.lower()
            if "legagility" not in fname_lower:
                skipped_wrong_task += 1
                continue
            if leg_keyword.lower() not in fname_lower:
                skipped_wrong_leg += 1
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
    missing_subjects = [d for d in all_subject_dirs
                        if d not in subjects_with_video]

    if not video_files:
        print(f"[ERROR] No '{leg_keyword}' videos found under: {args.input_dir}")
        print(f"        {skipped_wrong_leg} legagility video(s) skipped (wrong leg)")
        print(f"        {skipped_wrong_task} non-legagility video(s) ignored")
        return

    print(f"[INFO] Dataset root  : {args.input_dir}")
    print(f"[INFO] Leg           : {args.leg.upper()}  (matching '{leg_keyword}')")
    print(f"[INFO] Videos matched: {len(video_files)}")
    print(f"         skipped – wrong leg  : {skipped_wrong_leg}")
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
        "video", "subject", "rel_path", "leg", "status",
        "duration_s", "fps", "total_frames", "detected_frames", "detection_rate",
        "rep_count", "reps_per_second", "IRI_mean", "IRI_CV",
        "amp_mean", "amp_CV",
        "pause_count", "max_pause_s",
        "amp_slope",
        "x_vert_mean", "x_vert_std", "x_vert_min", "x_vert_max", "x_vert_range",
        "x_full_mean", "x_full_range",
    ]

    rows = []

    for i, vpath in enumerate(video_files):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))
        print(f"[{i+1}/{len(video_files)}] {subject}/{vname}")

        try:
            fps, detected, x_vert, x_full = process_one_video(
                vpath, inferencer, cfg, args.conf, args.fps
            )

            if len(x_vert) < 10:
                print(f"  [WARN] Too few detections ({detected}), skipping features.")
                row = {c: float("nan") for c in COLUMNS}
                row.update({"video": vname, "subject": subject,
                            "rel_path": rel_path, "leg": args.leg,
                            "status": "too_few_detections"})
                rows.append(row)
                continue

            feat = compute_features(x_vert, x_full, fps)
            feat.update({
                "video":           vname,
                "subject":         subject,
                "rel_path":        rel_path,
                "leg":             args.leg,
                "status":          "ok",
                "fps":             round(fps, 2),
                "detected_frames": detected,
                "detection_rate":  round(detected / len(x_vert), 4),
            })
            rows.append(feat)

            print(f"  reps={feat['rep_count']}  "
                  f"rate={feat.get('reps_per_second','nan')}  "
                  f"amp_mean={feat.get('amp_mean','nan')} px  "
                  f"IRI_CV={feat.get('IRI_CV','nan')}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            row = {c: float("nan") for c in COLUMNS}
            row.update({"video": vname, "subject": subject,
                        "rel_path": rel_path, "leg": args.leg,
                        "status": f"error: {e}"})
            rows.append(row)

    # ── Append missing-subject rows ───────────────────────────────────────────
    for ms in missing_subjects:
        row = {c: float("nan") for c in COLUMNS}
        row.update({"subject": ms, "leg": args.leg, "status": "missing_video"})
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
