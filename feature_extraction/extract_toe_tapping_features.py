"""
MDS-UPDRS Part 3 – Toe Tapping Feature Extractor
=================================================
Processes every matching video under a BIDS-style dataset tree and writes
one CSV row per video.

Filename convention:
    sub-XXX_task-toetappingR_acq-gopro_video.mp4
    sub-XXX_task-toetappingL_acq-gopro_video.mp4

Usage:
    python extract_toe_tapping_features.py --input_dir "./videos/" --foot right
    python extract_toe_tapping_features.py --input_dir "./videos/" --foot left
    python extract_toe_tapping_features.py --input_dir "./videos/" --foot right --cpu --conf 0.2

Signal definition
-----------------
  Ankle-anchored coordinates (removes camera/body drift):
      big_toe' = big_toe − ankle
      heel'    = heel    − ankle        (if heel confident, else zero)

  Lift signal:
      L(t) = heel'_y − big_toe'_y       (positive when toe is raised)

  If heel is not reliably detected, falls back to:
      L(t) = −big_toe'_y                (toe y relative to ankle only)

  Taps = local MAXIMA of L(t)  (toe fully lifted = peak of lift signal)

Output CSV columns:
    Identity   : video, subject, rel_path, foot, status
    Recording  : duration_s, fps, total_frames, detected_frames, detection_rate
    Speed      : tap_count, taps_per_second, ITI_mean, ITI_CV
    Amplitude  : amp_mean, amp_CV
    Hesitations: pause_count, max_pause_s
    Decrement  : amp_slope
    Raw signal : L_mean, L_std, L_min, L_max, L_range,
                 heel_available_fraction
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
#   Body  0-16  (COCO)
#   Foot 17-22  : left_big_toe=17  left_small_toe=18  left_heel=19
#                 right_big_toe=20 right_small_toe=21 right_heel=22
#   Face 23-90  | L-Hand 91-111   | R-Hand 112-132
#
# COCO body ankle indices:
#   15 = left_ankle   16 = right_ankle

FOOT_CONFIG = {
    "right": {
        "label":   "R",
        "keyword": "task-toetappingR",
        "ankle":   16,
        "big_toe": 20,
        "heel":    22,
    },
    "left": {
        "label":   "L",
        "keyword": "task-toetappingL",
        "ankle":   15,
        "big_toe": 17,
        "heel":    19,
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── SIGNAL HELPERS ────────────────────────────────────────────────────────────

def smooth(x, wlen=11):
    if len(x) < wlen:
        return x.copy()
    return savgol_filter(x, window_length=wlen, polyorder=3)


def detect_taps(L_smooth, min_dist_frames=8):
    """
    Taps = local MAXIMA of L(t)  (toe fully lifted).
    Adaptive prominence = 10 % of signal range.
    """
    sig_range  = float(np.max(L_smooth) - np.min(L_smooth))
    prominence = max(1.0, 0.10 * sig_range)
    peaks, _   = find_peaks(L_smooth, distance=min_dist_frames,
                             prominence=prominence)
    return peaks


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(L_raw, heel_avail_frac, fps):
    """
    L_raw           : lift signal array (pixels, ankle-anchored)
    heel_avail_frac : fraction of frames where heel was used (vs ankle fallback)
    fps             : frames per second

    Returns dict of all features.
    """
    feat = {}
    n        = len(L_raw)
    duration = n / fps

    # ── Raw signal stats ──────────────────────────────────────────────────────
    feat["duration_s"]            = round(duration, 3)
    feat["total_frames"]          = n
    feat["L_mean"]                = round(float(np.mean(L_raw)), 3)
    feat["L_std"]                 = round(float(np.std(L_raw)),  3)
    feat["L_min"]                 = round(float(np.min(L_raw)),  3)
    feat["L_max"]                 = round(float(np.max(L_raw)),  3)
    feat["L_range"]               = round(float(np.max(L_raw) - np.min(L_raw)), 3)
    feat["heel_available_fraction"] = round(heel_avail_frac, 4)

    NAN_KEYS = [
        "tap_count", "taps_per_second",
        "ITI_mean", "ITI_CV",
        "amp_mean", "amp_CV",
        "pause_count", "max_pause_s",
        "amp_slope",
    ]

    L_sm = smooth(L_raw)
    taps = detect_taps(L_sm)

    if len(taps) < 2:
        for k in NAN_KEYS:
            feat[k] = float("nan")
        return feat

    # ── Speed / Rhythm ────────────────────────────────────────────────────────
    tap_count    = len(taps)
    taps_per_sec = tap_count / duration

    ITI      = np.diff(taps) / fps
    ITI_mean = float(np.mean(ITI))
    ITI_CV   = float(np.std(ITI)) / ITI_mean if ITI_mean > 0 else float("nan")

    feat["tap_count"]       = tap_count
    feat["taps_per_second"] = round(taps_per_sec, 4)
    feat["ITI_mean"]        = round(ITI_mean, 4)
    feat["ITI_CV"]          = round(ITI_CV,   4) if not np.isnan(ITI_CV) else float("nan")

    # ── Amplitude per tap ─────────────────────────────────────────────────────
    # A_k = L_sm[tap_k] − nearest preceding valley (toe-down baseline)
    valleys, _ = find_peaks(-L_sm,
                            distance=8,
                            prominence=max(1.0, 0.05 * feat["L_range"]))

    amps = []
    for pk in taps:
        preceding = valleys[valleys < pk]
        if len(preceding) > 0:
            ref = preceding[-1]
        else:
            following = valleys[valleys > pk]
            ref = following[0] if len(following) > 0 else None

        if ref is not None:
            amps.append(float(L_sm[pk]) - float(L_sm[ref]))

    if len(amps) == 0:
        for k in ["amp_mean", "amp_CV", "amp_slope"]:
            feat[k] = float("nan")
    else:
        amps     = np.array(amps)
        amp_mean = float(np.mean(amps))
        amp_CV   = float(np.std(amps)) / amp_mean if amp_mean > 0 else float("nan")
        amp_slope = float(np.polyfit(np.arange(len(amps)), amps, 1)[0]) \
                    if len(amps) >= 3 else float("nan")

        feat["amp_mean"]  = round(amp_mean,  3)
        feat["amp_CV"]    = round(amp_CV,    4) if not np.isnan(amp_CV)    else float("nan")
        feat["amp_slope"] = round(amp_slope, 6) if not np.isnan(amp_slope) else float("nan")

    # ── Hesitations / Halts ───────────────────────────────────────────────────
    pause_thr   = 2.0 * float(np.median(ITI))
    pauses      = ITI[ITI > pause_thr]
    pause_count = int(len(pauses))
    max_pause   = float(np.max(pauses)) if pause_count > 0 else 0.0

    feat["pause_count"] = pause_count
    feat["max_pause_s"] = round(max_pause, 4)

    return feat


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def process_one_video(video_path, inferencer, cfg, conf_thr, fps_override=None):
    """
    Run MMPose on every frame, compute the lift signal L(t).

    Returns:
        fps             : float
        detected        : int   (frames with ankle + big_toe confident)
        L_raw           : np.ndarray  (lift signal, pixels)
        heel_avail_frac : float  (fraction of detected frames with heel)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    IDX_ANKLE   = cfg["ankle"]
    IDX_BIG_TOE = cfg["big_toe"]
    IDX_HEEL    = cfg["heel"]

    L_list         = []
    heel_used      = 0
    detected       = 0

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

            ankle_ok   = scores[IDX_ANKLE]   >= conf_thr
            big_toe_ok = scores[IDX_BIG_TOE] >= conf_thr

            if ankle_ok and big_toe_ok:
                A       = keypoints[IDX_ANKLE]
                T_anch  = keypoints[IDX_BIG_TOE] - A   # ankle-anchored big toe

                if scores[IDX_HEEL] >= conf_thr:
                    H_anch = keypoints[IDX_HEEL] - A    # ankle-anchored heel
                    # L = heel'_y − big_toe'_y  (positive when toe raised)
                    L = float(H_anch[1] - T_anch[1])
                    heel_used += 1
                else:
                    # Fallback: no heel → use negative anchored toe_y
                    # (toe above ankle = negative y = positive lift)
                    L = float(-T_anch[1])

                L_list.append(L)
                detected += 1
                got_kpt = True

        # Forward-fill to keep signal continuous
        if not got_kpt and len(L_list) > 0:
            L_list.append(L_list[-1])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"    [{frame_idx}/{total}]", end="\r", flush=True)

    cap.release()
    print()

    heel_frac = heel_used / detected if detected > 0 else 0.0
    return fps, detected, np.array(L_list), heel_frac


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Toe Tapping – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root (searched recursively)")
    parser.add_argument("--output",    default="updrs_toetapping_features.csv",
                        help="Output CSV path  (default: updrs_toetapping_features.csv)")
    parser.add_argument("--foot",      default="right", choices=["left", "right"],
                        help="Which foot to analyse  (default: right)")
    parser.add_argument("--conf",      default=0.3,  type=float,
                        help="Keypoint confidence threshold  (default: 0.3)")
    parser.add_argument("--fps",       default=None, type=float,
                        help="Override FPS if video metadata is wrong")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    cfg          = FOOT_CONFIG[args.foot]
    foot_keyword = cfg["keyword"]   # e.g. "task-toetappingR"

    # ── Recursive video discovery ─────────────────────────────────────────────
    video_files        = []
    skipped_wrong_task = 0
    skipped_wrong_foot = 0

    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            fname_lower = fname.lower()
            if "toetapping" not in fname_lower:
                skipped_wrong_task += 1
                continue
            if foot_keyword.lower() not in fname_lower:
                skipped_wrong_foot += 1
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
        print(f"[ERROR] No '{foot_keyword}' videos found under: {args.input_dir}")
        print(f"        {skipped_wrong_foot} toetapping video(s) skipped (wrong foot)")
        print(f"        {skipped_wrong_task} non-toetapping video(s) ignored")
        return

    print(f"[INFO] Dataset root  : {args.input_dir}")
    print(f"[INFO] Foot          : {args.foot.upper()}  (matching '{foot_keyword}')")
    print(f"[INFO] Videos matched: {len(video_files)}")
    print(f"         skipped – wrong foot : {skipped_wrong_foot}")
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
        "video", "subject", "rel_path", "foot", "status",
        # Recording
        "duration_s", "fps", "total_frames", "detected_frames", "detection_rate",
        # Speed / Rhythm
        "tap_count", "taps_per_second", "ITI_mean", "ITI_CV",
        # AmplitudeRun
        "amp_mean", "amp_CV",
        # Hesitations
        "pause_count", "max_pause_s",
        # Decrement
        "amp_slope",
        # Raw signal
        "L_mean", "L_std", "L_min", "L_max", "L_range",
        "heel_available_fraction",
    ]

    rows = []

    for i, vpath in enumerate(video_files):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))
        print(f"[{i+1}/{len(video_files)}] {subject}/{vname}")

        try:
            fps, detected, L_raw, heel_frac = process_one_video(
                vpath, inferencer, cfg, args.conf, args.fps
            )

            if len(L_raw) < 10:
                print(f"  [WARN] Too few detections ({detected}), skipping features.")
                row = {c: float("nan") for c in COLUMNS}
                row.update({"video": vname, "subject": subject,
                            "rel_path": rel_path, "foot": args.foot,
                            "status": "too_few_detections"})
                rows.append(row)
                continue

            feat = compute_features(L_raw, heel_frac, fps)

            feat["video"]           = vname
            feat["subject"]         = subject
            feat["rel_path"]        = rel_path
            feat["foot"]            = args.foot
            feat["status"]          = "ok"
            feat["fps"]             = round(fps, 2)
            feat["detected_frames"] = detected
            feat["detection_rate"]  = round(detected / len(L_raw), 4)

            rows.append(feat)

            print(f"  taps={feat['tap_count']}  "
                  f"rate={feat.get('taps_per_second','nan')}  "
                  f"amp_mean={feat.get('amp_mean','nan')} px  "
                  f"ITI_CV={feat.get('ITI_CV','nan')}  "
                  f"heel={heel_frac*100:.0f}%")

        except Exception as e:
            print(f"  [ERROR] {e}")
            row = {c: float("nan") for c in COLUMNS}
            row.update({"video": vname, "subject": subject,
                        "rel_path": rel_path, "foot": args.foot,
                        "status": f"error: {e}"})
            rows.append(row)

    # ── Append missing-subject rows ───────────────────────────────────────────
    for ms in missing_subjects:
        row = {c: float("nan") for c in COLUMNS}
        row.update({"subject": ms, "foot": args.foot, "status": "missing_video"})
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