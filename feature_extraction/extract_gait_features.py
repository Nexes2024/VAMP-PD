"""
MDS-UPDRS Part 3 – Gait Feature Extractor
==========================================
Revised from original. Changes made:
  - Removed nose keypoint and posture_lean feature (not clinically grounded)
  - Renamed stride_amplitude_max → ankle_separation_max
  - Renamed stride_amplitude_std → ankle_separation_std
  - Replaced flat os.listdir with recursive BIDS-aware os.walk
  - Added subject, rel_path, status columns (consistent with other extractors)
  - Added missing-subject detection and reporting
  - Added try/except per-video so one bad file doesn't stop the run

Filename convention:
    sub-XXX_task-gait_acq-gopro_video.mp4

Usage:
    python extract_gait_features.py --input_dir "./videos/"
    python extract_gait_features.py --input_dir "./videos/" --output gait.csv --cpu

Output CSV columns:
    Identity  : video, subject, rel_path, status
    Features  : ankle_separation_max, ankle_separation_std,
                foot_lift_l, foot_lift_r, max_foot_lift,
                arm_swing_l, arm_swing_r, arm_swing_intensity,
                gait_speed

COCO body keypoint indices used:
    5/6  = left/right shoulder   (for torso normalisation)
    9/10 = left/right wrist      (arm swing)
    11/12= left/right hip        (speed, normalisation)
    15/16= left/right ankle      (stride, foot lift)
"""

import os
import torch
import numpy as np
import pandas as pd
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OUTPUT_CSV  = "updrs_gait_features.csv"
BATCH_SIZE  = 512
TASK_KEYWORD = "task-gait"
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
# ─────────────────────────────────────────────────────────────────────────────


# ── GPU SETUP ─────────────────────────────────────────────────────────────────
def setup_device(force_cpu=False):
    if force_cpu:
        print("⚙️  CPU mode forced.")
        return "cpu"
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("⚠️  No GPU detected – running on CPU.")
    return "cpu"


# ── NORMALISATION ─────────────────────────────────────────────────────────────

def normalize_keypoints(kpts):
    """
    Scale-normalise by mean torso length (mid-shoulder → mid-hip).
    kpts shape: (T, 17, 2)
    """
    mid_shoulder = (kpts[:, 5, :] + kpts[:, 6, :])  / 2
    mid_hip      = (kpts[:, 11, :] + kpts[:, 12, :]) / 2

    torso_len  = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    mean_torso = np.mean(torso_len)

    if mean_torso < 1e-6:
        return kpts
    return kpts / mean_torso


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def calculate_gait_features(keypoints_sequence):
    """
    keypoints_sequence : np.ndarray of shape (T, 17, 2)

    Returns dict of gait features.
    """
    kpts = normalize_keypoints(keypoints_sequence)

    # ── Named keypoint slices ─────────────────────────────────────────────────
    l_ankle = kpts[:, 15, :]
    r_ankle = kpts[:, 16, :]
    l_wrist = kpts[:, 9,  :]
    r_wrist = kpts[:, 10, :]
    mid_hip = (kpts[:, 11, :] + kpts[:, 12, :]) / 2

    features = {}

    # ── 1. Ankle separation (renamed from stride_amplitude) ───────────────────
    ankle_dist = np.linalg.norm(l_ankle - r_ankle, axis=1)
    features["ankle_separation_max"] = float(np.max(ankle_dist))
    features["ankle_separation_std"] = float(np.std(ankle_dist))

    # ── 2. Foot lift ──────────────────────────────────────────────────────────
    # Std of vertical ankle position (Y axis); higher std = more lifting
    features["foot_lift_l"]   = float(np.std(l_ankle[:, 1]))
    features["foot_lift_r"]   = float(np.std(r_ankle[:, 1]))
    features["max_foot_lift"] = float(max(features["foot_lift_l"],
                                          features["foot_lift_r"]))

    # ── 3. Arm swing ──────────────────────────────────────────────────────────
    # Std of horizontal wrist position (X axis); higher std = more swing
    features["arm_swing_l"]         = float(np.std(l_wrist[:, 0]))
    features["arm_swing_r"]         = float(np.std(r_wrist[:, 0]))
    features["arm_swing_intensity"] = features["arm_swing_l"] + features["arm_swing_r"]

    # ── 4. Gait speed ─────────────────────────────────────────────────────────
    velocity              = np.diff(mid_hip, axis=0)
    features["gait_speed"] = float(np.mean(np.linalg.norm(velocity, axis=1)))

    # NOTE: nose / posture_lean removed – nose is unreliable during walking
    # and horizontal lean relative to nose is not a validated UPDRS proxy.

    return features


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MDS-UPDRS Gait – batch feature extractor"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root (searched recursively)")
    parser.add_argument("--output",    default=OUTPUT_CSV,
                        help=f"Output CSV path  (default: {OUTPUT_CSV})")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    device     = setup_device(args.cpu)
    inferencer = MMPoseInferencer(pose2d="human", device=device)

    # ── Recursive video discovery ─────────────────────────────────────────────
    video_files        = []
    skipped_wrong_task = 0

    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            if TASK_KEYWORD not in fname.lower():
                skipped_wrong_task += 1
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
        print(f"[ERROR] No '{TASK_KEYWORD}' videos found under: {args.input_dir}")
        print(f"        {skipped_wrong_task} non-gait video(s) ignored")
        return

    print(f"[INFO] Dataset root  : {args.input_dir}")
    print(f"[INFO] Videos matched: {len(video_files)}")
    print(f"         skipped – other task : {skipped_wrong_task}")
    if missing_subjects:
        print(f"[WARN] {len(missing_subjects)} subject folder(s) with no gait video:")
        for ms in missing_subjects:
            print(f"  - {ms}")
    print(f"[INFO] Output        : {args.output}\n")
    for vp in video_files:
        print(f"  + {os.path.relpath(vp, args.input_dir)}")
    print()

    # ── Process each video ────────────────────────────────────────────────────
    data_list = []

    for vpath in tqdm(video_files, desc="Processing"):
        vname    = os.path.basename(vpath)
        rel_path = os.path.relpath(vpath, args.input_dir)
        subject  = os.path.basename(os.path.dirname(vpath))

        try:
            result_generator = inferencer(
                vpath,
                show=False,
                batch_size=BATCH_SIZE,
            )

            video_kpts = []
            for result in result_generator:
                try:
                    kpt = result["predictions"][0][0]["keypoints"]
                    video_kpts.append(kpt)
                except (IndexError, KeyError):
                    continue

            if len(video_kpts) < 10:
                print(f"  [WARN] {subject}/{vname} – too few detections, skipping.")
                data_list.append({
                    "video":    vname,
                    "subject":  subject,
                    "rel_path": rel_path,
                    "status":   "too_few_detections",
                })
                continue

            video_kpts = np.array(video_kpts)   # (T, 17, 2)
            feats      = calculate_gait_features(video_kpts)

            feats["video"]    = vname
            feats["subject"]  = subject
            feats["rel_path"] = rel_path
            feats["status"]   = "ok"
            data_list.append(feats)

        except Exception as e:
            print(f"  [ERROR] {subject}/{vname}: {e}")
            data_list.append({
                "video":    vname,
                "subject":  subject,
                "rel_path": rel_path,
                "status":   f"error: {e}",
            })

    # ── Append missing-subject rows ───────────────────────────────────────────
    for ms in missing_subjects:
        data_list.append({
            "subject": ms,
            "status":  "missing_video",
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(data_list)

    if not df.empty:
        # Identity columns first, then features
        id_cols   = ["video", "subject", "rel_path", "status"]
        feat_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + feat_cols]
        df.to_csv(args.output, index=False)

        n_ok      = (df.status == "ok").sum()
        n_missing = (df.status == "missing_video").sum()
        n_err     = df.status.str.startswith("error").sum()
        n_few     = (df.status == "too_few_detections").sum()

        print(f"\n[DONE] Features saved → {args.output}")
        print(f"       Total rows    : {len(df)}")
        print(f"       OK            : {n_ok}")
        print(f"       Missing video : {n_missing}")
        print(f"       Too few kpts  : {n_few}")
        print(f"       Errors        : {n_err}")
    else:
        print("[WARN] No data extracted.")


if __name__ == "__main__":
    main()
