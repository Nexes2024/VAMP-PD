# VAMP-PD

Code and benchmark resources for **VAMP-PD: A Task-Structured Single-Camera Video Benchmark for MDS-UPDRS Motor Severity Assessment in Parkinson’s Disease**.

VAMP-PD contains recordings of 58 participants performing six MDS-UPDRS Part III motor tasks:

* Finger tapping
* Hand movements
* Pronation–supination
* Toe tapping
* Leg agility
* Gait

The benchmark uses pose-derived kinematic features and task-specific Random Forest classifiers for MDS-UPDRS severity prediction.

## Repository structure

```text
VAMP-PD/
├── feature_extraction/
│   ├── extract_finger_tapping_features.py
│   ├── extract_hand_movement_features.py
│   ├── extract_pronation_supination_features.py
│   ├── extract_toe_tapping_features.py
│   ├── extract_leg_agility_features.py
│   └── extract_gait_features.py
├── evaluation/
│   ├── finger_tapping_pipeline.py
│   ├── run_finger_tapping.py
│   ├── hand_movement_rf.py
│   ├── pronation_supination_rf.py
│   ├── toe_tapping_rf.py
│   ├── leg_agility_rf.py
│   └── gait_rf.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Nexes2024/VAMP-PD.git
cd VAMP-PD

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The feature-extraction pipeline was developed using **MMPose 1.3.2**.

## Feature extraction

Each task has a dedicated feature-extraction script.

Example:

```bash
python feature_extraction/extract_finger_tapping_features.py \
    --input_dir /path/to/videos \
    --hand right \
    --output finger_tapping_features.csv
```

Run a script with `--help` to view its task-specific options.

## Baseline evaluation

Evaluation scripts use Random Forest classifiers with MDI-based feature ranking and top-k feature selection.

For the five side-specific tasks, evaluation is grouped by subject so recordings from the same participant do not appear in both training and test folds. Gait uses 5-fold stratified cross-validation.

Example:

```bash
python evaluation/hand_movement_rf.py \
    --data /path/to/hand_movement_features.xlsx \
    --outdir outputs/hand_movement
```

The finger-tapping benchmark uses a nested cross-validation procedure for top-k feature selection.

### Evaluation input format

Evaluation scripts expect an Excel spreadsheet containing:

- `subject`: anonymized subject identifier
- `score`: MDS-UPDRS severity score (0–3)
- task-specific feature columns produced by the feature-extraction scripts

Metadata and acquisition columns such as video name, relative path, side, frame count, FPS, detection rate, and status may also be present; these are excluded from model features by the evaluation scripts.

## Dataset access

The VAMP-PD dataset is released under controlled access for non-commercial research use.

Zenodo DOI:

**https://doi.org/10.5281/zenodo.21847745**

The dataset record is currently being finalized. Access conditions include restrictions on re-identification and redistribution of participant videos.

## Privacy

Released videos are privacy processed. Faces are blurred, audio is removed, and participant identifiers are anonymized.

Because full-body movement videos may retain residual re-identification risk, access to the raw videos is controlled.

## Citation

Citation information will be updated when the final publication record is available.

## License

Code and dataset licensing information will be provided with the final release. Dataset access is intended for non-commercial research use subject to the terms provided with the Zenodo record.
