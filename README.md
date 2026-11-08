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
├── VAMP-PD_DUA.docx
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

The VAMP-PD dataset (face-blurred videos of all six tasks and MDS-UPDRS labels) is available on Zenodo for non-commercial research under a short, one-page Data Use Agreement (DUA).

Zenodo DOI: **https://doi.org/10.5281/zenodo.21847745**

To request access:

1. Download the DUA: [`VAMP-PD_DUA.docx`](VAMP-PD_DUA.docx).
2. Fill in your details, sign it by typing your name, and save it as a PDF.
3. Email the signed PDF to **mohadisaadat@gmail.com**.
4. Request access on the Zenodo record using the same email address.

Access is granted once the signed DUA is received. Independent researchers and students are welcome to apply.

Pose keypoints and features are not part of the release; they can be generated from the videos with the scripts in this repository.

## Privacy

Released videos are privacy processed. Faces are blurred, audio is removed, and participant identifiers are anonymized.

Because full-body movement videos may retain residual re-identification risk, access to the raw videos is controlled.

## Citation

If you use VAMP-PD, please cite:

```bibtex
@inproceedings{saadatnia2027vamppd,
  title     = {{VAMP-PD}: A Task-Structured Single-Camera Video Benchmark for {MDS-UPDRS} Motor Severity Assessment in {Parkinson's} Disease},
  author    = {Saadatnia, Mohammad Hadi and Haghighi, Ali and Poureh, Amiratabak and Baradaranfard, Mahdi and Saadatnia, Saba and Ahmadi, Mohammad Hossein and Tajmirriahi, Amirhossein and Nasrabadi, Erfan Nasri and Saadatnia, Mohammad and Safavi, Saeid},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Satellite Events},
  series    = {Lecture Notes in Computer Science},
  volume    = {17273},
  publisher = {Springer},
  year      = {2027},
  doi       = {10.1007/978-3-032-43973-4_11}
}
```

## License

Code and dataset licensing information will be provided with the final release. Dataset access is intended for non-commercial research use subject to the terms provided with the Zenodo record.
