# Human Activity Recognition Using Ensemble and Deep Learning Methods

**Author:** Md Talat Mahmud Tomal  
**Student ID:** 228801146

## Overview
This repository contains a course project report and implementation for **Human Activity Recognition (HAR)** using wearable sensor data (IMUs + heart rate). The work compares:

- **Tabular / engineered-feature models:** Random Forest, CatBoost
- **Sequential deep learning model:** Bidirectional LSTM (BiLSTM) over sliding windows

The main deliverable is the notebook [main.ipynb](main.ipynb).

## Dataset
The dataset description and column layout are documented in [data/README.md](data/README.md). In short:

- 9 subjects (IDs 101–109)
- 18 activity classes (excluding transient/unlabeled activity ID = 0)
- 54 sensor columns per timestep (timestamp, activity_id, heart_rate, and 3 IMUs: hand/chest/ankle)

### Data files
The notebook expects per-subject data files named like:

- `data/subject101.dat`
- `data/subject102.dat`
- ...

If your data is stored elsewhere, update the `DATA_DIR` variable inside [main.ipynb](main.ipynb).

> Note: In the current notebook, `DATA_DIR` is set to a Kaggle path. For local execution, point it to your local `data/` folder.

## Project structure
- [main.ipynb](main.ipynb): Full report + code (EDA → preprocessing → modeling → evaluation)
- [data/README.md](data/README.md): Dataset documentation
- `data/subject*.dat`: Raw data files (not included here unless you add them)

## Setup
### 1) Create an environment (recommended)
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Running the notebook
1. Open [main.ipynb](main.ipynb) in VS Code (Jupyter) or JupyterLab.
2. Set `DATA_DIR` to the folder containing the `subject*.dat` files.
3. Run all cells.

## Notes on evaluation protocol
The notebook uses **stratified random splitting** to preserve class proportions and class coverage across train/validation/test.

- This is useful for balanced evaluation across classes.
- It can be **optimistic** if your goal is strict generalization to unseen subjects (because subjects/time points may be mixed across splits).

If required by your course rubric, consider re-running with a strict **subject-wise split**.
