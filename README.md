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
- 12 activity classes (excluding transient/unlabeled activity ID = 0)
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