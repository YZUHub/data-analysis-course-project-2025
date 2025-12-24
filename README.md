# Human Activity Recognition (HAR) from Wearable Sensor Data

## Project Overview
This project addresses the problem of Human Activity Recognition (HAR) using multi-sensor time-series data collected from wearable devices. The primary goal is to build robust machine learning models that can accurately classify activities for **unseen subjects**, reflecting a real-world deployment scenario.

## Workflow Summary
- **Data Loading:** Combined all subject CSV files into a single DataFrame.
- **Label Cleaning:** Removed transient/unlabeled activity segments (e.g., `activity_id = 0`).
- **Missing Value Handling:** Imputed heart rate values **within each subject** to avoid subject leakage.
- **Feature Engineering:**
  - Built window-level datasets (window size = 200, step size = 100) within contiguous activity segments.
  - Extracted mean, std, min, max for the **top 30 most informative original features** (selected via Random Forest importances on training subjects only).
- **Modeling:**
  - Used a strict subject-wise split: **Train:** subject101,102,103,105,106,108,109; **Validation:** subject104; **Test:** subject107.
  - Trained and tuned Random Forest, SVM (with scaling), LightGBM, and XGBoost models.
  - Selected the best model based on validation performance and evaluated on the held-out test subject.
- **Evaluation:**
  - Reported accuracy, weighted F1, macro F1, and confusion matrix on the test set.

## Key Results
- **Best Model:** Tuned SVM (`C=0.3, gamma='scale'`)
- **Validation (subject104):** Accuracy = 0.9457, Weighted F1 = 0.9466
- **Test (subject107):** Accuracy = 0.9339, Weighted F1 = 0.9324, Macro F1 = 0.9205
- **Classes:** 12 activities
- **Window counts:** Train: 14,500 | Val: 2,266 | Test: 2,286

## Challenges & Solutions
- **Subject Leakage:** All preprocessing (imputation, scaling, feature selection) was performed using only training data to prevent leakage.
- **High Heart Rate Missingness:** Addressed by forward/backward filling within each subject.
- **Class Imbalance:** Some activities are rare, leading to lower recall/F1 for those classes. This is visible in the confusion matrix and classification report.
- **Time-Series Noise:** Window-based features (vs. per-timestep) improved stability and accuracy.
- **Compute Constraints:** Used stratified downsampling and sampled parameter grids for efficient hyperparameter tuning.

## Recommendations for Future Work
- **Evaluation:** Use leave-one-subject-out cross-validation (LOSO-CV) for more robust generalization estimates.
- **Feature Engineering:** Add richer time-series features (energy, entropy, correlations, frequency-domain, etc.).
- **Class Imbalance:** Apply class weighting, balanced sampling, or targeted augmentation for rare activities.
- **Windowing:** Experiment with different window/step sizes and analyze stability across subjects.
- **Temporal Smoothing:** Consider post-processing predictions with temporal smoothing or sequence models.

## How to Reproduce
1. Place all subject CSV files in the `csvData/` directory.
2. Run the notebook `har.ipynb` from start to finish.
3. Review the final evaluation metrics and confusion matrix for the held-out test subject.

## Acknowledgments
- Data and problem context inspired by public HAR datasets and real-world wearable sensor deployments.
- This project demonstrates best practices for leakage-safe, subject-wise HAR modeling in Python.

---
For questions or suggestions, please contact me.
