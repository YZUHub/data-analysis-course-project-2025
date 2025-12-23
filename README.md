# Human Activity Recognition Using Wearable Sensor Data  
### CNN-Based Feature Extraction and LightGBM Classification

**Author:** MD Rakibul Hasan  
**Student ID:** 228801134  
**Date:** December 2025  

---

## 📌 Project Overview

This project focuses on **Human Activity Recognition (HAR)** using multi-modal wearable sensor data. A **hybrid learning framework** is proposed that combines:

- **Convolutional Neural Networks (CNNs)** for automatic temporal feature extraction from raw sensor signals
- **Light Gradient Boosting Machine (LightGBM)** for robust multi-class classification

The system is evaluated under a **subject-independent setting**, ensuring realistic deployment performance and preventing data leakage.

---

## 📊 Dataset Description

- **Data type:** Multivariate time-series
- **Sensors:**
  - Inertial Measurement Units (IMUs): hand, chest, ankle
  - Physiological signal: heart rate
- **Sampling rate:** IMU signals at 100 Hz
- **Activities:** 18 daily and sports-related activities  
- **Subjects:** Multiple human subjects with uneven activity coverage

⚠️ A **transient activity class (ID = 0)** representing transitions was removed to reduce label noise.

---

## 🔍 Exploratory Data Analysis (EDA)

Key EDA steps include:

- Activity label distribution analysis (revealed strong class imbalance)
- Activity coverage per subject (not all subjects perform all activities)
- Missing value analysis:
  - Heart rate contained most missing values due to lower sampling frequency
  - IMU channels were nearly complete
- Sensor signal inspection showing:
  - High variance for dynamic activities (e.g., running, cycling)
  - Stable patterns for static activities (e.g., sitting, standing)

These findings motivated:
- Subject-wise splitting
- Use of **Macro F1-score** for fair evaluation

---

## 🧹 Data Preparation Pipeline

The following preprocessing steps were applied:

1. Subject-wise data loading and identification
2. Removal of transient activities (ID = 0)
3. Sensor channel selection:
   - Retained accelerometer (±16g) and gyroscope channels
   - Excluded magnetometer and orientation data
4. Missing value handling:
   - Forward-filling heart rate per subject
   - Median imputation and interpolation for remaining gaps
5. Sliding-window segmentation
6. Normalization using training-set statistics only

All steps were applied **consistently across subjects** to avoid data leakage.

---

## 🧠 Model Architecture

### 1️⃣ Baseline Model
- Statistical features extracted from sliding windows:
  - Mean, standard deviation, min, max
- Ensemble-based classifier
- Serves as an interpretable performance reference

### 2️⃣ CNN-Based Model
- 1D Convolutional Neural Network
- Stacked convolutional layers with increasing filters
- Batch normalization, pooling, dropout
- Global average pooling for compact representation

### 3️⃣ Hybrid CNN + LightGBM
- CNN acts as a **temporal feature extractor**
- LightGBM performs final multi-class classification
- Combines deep representation learning with gradient-boosted decision trees

---

## ⚙️ Training Strategy

- Optimizer: **Adam**
- Fixed learning rate
- Early stopping to mitigate overfitting
- Subject-independent train / validation / test splits
- Normalization applied using training-set statistics only

---

## 📈 Evaluation Metrics

- Accuracy
- Macro F1-score (class-balanced evaluation)
- Weighted F1-score
- Confusion matrix (row-normalized)
- Per-class performance breakdown

---

## 🏆 Results Summary

- The baseline LightGBM model provides a reasonable but limited performance
- The CNN-based model significantly improves **Macro F1-score**
- Hybrid CNN + LightGBM achieves the best balance between accuracy and class-level fairness
- Confusion matrix analysis shows strong performance on frequent activities and reduced recall for rare classes

These results highlight the importance of **temporal modeling** and **class-aware evaluation** in HAR systems.

---

## 📌 Key Contributions

- Subject-independent HAR pipeline with no data leakage
- Comprehensive EDA and preprocessing strategy
- Hybrid CNN–LightGBM architecture for wearable sensor data
- Detailed evaluation under class imbalance
- Clear comparison between feature-based and deep learning approaches

---

## 🔮 Future Work

- Explore recurrent and attention-based temporal models
- Investigate multi-sensor fusion strategies
- Incorporate demographic or contextual subject information
- Improve recognition of rare and complex activities

---

## 📄 Report

The complete project report (PDF) contains:
- Full mathematical formulation
- Detailed figures and plots
- Training curves
- Confusion matrices
- Per-class evaluation

📎 **File:** `report_228801134.pdf`

---

## 📝 License

This project is intended for **academic and educational purposes only**.
