# Human Activity Recognition Using LightGBM and Hybrid CNN Features

## 👤 Author Information
- **Name:** Biswas Sougato (李安)  
- **Student ID:** 228801152  
- **Course:** Data Analysis  
- **Year:** 2025  

---

## 📌 Project Overview
This project implements a **subject-independent Human Activity Recognition (HAR)** system using wearable sensor data. The objective is to classify daily human activities from physiological and motion sensor time-series signals while preventing data leakage and ensuring strong generalization to unseen subjects.

Two modeling strategies are explored:
- A **baseline LightGBM model** using engineered statistical features
- A **hybrid CNN → LightGBM model** that combines deep feature learning with gradient boosting

This work was completed as part of a **Data Analysis Course Project (2025)**.

---

## 📂 Repository Structure


---

## 📊 Dataset Description
- **Subjects:** 9 participants  
- **Sensors:**  
  - Inertial Measurement Units (IMUs) on **hand, chest, and ankle**
  - **Heart rate** sensor  
- **Signals:** Accelerometer, gyroscope, magnetometer  
- **Sampling rate:** ~100 Hz  
- **Activities:** 18 daily activities (static and dynamic)  
- **Note:** Transitional activities (activity ID = 0) are removed  
- **Challenge:** Class imbalance and subject dependency  

The dataset is well suited for **subject-independent human activity recognition**.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removal of transient activity samples  
- Handling of missing values  
- Multi-class label encoding  
- **Subject-wise train / validation / test split** to prevent data leakage  

### 2️⃣ Windowing & Feature Engineering
- Sliding window segmentation with fixed window and step size  
- Majority voting for window labels  
- Statistical time-domain feature extraction  
- Feature normalization (fit on training data only)  

### 3️⃣ Baseline Model (LightGBM)
- Gradient boosting decision tree classifier  
- Trained on engineered statistical features  
- Class weighting used to address imbalance  
- Efficient and interpretable baseline model  

### 4️⃣ Deep Feature Learning (CNN)
- 1D Convolutional Neural Network used as a **feature extractor**
- Learns temporal and cross-channel representations
- CNN is not used as a standalone classifier  

### 5️⃣ Hybrid Model (CNN → LightGBM)
- CNN-extracted features fused with statistical features  
- LightGBM used as the final classifier  
- Combines deep representation learning with robust gradient boosting  

---

## 📈 Evaluation Metrics
- Accuracy  
- Macro F1-score  
- Confusion Matrix  
- Per-class F1-score analysis  
- Comparative visualizations (bar charts and radar charts)

Evaluation is performed **only on unseen test subjects**.

---

## 🧪 Experimental Results (Summary)
- The baseline LightGBM model demonstrates strong performance using engineered features.
- The hybrid CNN → LightGBM model further improves robustness by leveraging learned temporal representations.
- Subject-wise evaluation confirms strong generalization across unseen individuals.

Detailed results and figures are provided in the **project report**.

---

## 🛠️ Installation & Requirements

### Install dependencies
```bash
pip install -r requirements.txt




## 🛠️ Installation & Requirements

### Install dependencies
```bash
pip install -r requirements.txt
