"""
Human Activity Recognition using Physiological and Motion Sensor Data
=====================================================================

This script implements a complete pipeline for human activity recognition including:
- Data loading and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Model training (Traditional ML + Deep Learning)
- Evaluation and reporting

Author: Generated for HAR Project
Date: 2025-11-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning (optional - requires TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
    DEEP_LEARNING_AVAILABLE = False

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. XGBoost models will be skipped.")
    XGBOOST_AVAILABLE = False

from scipy import stats
from scipy.signal import find_peaks
import joblib


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the project"""
    
    # Data paths
    DATA_DIR = Path("./data")  # Directory containing .dat files
    OUTPUT_DIR = Path("./output")
    MODELS_DIR = Path("./models")
    
    # Activity mapping
    ACTIVITY_LABELS = {
        1: "Lying",
        2: "Sitting", 
        3: "Standing",
        4: "Walking",
        5: "Running",
        6: "Cycling",
        7: "Nordic Walking",
        9: "Watching TV",
        10: "Computer Work",
        11: "Car Driving",
        12: "Ascending Stairs",
        13: "Descending Stairs",
        16: "Vacuum Cleaning",
        17: "Ironing",
        18: "Folding Laundry",
        19: "House Cleaning",
        20: "Playing Soccer",
        24: "Rope Jumping",
        0: "Other (Transient)"  # To be excluded
    }
    
    # Subject demographics
    SUBJECT_INFO = pd.DataFrame({
        'subject_id': [101, 102, 103, 104, 105, 106, 107, 108, 109],
        'sex': ['Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male'],
        'age': [27, 25, 31, 24, 26, 26, 23, 32, 31],
        'height': [182, 169, 187, 194, 180, 183, 173, 179, 168],
        'weight': [83, 78, 92, 95, 73, 69, 86, 87, 65],
        'resting_hr': [75, 74, 68, 58, 70, 60, 60, 66, 54],
        'max_hr': [193, 195, 189, 196, 194, 194, 197, 188, 189],
        'dominant_hand': ['Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Left', 'Right']
    })
    
    # Column indices
    TIMESTAMP_COL = 0
    ACTIVITY_COL = 1
    HR_COL = 2
    HAND_COLS = list(range(3, 20))
    CHEST_COLS = list(range(20, 37))
    ANKLE_COLS = list(range(37, 54))
    
    # IMU sensor structure (17 columns per IMU)
    IMU_FEATURES = [
        'temp',  # Temperature
        'acc16_x', 'acc16_y', 'acc16_z',  # 3D Acceleration (±16g)
        'acc6_x', 'acc6_y', 'acc6_z',  # 3D Acceleration (±6g)
        'gyro_x', 'gyro_y', 'gyro_z',  # 3D Gyroscope
        'mag_x', 'mag_y', 'mag_z',  # 3D Magnetometer
        'orient_1', 'orient_2', 'orient_3', 'orient_4'  # Orientation (invalid)
    ]
    
    # Feature engineering parameters
    WINDOW_SIZE = 100  # 1 second at 100Hz
    WINDOW_OVERLAP = 50  # 50% overlap
    
    # Model parameters
    TEST_SUBJECTS = [101, 102]  # Subjects for test set
    VAL_SUBJECTS = [103]  # Subjects for validation set
    RANDOM_STATE = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """Load and preprocess raw sensor data"""
    
    def __init__(self, config):
        self.config = config
        
    def load_subject_data(self, subject_id):
        """Load data for a single subject"""
        # Try different possible file naming conventions
        possible_files = [
            self.config.DATA_DIR / f"subject{subject_id}.dat",
            self.config.DATA_DIR / f"{subject_id}.dat",
            self.config.DATA_DIR / f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat",
        ]
        
        for filepath in possible_files:
            if filepath.exists():
                print(f"Loading data from {filepath}")
                data = pd.read_csv(filepath, sep=r'\s+', header=None)
                data['subject_id'] = subject_id
                return data
        
        raise FileNotFoundError(f"Could not find data file for subject {subject_id}")
    
    def load_all_data(self):
        """Load data for all subjects"""
        all_data = []
        
        for subject_id in self.config.SUBJECT_INFO['subject_id']:
            try:
                subject_data = self.load_subject_data(subject_id)
                all_data.append(subject_data)
                print(f"Loaded {len(subject_data)} samples for subject {subject_id}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data files found. Please check DATA_DIR configuration.")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal samples loaded: {len(df)}")
        return df
    
    def create_column_names(self):
        """Create descriptive column names"""
        cols = ['timestamp', 'activity_id', 'heart_rate']
        
        for sensor_name in ['hand', 'chest', 'ankle']:
            for feature in self.config.IMU_FEATURES:
                cols.append(f'{sensor_name}_{feature}')
        
        cols.append('subject_id')
        return cols
    
    def preprocess_data(self, df):
        """Preprocess the raw data"""
        # Assign column names
        df.columns = self.create_column_names()
        
        # Remove transient activities (activity_id = 0)
        print(f"\nRemoving transient activities (ID=0)...")
        print(f"Before: {len(df)} samples")
        df = df[df['activity_id'] != 0].copy()
        print(f"After: {len(df)} samples")
        
        # Add activity labels
        df['activity_label'] = df['activity_id'].map(self.config.ACTIVITY_LABELS)
        
        # Merge with subject demographics
        df = df.merge(self.config.SUBJECT_INFO, on='subject_id', how='left')
        
        # Calculate BMI
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        
        # Encode categorical variables
        df['sex_encoded'] = (df['sex'] == 'Male').astype(int)
        df['dominant_hand_encoded'] = (df['dominant_hand'] == 'Right').astype(int)
        
        return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDA:
    """Exploratory Data Analysis"""
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        
    def plot_activity_distribution(self):
        """Plot activity distribution across subjects"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall distribution
        activity_counts = self.df['activity_label'].value_counts()
        axes[0].barh(activity_counts.index, activity_counts.values)
        axes[0].set_xlabel('Number of Samples')
        axes[0].set_title('Activity Distribution (Overall)')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Distribution by subject
        activity_by_subject = pd.crosstab(self.df['subject_id'], self.df['activity_label'])
        activity_by_subject.plot(kind='bar', stacked=True, ax=axes[1], 
                                 figsize=(10, 6), colormap='tab20')
        axes[1].set_xlabel('Subject ID')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Activity Distribution by Subject')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / 'activity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_sensor_comparison(self):
        """Compare sensor readings across different activities"""
        # Sample activities for comparison
        sample_activities = ['Walking', 'Running', 'Sitting', 'Lying']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        sensors = ['hand_acc16_x', 'chest_acc16_y', 'ankle_acc16_z', 'heart_rate']
        
        for idx, sensor in enumerate(sensors):
            for activity in sample_activities:
                activity_data = self.df[self.df['activity_label'] == activity][sensor].dropna()
                if len(activity_data) > 0:
                    axes[idx].hist(activity_data, alpha=0.5, label=activity, bins=50)
            
            axes[idx].set_xlabel(sensor)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {sensor}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / 'sensor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_missing_data(self):
        """Visualize missing data patterns"""
        # Calculate missing percentage for each column
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        if len(missing_pct) > 0:
            plt.figure(figsize=(12, 6))
            missing_pct.plot(kind='bar')
            plt.xlabel('Features')
            plt.ylabel('Missing Percentage (%)')
            plt.title('Missing Data by Feature')
            plt.xticks(rotation=90)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.config.OUTPUT_DIR / 'missing_data.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        summary = {
            'Total Samples': len(self.df),
            'Number of Subjects': self.df['subject_id'].nunique(),
            'Number of Activities': self.df['activity_id'].nunique(),
            'Missing Data (%)': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
        }
        
        # Activity statistics
        activity_stats = self.df.groupby('activity_label').agg({
            'timestamp': 'count',
            'subject_id': 'nunique'
        }).rename(columns={'timestamp': 'samples', 'subject_id': 'num_subjects'})
        
        # Save to file
        with open(self.config.OUTPUT_DIR / 'summary_statistics.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATASET SUMMARY STATISTICS\n")
            f.write("="*60 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("ACTIVITY STATISTICS\n")
            f.write("="*60 + "\n\n")
            f.write(activity_stats.to_string())
        
        return summary, activity_stats
    
    def run_all(self):
        """Run all EDA functions"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\nGenerating activity distribution plots...")
        self.plot_activity_distribution()
        
        print("Generating sensor comparison plots...")
        self.plot_sensor_comparison()
        
        print("Generating missing data visualization...")
        self.plot_missing_data()
        
        print("Generating summary statistics...")
        summary, activity_stats = self.generate_summary_statistics()
        
        print("\nEDA complete! Results saved to output directory.")
        return summary, activity_stats


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """Extract features from time-series sensor data"""
    
    def __init__(self, config):
        self.config = config
        
    def extract_statistical_features(self, window):
        """Extract statistical features from a window of data"""
        features = {}
        
        # For each sensor column
        sensor_cols = [col for col in window.columns if any(
            sensor in col for sensor in ['acc16', 'gyro', 'mag', 'heart_rate']
        )]
        
        for col in sensor_cols:
            data = window[col].dropna()
            
            if len(data) > 0:
                # Basic statistics
                features[f'{col}_mean'] = data.mean()
                features[f'{col}_std'] = data.std()
                features[f'{col}_min'] = data.min()
                features[f'{col}_max'] = data.max()
                features[f'{col}_median'] = data.median()
                features[f'{col}_range'] = data.max() - data.min()
                
                # Percentiles
                features[f'{col}_q25'] = data.quantile(0.25)
                features[f'{col}_q75'] = data.quantile(0.75)
                features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
                
                # Advanced statistics
                features[f'{col}_skew'] = stats.skew(data)
                features[f'{col}_kurtosis'] = stats.kurtosis(data)
                features[f'{col}_energy'] = np.sum(data ** 2) / len(data)
                
                # Zero crossing rate
                if len(data) > 1:
                    features[f'{col}_zcr'] = np.sum(np.diff(np.sign(data)) != 0) / len(data)
        
        return features
    
    def extract_magnitude_features(self, window):
        """Extract magnitude features from 3D sensors"""
        features = {}
        
        for sensor in ['hand', 'chest', 'ankle']:
            for sensor_type in ['acc16', 'gyro', 'mag']:
                x_col = f'{sensor}_{sensor_type}_x'
                y_col = f'{sensor}_{sensor_type}_y'
                z_col = f'{sensor}_{sensor_type}_z'
                
                if all(col in window.columns for col in [x_col, y_col, z_col]):
                    x = window[x_col].fillna(0)
                    y = window[y_col].fillna(0)
                    z = window[z_col].fillna(0)
                    
                    magnitude = np.sqrt(x**2 + y**2 + z**2)
                    
                    features[f'{sensor}_{sensor_type}_magnitude_mean'] = magnitude.mean()
                    features[f'{sensor}_{sensor_type}_magnitude_std'] = magnitude.std()
                    features[f'{sensor}_{sensor_type}_magnitude_max'] = magnitude.max()
        
        return features
    
    def create_windows(self, df):
        """Create sliding windows from time-series data"""
        windows = []
        labels = []
        subjects = []
        
        # Group by subject and activity to maintain continuity
        for (subject_id, activity_id), group in df.groupby(['subject_id', 'activity_id']):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Create sliding windows
            for i in range(0, len(group) - self.config.WINDOW_SIZE, 
                          self.config.WINDOW_SIZE - self.config.WINDOW_OVERLAP):
                window = group.iloc[i:i + self.config.WINDOW_SIZE]
                
                if len(window) == self.config.WINDOW_SIZE:
                    # Extract features
                    stat_features = self.extract_statistical_features(window)
                    mag_features = self.extract_magnitude_features(window)
                    
                    # Combine all features
                    all_features = {**stat_features, **mag_features}
                    
                    # Add demographic features (from first row of window)
                    all_features['age'] = window.iloc[0]['age']
                    all_features['height'] = window.iloc[0]['height']
                    all_features['weight'] = window.iloc[0]['weight']
                    all_features['bmi'] = window.iloc[0]['bmi']
                    all_features['sex_encoded'] = window.iloc[0]['sex_encoded']
                    all_features['dominant_hand_encoded'] = window.iloc[0]['dominant_hand_encoded']
                    all_features['resting_hr'] = window.iloc[0]['resting_hr']
                    all_features['max_hr'] = window.iloc[0]['max_hr']
                    
                    windows.append(all_features)
                    labels.append(activity_id)
                    subjects.append(subject_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(windows)
        features_df['activity_id'] = labels
        features_df['subject_id'] = subjects
        
        return features_df
    
    def create_sequences_for_deep_learning(self, df):
        """Create sequences for deep learning models (LSTM, CNN)"""
        sequences = []
        labels = []
        subjects = []
        
        # Select sensor columns (excluding orientation which is invalid)
        sensor_cols = []
        for sensor in ['hand', 'chest', 'ankle']:
            for feature in ['acc16_x', 'acc16_y', 'acc16_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                sensor_cols.append(f'{sensor}_{feature}')
        sensor_cols.append('heart_rate')
        
        # Group by subject and activity
        for (subject_id, activity_id), group in df.groupby(['subject_id', 'activity_id']):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Create sequences
            for i in range(0, len(group) - self.config.WINDOW_SIZE,
                          self.config.WINDOW_SIZE - self.config.WINDOW_OVERLAP):
                window = group.iloc[i:i + self.config.WINDOW_SIZE]
                
                if len(window) == self.config.WINDOW_SIZE:
                    # Extract sensor data
                    sequence = window[sensor_cols].fillna(method='ffill').fillna(0).values
                    
                    sequences.append(sequence)
                    labels.append(activity_id)
                    subjects.append(subject_id)
        
        return np.array(sequences), np.array(labels), np.array(subjects)


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        
    def prepare_data_splits(self, features_df):
        """Split data by subjects"""
        # Separate by subjects
        train_subjects = [s for s in self.config.SUBJECT_INFO['subject_id'] 
                         if s not in self.config.TEST_SUBJECTS + self.config.VAL_SUBJECTS]
        
        train_df = features_df[features_df['subject_id'].isin(train_subjects)]
        val_df = features_df[features_df['subject_id'].isin(self.config.VAL_SUBJECTS)]
        test_df = features_df[features_df['subject_id'].isin(self.config.TEST_SUBJECTS)]
        
        # Separate features and labels
        feature_cols = [col for col in features_df.columns 
                       if col not in ['activity_id', 'subject_id']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['activity_id']
        
        X_val = val_df[feature_cols]
        y_val = val_df['activity_id']
        
        X_test = test_df[feature_cols]
        y_test = test_df['activity_id']
        
        print(f"\nData splits:")
        print(f"Train: {len(X_train)} samples from subjects {train_subjects}")
        print(f"Validation: {len(X_val)} samples from subjects {self.config.VAL_SUBJECTS}")
        print(f"Test: {len(X_test)} samples from subjects {self.config.TEST_SUBJECTS}")
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, self.config.MODELS_DIR / 'scaler.pkl')
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("Training Random Forest Classifier")
        print("="*60)
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(y_train), 
                                             y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Validation performance
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        self.models['random_forest'] = model
        joblib.dump(model, self.config.MODELS_DIR / 'random_forest.pkl')
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost classifier"""
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost not available, skipping...")
            return None
        
        print("\n" + "="*60)
        print("Training XGBoost Classifier")
        print("="*60)
        
        # Encode labels to start from 0
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(y_train_encoded),
                                             y=y_train_encoded)
        sample_weights = np.array([class_weights[y] for y in y_train_encoded])
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train_encoded, 
                 sample_weight=sample_weights,
                 eval_set=[(X_val, y_val_encoded)],
                 verbose=True)
        
        # Validation performance
        y_val_pred_encoded = model.predict(X_val)
        y_val_pred = le.inverse_transform(y_val_pred_encoded)
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        self.models['xgboost'] = {'model': model, 'label_encoder': le}
        joblib.dump({'model': model, 'label_encoder': le}, 
                   self.config.MODELS_DIR / 'xgboost.pkl')
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name, label_encoder=None):
        """Evaluate model on test set"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Make predictions
        if label_encoder is not None:
            y_test_encoded = label_encoder.transform(y_test)
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[self.config.ACTIVITY_LABELS[i] 
                                               for i in sorted(np.unique(y_test))],
                                   zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, y_test, model_name)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        return accuracy, precision, recall, f1
    
    def plot_confusion_matrix(self, cm, y_test, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        labels = [self.config.ACTIVITY_LABELS[i] for i in sorted(np.unique(y_test))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / f'confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


class DeepLearningTrainer:
    """Train deep learning models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def prepare_sequences(self, sequences, labels, subjects):
        """Prepare sequences for deep learning"""
        # Split by subjects
        train_subjects = [s for s in self.config.SUBJECT_INFO['subject_id']
                         if s not in self.config.TEST_SUBJECTS + self.config.VAL_SUBJECTS]
        
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, self.config.VAL_SUBJECTS)
        test_mask = np.isin(subjects, self.config.TEST_SUBJECTS)
        
        X_train = sequences[train_mask]
        y_train = labels[train_mask]
        
        X_val = sequences[val_mask]
        y_val = labels[val_mask]
        
        X_test = sequences[test_mask]
        y_test = labels[test_mask]
        
        # Normalize
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
        
        X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)
        y_test_encoded = le.transform(y_test)
        
        # Convert to categorical
        num_classes = len(np.unique(y_train))
        y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val_encoded, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test_encoded, num_classes)
        
        print(f"\nDeep Learning Data Shapes:")
        print(f"X_train: {X_train_scaled.shape}, y_train: {y_train_cat.shape}")
        print(f"X_val: {X_val_scaled.shape}, y_val: {y_val_cat.shape}")
        print(f"X_test: {X_test_scaled.shape}, y_test: {y_test_cat.shape}")
        
        return ((X_train_scaled, y_train_cat, y_train), 
                (X_val_scaled, y_val_cat, y_val),
                (X_test_scaled, y_test_cat, y_test),
                le)
    
    def build_lstm_model(self, input_shape, num_classes):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, input_shape, num_classes):
        """Build CNN model"""
        model = Sequential([
            Conv1D(64, 5, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Conv1D(128, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Conv1D(256, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Train deep learning model"""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save(self.config.MODELS_DIR / f'{model_name}.h5')
        
        # Plot training history
        self.plot_training_history(history, model_name)
        
        return model, history
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / f'training_history_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, model, X_test, y_test_cat, y_test_original, 
                      label_encoder, model_name, config):
        """Evaluate deep learning model"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Metrics
        accuracy = accuracy_score(y_test_original, y_pred)
        precision = precision_score(y_test_original, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_original, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_original, y_pred, average='weighted', zero_division=0)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred,
                                   target_names=[config.ACTIVITY_LABELS[i]
                                               for i in sorted(np.unique(y_test_original))],
                                   zero_division=0))
        
        return accuracy, precision, recall, f1


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("HUMAN ACTIVITY RECOGNITION PROJECT")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    
    loader = DataLoader(config)
    
    try:
        df_raw = loader.load_all_data()
        df = loader.preprocess_data(df_raw)
        print(f"\nPreprocessed data shape: {df.shape}")
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nPlease ensure data files are in the correct directory.")
        print("Expected directory structure:")
        print("  ./data/subject101.dat")
        print("  ./data/subject102.dat")
        print("  ...")
        return
    
    # ========================================================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    eda = EDA(df, config)
    summary, activity_stats = eda.run_all()
    
    # ========================================================================
    # 3. FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineering(config)
    
    print("\nExtracting features from sliding windows...")
    features_df = fe.create_windows(df)
    print(f"Features extracted: {features_df.shape}")
    
    # Save features
    features_df.to_csv(config.OUTPUT_DIR / 'features.csv', index=False)
    print(f"Features saved to {config.OUTPUT_DIR / 'features.csv'}")
    
    # ========================================================================
    # 4. TRADITIONAL MACHINE LEARNING
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: TRADITIONAL MACHINE LEARNING")
    print("="*60)
    
    trainer = ModelTrainer(config)
    
    # Prepare data splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data_splits(features_df)
    
    # Train Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train, X_val, y_val)
    trainer.evaluate_model(rf_model, X_test, y_test, 'Random_Forest')
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    if xgb_model is not None:
        trainer.evaluate_model(xgb_model, X_test, y_test, 'XGBoost',
                             label_encoder=trainer.models['xgboost']['label_encoder'])
    
    # ========================================================================
    # 5. DEEP LEARNING (Optional)
    # ========================================================================
    if DEEP_LEARNING_AVAILABLE:
        print("\n" + "="*60)
        print("STEP 5: DEEP LEARNING")
        print("="*60)
        
        print("\nCreating sequences for deep learning...")
        sequences, labels, subjects = fe.create_sequences_for_deep_learning(df)
        print(f"Sequences created: {sequences.shape}")
        
        dl_trainer = DeepLearningTrainer(config)
        
        # Prepare data
        (X_train_dl, y_train_dl, y_train_orig), \
        (X_val_dl, y_val_dl, y_val_orig), \
        (X_test_dl, y_test_dl, y_test_orig), \
        label_encoder = dl_trainer.prepare_sequences(sequences, labels, subjects)
        
        # Train LSTM
        lstm_model = dl_trainer.build_lstm_model(
            input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]),
            num_classes=y_train_dl.shape[1]
        )
        lstm_model, lstm_history = dl_trainer.train_model(
            lstm_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, 'LSTM'
        )
        dl_trainer.evaluate_model(lstm_model, X_test_dl, y_test_dl, y_test_orig,
                                 label_encoder, 'LSTM', config)
        
        # Train CNN
        cnn_model = dl_trainer.build_cnn_model(
            input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]),
            num_classes=y_train_dl.shape[1]
        )
        cnn_model, cnn_history = dl_trainer.train_model(
            cnn_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, 'CNN'
        )
        dl_trainer.evaluate_model(cnn_model, X_test_dl, y_test_dl, y_test_orig,
                                 label_encoder, 'CNN', config)
    
    # ========================================================================
    # 6. GENERATE FINAL REPORT
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 6: GENERATING FINAL REPORT")
    print("="*60)
    
    generate_final_report(trainer.results, config)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    print(f"Models saved to: {config.MODELS_DIR}")


def generate_final_report(results, config):
    """Generate final report with all results"""
    
    with open(config.OUTPUT_DIR / 'final_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("HUMAN ACTIVITY RECOGNITION - FINAL REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROJECT OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write("This project implements a complete pipeline for human activity recognition\n")
        f.write("using physiological and motion sensor data from 9 subjects performing 18\n")
        f.write("different activities.\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("1. Data Loading and Preprocessing\n")
        f.write("   - Loaded data from multiple subjects\n")
        f.write("   - Removed transient activities (ID=0)\n")
        f.write("   - Merged with subject demographics\n\n")
        
        f.write("2. Exploratory Data Analysis\n")
        f.write("   - Analyzed activity distribution\n")
        f.write("   - Visualized sensor readings\n")
        f.write("   - Identified missing data patterns\n\n")
        
        f.write("3. Feature Engineering\n")
        f.write("   - Created sliding windows (100 samples, 50% overlap)\n")
        f.write("   - Extracted statistical features (mean, std, min, max, etc.)\n")
        f.write("   - Computed magnitude features from 3D sensors\n")
        f.write("   - Included demographic features\n\n")
        
        f.write("4. Model Training\n")
        f.write("   - Subject-based train/validation/test split\n")
        f.write("   - Handled class imbalance with class weights\n")
        f.write("   - Trained multiple models (RF, XGBoost, LSTM, CNN)\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # Create results table
        results_data = []
        for model_name, metrics in results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Score (Acc × F1)': f"{metrics['accuracy'] * metrics['f1_score']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        # Best model
        best_model = max(results.items(), 
                        key=lambda x: x[1]['accuracy'] * x[1]['f1_score'])
        f.write(f"BEST MODEL: {best_model[0]}\n")
        f.write(f"  Accuracy: {best_model[1]['accuracy']:.4f}\n")
        f.write(f"  F1-Score: {best_model[1]['f1_score']:.4f}\n")
        f.write(f"  Combined Score: {best_model[1]['accuracy'] * best_model[1]['f1_score']:.4f}\n\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-"*80 + "\n")
        f.write("1. Subject-based splitting ensures realistic evaluation\n")
        f.write("2. Feature engineering significantly improves performance\n")
        f.write("3. Class imbalance handling is crucial for fair evaluation\n")
        f.write("4. Deep learning models can capture temporal dependencies\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-"*80 + "\n")
        f.write("- features.csv: Extracted features\n")
        f.write("- activity_distribution.png: Activity distribution plots\n")
        f.write("- sensor_comparison.png: Sensor comparison plots\n")
        f.write("- confusion_matrix_*.png: Confusion matrices for each model\n")
        f.write("- training_history_*.png: Training history for deep learning models\n")
        f.write("- Models saved in ./models/ directory\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Final report saved to {config.OUTPUT_DIR / 'final_report.txt'}")


if __name__ == "__main__":
    main()
