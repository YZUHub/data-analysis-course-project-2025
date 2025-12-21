"""
Quick Start Example for Human Activity Recognition
===================================================

This is a simplified version for quick testing and understanding the workflow.
Use this if you want to test with a smaller subset of features or models.

Author: Generated for HAR Project
Date: 2025-11-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# Configuration
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output_quick")
OUTPUT_DIR.mkdir(exist_ok=True)

# Activity labels
ACTIVITY_LABELS = {
    1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking",
    5: "Running", 6: "Cycling", 7: "Nordic Walking", 9: "Watching TV",
    10: "Computer Work", 11: "Car Driving", 12: "Ascending Stairs",
    13: "Descending Stairs", 16: "Vacuum Cleaning", 17: "Ironing",
    18: "Folding Laundry", 19: "House Cleaning", 20: "Playing Soccer",
    24: "Rope Jumping"
}

# Subject demographics
SUBJECT_INFO = pd.DataFrame({
    'subject_id': [101, 102, 103, 104, 105, 106, 107, 108, 109],
    'age': [27, 25, 31, 24, 26, 26, 23, 32, 31],
    'height': [182, 169, 187, 194, 180, 183, 173, 179, 168],
    'weight': [83, 78, 92, 95, 73, 69, 86, 87, 65],
})


def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    all_data = []
    for subject_id in [101, 102, 103, 104, 105]:  # Load first 5 subjects for quick test
        filepath = DATA_DIR / f"subject{subject_id}.dat"
        if filepath.exists():
            data = pd.read_csv(filepath, sep=r'\s+', header=None)
            data['subject_id'] = subject_id
            all_data.append(data)
            print(f"  Loaded subject {subject_id}")
    
    if not all_data:
        print("ERROR: No data files found!")
        print(f"Please place .dat files in {DATA_DIR}")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Create column names (simplified)
    cols = ['timestamp', 'activity_id', 'heart_rate']
    for i in range(3, 54):
        cols.append(f'sensor_{i}')
    cols.append('subject_id')
    df.columns = cols
    
    # Remove transient activities
    df = df[df['activity_id'] != 0].copy()
    
    # Add activity labels
    df['activity_label'] = df['activity_id'].map(ACTIVITY_LABELS)
    
    print(f"Total samples: {len(df)}")
    print(f"Activities: {df['activity_id'].nunique()}")
    
    return df


def extract_simple_features(df, window_size=100):
    """Extract simple features from windows"""
    print("\nExtracting features...")
    
    features_list = []
    labels_list = []
    subjects_list = []
    
    # Use only key sensors for quick processing
    sensor_cols = ['heart_rate', 'sensor_3', 'sensor_4', 'sensor_5',  # Hand acc
                   'sensor_20', 'sensor_21', 'sensor_22',  # Chest acc
                   'sensor_37', 'sensor_38', 'sensor_39']  # Ankle acc
    
    for (subject_id, activity_id), group in df.groupby(['subject_id', 'activity_id']):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # Create windows
        for i in range(0, len(group) - window_size, window_size // 2):
            window = group.iloc[i:i + window_size]
            
            if len(window) == window_size:
                features = {}
                
                # Extract basic statistics for each sensor
                for col in sensor_cols:
                    data = window[col].fillna(method='ffill').fillna(0)
                    features[f'{col}_mean'] = data.mean()
                    features[f'{col}_std'] = data.std()
                    features[f'{col}_max'] = data.max()
                    features[f'{col}_min'] = data.min()
                
                features_list.append(features)
                labels_list.append(activity_id)
                subjects_list.append(subject_id)
    
    features_df = pd.DataFrame(features_list)
    features_df['activity_id'] = labels_list
    features_df['subject_id'] = subjects_list
    
    print(f"Features extracted: {features_df.shape}")
    return features_df


def train_and_evaluate(features_df):
    """Train and evaluate a simple model"""
    print("\nTraining model...")
    
    # Split by subjects
    test_subjects = [101, 102]
    train_df = features_df[~features_df['subject_id'].isin(test_subjects)]
    test_df = features_df[features_df['subject_id'].isin(test_subjects)]
    
    # Prepare features
    feature_cols = [col for col in features_df.columns 
                   if col not in ['activity_id', 'subject_id']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['activity_id']
    X_test = test_df[feature_cols]
    y_test = test_df['activity_id']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Combined Score: {accuracy * f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=[ACTIVITY_LABELS[i] 
                                           for i in sorted(y_test.unique())],
                               zero_division=0))
    
    # Plot activity distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    train_df['activity_label'] = train_df['activity_id'].map(ACTIVITY_LABELS)
    train_df['activity_label'].value_counts().plot(kind='barh')
    plt.title('Training Set - Activity Distribution')
    plt.xlabel('Number of Samples')
    
    plt.subplot(1, 2, 2)
    test_df['activity_label'] = test_df['activity_id'].map(ACTIVITY_LABELS)
    test_df['activity_label'].value_counts().plot(kind='barh')
    plt.title('Test Set - Activity Distribution')
    plt.xlabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quick_results.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {OUTPUT_DIR / 'quick_results.png'}")
    
    return model, accuracy, f1


def main():
    """Main execution"""
    print("="*60)
    print("QUICK START - HUMAN ACTIVITY RECOGNITION")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Extract features
    features_df = extract_simple_features(df)
    
    # Save features
    features_df.to_csv(OUTPUT_DIR / 'features_quick.csv', index=False)
    print(f"Features saved to {OUTPUT_DIR / 'features_quick.csv'}")
    
    # Train and evaluate
    model, accuracy, f1 = train_and_evaluate(features_df)
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    print(f"\nFor full pipeline with more features and models,")
    print(f"run: python human_activity_recognition.py")


if __name__ == "__main__":
    main()
