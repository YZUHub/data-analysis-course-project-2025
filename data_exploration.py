"""
Data Exploration Script for Human Activity Recognition
========================================================

This script helps you explore and understand the dataset before training models.
Run this first to get insights about your data.

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

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./exploration")
OUTPUT_DIR.mkdir(exist_ok=True)

# Activity labels
ACTIVITY_LABELS = {
    1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking",
    5: "Running", 6: "Cycling", 7: "Nordic Walking", 9: "Watching TV",
    10: "Computer Work", 11: "Car Driving", 12: "Ascending Stairs",
    13: "Descending Stairs", 16: "Vacuum Cleaning", 17: "Ironing",
    18: "Folding Laundry", 19: "House Cleaning", 20: "Playing Soccer",
    24: "Rope Jumping", 0: "Transient"
}


def load_all_subjects():
    """Load data from all subjects"""
    print("Loading data from all subjects...")
    all_data = []
    
    for subject_id in range(101, 110):
        filepath = DATA_DIR / f"subject{subject_id}.dat"
        if filepath.exists():
            data = pd.read_csv(filepath, sep=r'\s+', header=None)
            data['subject_id'] = subject_id
            all_data.append(data)
            print(f"  ✓ Subject {subject_id}: {len(data)} samples")
        else:
            print(f"  ✗ Subject {subject_id}: File not found")
    
    if not all_data:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Create column names
    cols = ['timestamp', 'activity_id', 'heart_rate']
    
    # Hand IMU (17 columns)
    for feature in ['temp', 'acc16_x', 'acc16_y', 'acc16_z', 'acc6_x', 'acc6_y', 'acc6_z',
                   'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z',
                   'orient_1', 'orient_2', 'orient_3', 'orient_4']:
        cols.append(f'hand_{feature}')
    
    # Chest IMU (17 columns)
    for feature in ['temp', 'acc16_x', 'acc16_y', 'acc16_z', 'acc6_x', 'acc6_y', 'acc6_z',
                   'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z',
                   'orient_1', 'orient_2', 'orient_3', 'orient_4']:
        cols.append(f'chest_{feature}')
    
    # Ankle IMU (17 columns)
    for feature in ['temp', 'acc16_x', 'acc16_y', 'acc16_z', 'acc6_x', 'acc6_y', 'acc6_z',
                   'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z',
                   'orient_1', 'orient_2', 'orient_3', 'orient_4']:
        cols.append(f'ankle_{feature}')
    
    cols.append('subject_id')
    df.columns = cols
    
    # Add activity labels
    df['activity_label'] = df['activity_id'].map(ACTIVITY_LABELS)
    
    print(f"\nTotal samples loaded: {len(df):,}")
    print(f"Total subjects: {df['subject_id'].nunique()}")
    
    return df


def explore_basic_statistics(df):
    """Explore basic statistics"""
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Subjects: {df['subject_id'].nunique()}")
    print(f"Number of Activities: {df['activity_id'].nunique()}")
    print(f"Total Recording Time: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
    print(f"                      ({(df['timestamp'].max() - df['timestamp'].min())/3600:.2f} hours)")
    
    # Activity distribution
    print("\n" + "-"*80)
    print("Activity Distribution:")
    print("-"*80)
    activity_counts = df['activity_label'].value_counts().sort_index()
    for activity, count in activity_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{activity:20s}: {count:8,} samples ({percentage:5.2f}%)")
    
    # Subject distribution
    print("\n" + "-"*80)
    print("Subject Distribution:")
    print("-"*80)
    subject_counts = df['subject_id'].value_counts().sort_index()
    for subject, count in subject_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Subject {subject}: {count:8,} samples ({percentage:5.2f}%)")
    
    # Missing data
    print("\n" + "-"*80)
    print("Missing Data:")
    print("-"*80)
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if len(missing_pct) > 0:
        print(f"Total missing values: {df.isnull().sum().sum():,}")
        print(f"Columns with missing data: {len(missing_pct)}")
        print("\nTop 10 columns with most missing data:")
        for col, pct in missing_pct.head(10).items():
            print(f"  {col:30s}: {pct:6.2f}%")
    else:
        print("No missing data found!")


def plot_activity_distributions(df):
    """Plot activity distributions"""
    print("\nGenerating activity distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall activity distribution
    ax = axes[0, 0]
    activity_counts = df[df['activity_id'] != 0]['activity_label'].value_counts()
    activity_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Samples')
    ax.set_title('Overall Activity Distribution (Excluding Transient)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Activity distribution by subject
    ax = axes[0, 1]
    activity_by_subject = pd.crosstab(df[df['activity_id'] != 0]['subject_id'], 
                                      df[df['activity_id'] != 0]['activity_label'])
    activity_by_subject.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Activity Distribution by Subject', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Subject participation in activities
    ax = axes[1, 0]
    participation = (activity_by_subject > 0).sum(axis=0).sort_values(ascending=True)
    participation.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Number of Subjects')
    ax.set_title('Subject Participation by Activity', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Transient vs labeled data
    ax = axes[1, 1]
    transient_counts = df.groupby('subject_id')['activity_id'].apply(
        lambda x: pd.Series({'Labeled': (x != 0).sum(), 'Transient': (x == 0).sum()})
    )
    transient_counts.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'lightcoral'])
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Labeled vs Transient Data by Subject', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activity_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'activity_distributions.png'}")
    plt.close()


def plot_sensor_analysis(df):
    """Analyze and plot sensor data"""
    print("\nGenerating sensor analysis plots...")
    
    # Sample data for visualization (to avoid memory issues)
    df_sample = df[df['activity_id'] != 0].groupby('activity_label').apply(
        lambda x: x.sample(min(1000, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    # Plot 1: Accelerometer comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sensors = [
        ('hand_acc16_x', 'Hand Accelerometer X'),
        ('chest_acc16_y', 'Chest Accelerometer Y'),
        ('ankle_acc16_z', 'Ankle Accelerometer Z'),
        ('heart_rate', 'Heart Rate')
    ]
    
    for idx, (sensor, title) in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]
        
        # Select key activities for comparison
        key_activities = ['Walking', 'Running', 'Sitting', 'Lying', 'Ascending Stairs']
        for activity in key_activities:
            if activity in df_sample['activity_label'].values:
                data = df_sample[df_sample['activity_label'] == activity][sensor].dropna()
                if len(data) > 0:
                    ax.hist(data, alpha=0.5, label=activity, bins=30)
        
        ax.set_xlabel(sensor)
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sensor_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'sensor_comparison.png'}")
    plt.close()
    
    # Plot 2: Sensor magnitude analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, sensor_location in enumerate(['hand', 'chest', 'ankle']):
        ax = axes[idx]
        
        # Calculate acceleration magnitude
        x_col = f'{sensor_location}_acc16_x'
        y_col = f'{sensor_location}_acc16_y'
        z_col = f'{sensor_location}_acc16_z'
        
        df_sample['magnitude'] = np.sqrt(
            df_sample[x_col].fillna(0)**2 + 
            df_sample[y_col].fillna(0)**2 + 
            df_sample[z_col].fillna(0)**2
        )
        
        # Box plot by activity
        key_activities = ['Walking', 'Running', 'Sitting', 'Lying']
        data_to_plot = []
        labels_to_plot = []
        
        for activity in key_activities:
            if activity in df_sample['activity_label'].values:
                data = df_sample[df_sample['activity_label'] == activity]['magnitude'].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    labels_to_plot.append(activity)
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels_to_plot)
            ax.set_ylabel('Acceleration Magnitude (m/s²)')
            ax.set_title(f'{sensor_location.capitalize()} Sensor', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sensor_magnitude_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'sensor_magnitude_analysis.png'}")
    plt.close()


def plot_time_series_examples(df):
    """Plot time series examples for different activities"""
    print("\nGenerating time series examples...")
    
    activities_to_plot = ['Walking', 'Running', 'Sitting', 'Lying']
    
    fig, axes = plt.subplots(len(activities_to_plot), 1, figsize=(16, 12))
    
    for idx, activity in enumerate(activities_to_plot):
        ax = axes[idx]
        
        # Get a sample window for this activity
        activity_data = df[(df['activity_label'] == activity) & (df['subject_id'] == 101)]
        
        if len(activity_data) > 0:
            # Take first 500 samples (5 seconds at 100Hz)
            sample = activity_data.head(500).copy()
            sample['time'] = sample['timestamp'] - sample['timestamp'].iloc[0]
            
            # Plot accelerometer data
            ax.plot(sample['time'], sample['hand_acc16_x'], label='Hand X', alpha=0.7)
            ax.plot(sample['time'], sample['hand_acc16_y'], label='Hand Y', alpha=0.7)
            ax.plot(sample['time'], sample['hand_acc16_z'], label='Hand Z', alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.set_title(f'{activity} - Hand Accelerometer', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_series_examples.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'time_series_examples.png'}")
    plt.close()


def plot_correlation_matrix(df):
    """Plot correlation matrix for key sensors"""
    print("\nGenerating correlation matrix...")
    
    # Select key sensor columns
    sensor_cols = [
        'heart_rate',
        'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
        'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
        'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z'
    ]
    
    # Sample data
    df_sample = df[df['activity_id'] != 0].sample(min(10000, len(df)), random_state=42)
    
    # Calculate correlation
    corr = df_sample[sensor_cols].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Key Sensors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")
    plt.close()


def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    print("\nGenerating summary report...")
    
    with open(OUTPUT_DIR / 'exploration_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATA EXPLORATION SUMMARY REPORT\n")
        f.write("Human Activity Recognition Dataset\n")
        f.write("="*80 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Number of Subjects: {df['subject_id'].nunique()}\n")
        f.write(f"Number of Activities: {df['activity_id'].nunique()} (including transient)\n")
        f.write(f"Number of Features: {len(df.columns)}\n")
        f.write(f"Sampling Rate: 100 Hz\n")
        f.write(f"Total Duration: {(df['timestamp'].max() - df['timestamp'].min())/3600:.2f} hours\n\n")
        
        # Activity statistics
        f.write("ACTIVITY STATISTICS\n")
        f.write("-"*80 + "\n")
        activity_stats = df[df['activity_id'] != 0].groupby('activity_label').agg({
            'timestamp': 'count',
            'subject_id': 'nunique'
        }).rename(columns={'timestamp': 'samples', 'subject_id': 'num_subjects'})
        activity_stats['duration_seconds'] = activity_stats['samples'] / 100  # 100Hz
        activity_stats['percentage'] = (activity_stats['samples'] / activity_stats['samples'].sum()) * 100
        f.write(activity_stats.to_string())
        f.write("\n\n")
        
        # Subject statistics
        f.write("SUBJECT STATISTICS\n")
        f.write("-"*80 + "\n")
        subject_stats = df.groupby('subject_id').agg({
            'timestamp': 'count',
            'activity_id': lambda x: (x != 0).sum()
        }).rename(columns={'timestamp': 'total_samples', 'activity_id': 'labeled_samples'})
        subject_stats['transient_samples'] = subject_stats['total_samples'] - subject_stats['labeled_samples']
        subject_stats['labeled_percentage'] = (subject_stats['labeled_samples'] / subject_stats['total_samples']) * 100
        f.write(subject_stats.to_string())
        f.write("\n\n")
        
        # Missing data analysis
        f.write("MISSING DATA ANALYSIS\n")
        f.write("-"*80 + "\n")
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            f.write(f"Total missing values: {df.isnull().sum().sum():,}\n")
            f.write(f"Percentage of missing data: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%\n")
            f.write(f"Columns with missing data: {len(missing_pct)}\n\n")
            f.write("Top 20 columns with most missing data:\n")
            for col, pct in missing_pct.head(20).items():
                f.write(f"  {col:30s}: {pct:6.2f}%\n")
        else:
            f.write("No missing data found!\n")
        f.write("\n")
        
        # Sensor statistics
        f.write("SENSOR STATISTICS\n")
        f.write("-"*80 + "\n")
        key_sensors = ['heart_rate', 'hand_acc16_x', 'chest_acc16_y', 'ankle_acc16_z']
        sensor_stats = df[key_sensors].describe()
        f.write(sensor_stats.to_string())
        f.write("\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR MODELING\n")
        f.write("-"*80 + "\n")
        f.write("1. Handle missing data:\n")
        f.write("   - Heart rate: Forward fill (HR sampled at ~9Hz)\n")
        f.write("   - IMU sensors: Interpolation or forward fill\n")
        f.write("   - Consider removing features with >50% missing data\n\n")
        
        f.write("2. Address class imbalance:\n")
        f.write("   - Use class weights in model training\n")
        f.write("   - Consider oversampling minority classes\n")
        f.write("   - Use stratified sampling for train/test split\n\n")
        
        f.write("3. Feature engineering:\n")
        f.write("   - Extract statistical features from windows\n")
        f.write("   - Compute magnitude from 3D sensors\n")
        f.write("   - Consider frequency domain features (FFT)\n")
        f.write("   - Use subject demographics as additional features\n\n")
        
        f.write("4. Model selection:\n")
        f.write("   - Start with Random Forest for baseline\n")
        f.write("   - Try XGBoost for better performance\n")
        f.write("   - Consider LSTM/CNN for temporal patterns\n")
        f.write("   - Use subject-based cross-validation\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  Saved: {OUTPUT_DIR / 'exploration_summary.txt'}")


def main():
    """Main execution"""
    print("="*80)
    print("DATA EXPLORATION - HUMAN ACTIVITY RECOGNITION")
    print("="*80)
    
    try:
        # Load data
        df = load_all_subjects()
        
        # Basic statistics
        explore_basic_statistics(df)
        
        # Generate plots
        plot_activity_distributions(df)
        plot_sensor_analysis(df)
        plot_time_series_examples(df)
        plot_correlation_matrix(df)
        
        # Generate report
        generate_summary_report(df)
        
        print("\n" + "="*80)
        print("EXPLORATION COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        print("  - activity_distributions.png")
        print("  - sensor_comparison.png")
        print("  - sensor_magnitude_analysis.png")
        print("  - time_series_examples.png")
        print("  - correlation_matrix.png")
        print("  - exploration_summary.txt")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"  1. Data files are in {DATA_DIR}/")
        print("  2. Files are named as subject101.dat, subject102.dat, etc.")
        print("  3. Files are in the correct format (space-separated values)")


if __name__ == "__main__":
    main()
