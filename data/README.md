# Human Activity Recognition Dataset

## Overview

This dataset contains physiological and motion sensor data collected from multiple subjects performing various physical activities over 10 hours with approximately 8 hours of labeled data across 18 different activities. The data can be used for activity classification and recognition tasks using machine learning techniques.

## Dataset Description

### Data Collection

**Subjects**: 9 participants
- 1 female, 8 males
- Age: 27.22 ± 3.31 years
- BMI: 25.11 ± 2.62 kg/m²
- Mix of right-handed (8) and left-handed (1) individuals

| Subject ID | Sex | Age (years) | Height (cm) | Weight (kg) | Resting HR (bpm) | Max HR (bpm) | Dominant Hand |
|------------|-----|-------------|-------------|-------------|------------------|--------------|---------------|
| 101 | Male | 27 | 182 | 83 | 75 | 193 | Right |
| 102 | Female | 25 | 169 | 78 | 74 | 195 | Right |
| 103 | Male | 31 | 187 | 92 | 68 | 189 | Right |
| 104 | Male | 24 | 194 | 95 | 58 | 196 | Right |
| 105 | Male | 26 | 180 | 73 | 70 | 194 | Right |
| 106 | Male | 26 | 183 | 69 | 60 | 194 | Right |
| 107 | Male | 23 | 173 | 86 | 60 | 197 | Right |
| 108 | Male | 32 | 179 | 87 | 66 | 188 | Left |
| 109 | Male | 31 | 168 | 65 | 54 | 189 | Right |

### Activities Included

The dataset contains 18 different activities as listed below:

| ID | Activity | Description                                             |
|----|----------|---------------------------------------------------------|
| 1 | Lying | lying quietly while doing nothing, small movements – e.g. changing the lying posture – are allowed |
| 2 | Sitting | sitting in a chair in whatever posture the subject feels comfortable, changing sitting postures is also allowed |
| 3 | Standing | consists of standing still or standing still and talking, possibly gesticulating |
| 4 | Walking | walking outside with moderate to brisk pace with a speed of 4-6km/h, according to what was suitable for the subject |
| 5 | Running | meant jogging outside with a suitable speed for the individual subjects |
| 6 | Cycling | was performed outside with a real bike with slow to moderate pace, as if the subject would bike to work or bike for pleasure (but not as a sport activity) |
| 7 | Nordic walking | was performed outside on asphaltic terrain, using asphalt pads on the walking poles (it has to be noted, that none of the subjects was very familiar with this sport activity) |
| 9 | Watching TV | watching TV at home, in whatever posture – lying, sitting – the subject feels comfortable |
| 10 | Computer work | working normally in the office |
| 11 | Car driving | driving between office and subject's home |
| 12 | Ascending stairs | was performed in a building between the ground and the top floors, a distance of five floors had to be covered going upstairs |
| 13 | Descending stairs | was performed in a building between the ground and the top floors, a distance of five floors had to be covered going downstairs |
| 16 | Vacuum cleaning | vacuum cleaning one or two office rooms (which includes moving objects, e.g. chairs, placed on the floor) |
| 17 | Ironing | ironing 1-2 shirts or T-shirts |
| 18 | Folding laundry | folding shirts, T-shirts and/or bed linnens |
| 19 | House cleaning | dusting some shelves, including removing books and other things and putting them back again onto the shelves |
| 20 | Playing soccer | playing 1 vs. 1 or 2 vs. 1, running with the ball, dribbling, passing the ball and shooting the ball on goal |
| 24 | Rope jumping | the subjects used the technique most suitable for them, which mainly consisted of the basic jump (where both feet jump at the same time over the rope) or the alternate foot jump (where alternate feet are used to jump off the ground) |
| 0 | Other (transient) | transient activities (transitions between activities, preparation time, etc.) |

**Important**: Data labeled with ID = 0 represents transient activities (transitions between activities, preparation time, etc.) and should be excluded from analysis.

## Data Format

### File Structure

Data files are provided as text files (.dat) with one file per subject per session.

### Column Description

Each file contains **54 columns** organized as follows:

| Column(s) | Description |
|-----------|-------------|
| 1 | Timestamp (seconds) |
| 2 | Activity ID |
| 3 | Heart rate (bpm) |
| 4-20 | IMU Hand sensor data |
| 21-37 | IMU Chest sensor data |
| 38-54 | IMU Ankle sensor data |

### IMU Sensor Data (17 columns per IMU)

Each IMU provides the following measurements:

| Column | Description | Unit | Notes |
|--------|-------------|------|-------|
| 1 | Temperature | °C | |
| 2-4 | 3D Acceleration | m/s² | Scale: ±16g, Resolution: 13-bit (recommended) |
| 5-7 | 3D Acceleration | m/s² | Scale: ±6g, Resolution: 13-bit (not precisely calibrated) |
| 8-10 | 3D Gyroscope | rad/s | |
| 11-13 | 3D Magnetometer | μT | |
| 14-17 | Orientation | - | Invalid in this dataset |

**Recommendation**: Use the first accelerometer (columns 2-4, ±16g scale) as it is better calibrated and less prone to saturation during high-impact movements.

## Data Quality Notes

### Missing Data

Missing values are indicated by `NaN` and occur due to:
1. **Wireless data transmission**: Occasional data packet loss (very rare)
2. **Heart rate sampling**: HR monitor samples at ~9Hz while IMUs sample at 100Hz, resulting in NaN values between HR measurements
3. **Hardware issues**: Connection problems or system crashes during collection

### Data Completeness by Subject

Not all subjects performed all activities. The total labeled data per subject ranges from approximately 1,743 to 4,693 seconds. Some activities were performed by only a subset of subjects.

**Activity Duration by Subject (in seconds)**:

| Activity | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | Total | # Subjects |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------|------------|
| Lying | 271.86 | 234.29 | 220.43 | 230.46 | 236.98 | 233.39 | 256.10 | 241.64 | 0 | 1925.15 | 8 |
| Sitting | 234.79 | 223.44 | 287.60 | 254.91 | 268.63 | 230.40 | 122.81 | 229.22 | 0 | 1851.80 | 8 |
| Standing | 217.16 | 255.75 | 205.32 | 247.05 | 221.31 | 243.55 | 257.50 | 251.59 | 0 | 1899.23 | 8 |
| Walking | 222.52 | 325.32 | 290.35 | 319.31 | 320.32 | 257.20 | 337.19 | 315.32 | 0 | 2387.53 | 8 |
| Running | 212.64 | 92.37 | 0 | 0 | 246.45 | 228.24 | 36.91 | 165.31 | 0 | 981.92 | 6 |
| Cycling | 235.74 | 251.07 | 0 | 226.98 | 245.76 | 204.85 | 226.79 | 254.74 | 0 | 1645.93 | 7 |
| Nordic Walking | 202.64 | 297.38 | 0 | 275.32 | 262.70 | 266.85 | 287.24 | 288.87 | 0 | 1881.00 | 7 |
| Watching TV | 836.45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 836.45 | 1 |
| Computer Work | 0 | 0 | 0 | 0 | 1108.82 | 617.76 | 0 | 687.24 | 685.49 | 3099.31 | 4 |
| Car Driving | 545.18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 545.18 | 1 |
| Ascending Stairs | 158.88 | 173.40 | 103.87 | 166.92 | 142.79 | 132.89 | 176.44 | 116.81 | 0 | 1172.00 | 8 |
| Descending Stairs | 148.97 | 152.11 | 152.72 | 142.83 | 127.25 | 112.70 | 116.16 | 96.53 | 0 | 1049.27 | 8 |
| Vacuum Cleaning | 229.40 | 206.82 | 203.24 | 200.36 | 244.44 | 210.77 | 215.51 | 242.91 | 0 | 1753.45 | 8 |
| Ironing | 235.72 | 288.79 | 279.74 | 249.94 | 330.33 | 377.43 | 294.98 | 329.89 | 0 | 2386.82 | 8 |
| Folding Laundry | 271.13 | 0 | 0 | 0 | 0 | 217.85 | 0 | 236.49 | 273.27 | 998.74 | 4 |
| House Cleaning | 540.88 | 0 | 0 | 0 | 284.87 | 287.13 | 0 | 416.90 | 342.05 | 1871.83 | 5 |
| Playing Soccer | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 181.24 | 287.88 | 469.12 | 2 |
| Rope Jumping | 129.11 | 132.61 | 0 | 0 | 77.32 | 2.55 | 0 | 88.05 | 63.90 | 493.54 | 6 |
| **Labeled Total** | **4693.07** | **2633.35** | **1743.27** | **2314.08** | **4117.97** | **3623.56** | **2327.63** | **4142.75** | **1652.59** | **27248.27** | |
| **Total Recording** | **6957.67** | **4469.99** | **2528.32** | **3295.75** | **5295.54** | **4917.78** | **3135.98** | **5884.41** | **2019.47** | **38504.91** | |

**Notes**:
- A value of 0 indicates the subject did not perform that activity
- "Labeled Total" is the sum of all labeled activity durations per subject
- "Total Recording" includes both labeled activities and transient/unlabeled periods

# Task

Using this dataset, you are required to train and evaluate machine learning models for human activity recognition based on physiological and motion sensor data. Please note that this is a multi-class classification problem with 18 activity classes (excluding the transient class). For this project, you are expected to:

- split the dataset into training, validation, and test sets, ensuring that data from the same subject does not appear in both training and test sets
- explore and visualize the dataset to understand the distribution of activities and sensor readings
- preprocess the data, handle missing values appropriately, and consider the class imbalance in your modeling approach
- use techniques such as feature extraction, feature engineering, normalization, and data augmentation to improve model performance
- provide a detailed analysis of the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score
- you may utilize the subject demographics data provided above as additional features in your model if you find it relevant

If you are confident, you may also explore advanced techniques such as deep learning architectures (e.g., CNNs, RNNs) for this task. Please note that, this is a time-series dataset, so I suggest you to consider temporal dependencies in your modeling approach.

The final deliverable should include the code, a report detailing your methodology, results, and insights gained from the analysis. I will provide a sample submission format later on. There will be separate grading criteria for code quality, model performance, and report clarity.

### The Challenge

The upper threshold for your score will be dynamically adjusted based on the evaluation results of each individual. The accuracy and the F1-score will be the primary metrics for evaluation. The upper threshold for your project score will be set to the product of the highest accuracy and F1-score achieved among all participants. For example, the upper threshold for one student's project score will be calculated as follows - If the **highest accuracy** achieved is `95%` and the **average F1-score** is `90%`, then the upper threshold for that student's project score will be `0.95 * 0.90 = 0.855` or `85.5%`. This dynamic adjustment encourages continuous improvement.

Of course, you can also earn bonus points for additional analyses, visualizations, or insights that go beyond the basic requirements of the project. The bonus point, **not exceeding 25%**, will be added to your project score after the dynamic adjustment. So if your project score after the dynamic adjustment is `70%` and you earn the `full bonus point`, your final score will be `95%`. So, aim to achieve the best possible accuracy and F1-score, and also consider adding extra value to your project through additional analyses or insights.

**PLEASE NOTE**: The scores here accounts only for the project, not the entire course. This is a challenging project that requires a good understanding of machine learning, data preprocessing, and time-series analysis. Make sure to plan your time accordingly. Good luck!
