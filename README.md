# Machine Sensor Anomaly Detection using Isolation Forest

An end-to-end **Machine Sensor Anomaly Detection** project using
**Isolation Forest**, an unsupervised machine learning algorithm
designed to identify unusual observations in machine telemetry data.

The project analyzes machine sensor readings such as vibration,
temperature, pressure, acoustic noise, rotation speed, and power
consumption to identify records that behave differently from normal
machine patterns.

> **Note:** This project uses a statistical 3-sigma rule as a
> **reference label** for evaluation. These labels are not confirmed
> machine-failure ground truth.

------------------------------------------------------------------------

## 📌 Project Overview

Machine failures can be preceded by unusual sensor behavior. Detecting
these abnormal patterns early can help with predictive maintenance and
reduce unexpected downtime.

In this project:

-   Machine telemetry data is loaded and cleaned.
-   Missing and infinite values are handled.
-   Sensor distributions and correlations are explored.
-   A statistical 3-sigma rule creates reference anomaly labels.
-   Sensor features are standardized.
-   A baseline Isolation Forest model is trained.
-   Anomaly scores and predictions are analyzed.
-   Multiple Isolation Forest parameter combinations are tested.
-   The best configuration is selected using F1-score against the
    reference labels.
-   A final model detects anomalies.
-   Detected anomalies are exported to a CSV file.

------------------------------------------------------------------------

## 🎯 Objective

The main objective is to build an anomaly detection pipeline that can:

1.  Process machine sensor telemetry.
2.  Identify unusual machine behavior without requiring manually labeled
    failure data.
3.  Compare Isolation Forest predictions with a statistical reference
    rule.
4.  Tune important Isolation Forest parameters.
5.  Produce a final list of detected anomalies.

------------------------------------------------------------------------

## 🧠 Algorithm Used

### Isolation Forest

**Isolation Forest** is an unsupervised anomaly detection algorithm.

The basic idea is simple:

> Anomalous observations are usually easier to isolate than normal
> observations.

The algorithm builds random trees by repeatedly selecting:

-   a feature
-   a random split value

An unusual data point often becomes isolated using fewer splits.

### Isolation Forest Output

  Isolation Forest Output   Meaning   Project Binary Label
  ------------------------- --------- ----------------------
  `1`                       Normal    `0`
  `-1`                      Anomaly   `1`

The notebook converts the original Isolation Forest output into binary
labels so that classification metrics can be calculated.

------------------------------------------------------------------------

## 📊 Dataset Features

The project uses the following machine telemetry features:

  Feature                  Description
  ------------------------ -------------------------------
  `Vibration_Hz`           Machine vibration measurement
  `Temperature_C`          Machine temperature
  `Pressure_PSI`           Machine pressure
  `Acoustic_Noise_dB`      Acoustic noise level
  `Rotation_RPM`           Machine rotation speed
  `Power_Consumption_kW`   Power consumption

The dataset is expected at:

``` text
/content/machine_sensor_telemetry.csv
```

------------------------------------------------------------------------

## 🔄 Project Workflow

``` text
Machine Sensor Dataset
        ↓
Load Dataset
        ↓
Data Inspection
        ↓
Missing / Infinite Value Handling
        ↓
Exploratory Data Analysis
        ↓
Create Statistical Reference Labels
        ↓
Select Sensor Features
        ↓
StandardScaler
        ↓
Baseline Isolation Forest
        ↓
Anomaly Prediction
        ↓
Anomaly Score Analysis
        ↓
Evaluation
        ↓
Parameter Tuning
        ↓
Select Best Parameters
        ↓
Final Isolation Forest Model
        ↓
Final Anomaly Detection
        ↓
Anomaly Summary
        ↓
Export Detected Anomalies
```

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   **Python**
-   **NumPy**
-   **Pandas**
-   **Matplotlib**
-   **Seaborn**
-   **Scikit-learn**
-   **Jupyter Notebook / Google Colab**

### Machine Learning

-   Isolation Forest
-   StandardScaler
-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

------------------------------------------------------------------------

## 🧹 Data Preprocessing

### 1. Missing Value Check

The project first checks for missing values using:

``` python
df.isnull().sum()
```

### 2. Infinite Value Detection

Numerical columns are checked for:

``` python
np.inf
-inf
```

### 3. Replace Infinite Values

Infinite values are replaced with `NaN`:

``` python
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)
```

### 4. Remove Missing Values

``` python
df.dropna(inplace=True)
```

This produces a cleaner dataset before anomaly detection.

------------------------------------------------------------------------

## 📈 Exploratory Data Analysis

The project performs EDA using:

### Correlation Heatmap

Used to understand relationships between numerical sensor variables.

### Feature Distributions

Histograms with KDE are generated for:

-   Vibration
-   Temperature
-   Pressure
-   Acoustic Noise
-   Rotation
-   Power Consumption

These visualizations help understand the normal distribution and
potential extreme observations in the telemetry data.

------------------------------------------------------------------------

## 📐 Statistical Reference Label

Because anomaly detection is normally performed without manually labeled
anomalies, this project creates a **reference label** using the 3-sigma
rule.

For every sensor feature:

``` text
Lower Bound = Mean - 3 × Standard Deviation

Upper Bound = Mean + 3 × Standard Deviation
```

If a record falls outside this range for at least one selected sensor:

``` text
Reference Label = 1 → Anomaly
```

Otherwise:

``` text
Reference Label = 0 → Normal
```

### Important

This reference label is a **statistical benchmark**, not a verified
machine-failure label.

Therefore, the evaluation metrics indicate how closely Isolation Forest
agrees with this statistical rule. They should not be interpreted as
confirmed real-world failure-detection accuracy unless actual labeled
failure data is available.

------------------------------------------------------------------------

## ⚙️ Feature Scaling

The six sensor features are standardized using:

``` python
StandardScaler()
```

This transforms the features into a common scale.

Conceptually:

``` text
Scaled Value = (Value - Mean) / Standard Deviation
```

This prevents features with larger numerical ranges from dominating the
analysis.

------------------------------------------------------------------------

## 🌲 Baseline Isolation Forest

The baseline model uses:

``` python
IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)
```

### Important Parameters

  Parameter         Purpose
  ----------------- -------------------------------------------
  `n_estimators`    Number of isolation trees
  `max_samples`     Number of samples used to build each tree
  `contamination`   Expected proportion of anomalies
  `max_features`    Number/proportion of features used
  `bootstrap`       Whether bootstrap sampling is used
  `n_jobs`          Number of CPU cores used
  `random_state`    Makes results reproducible

------------------------------------------------------------------------

## 📏 Anomaly Score

The project also calculates:

``` python
model.decision_function(X_Scaled)
```

This produces an anomaly score for each record.

In general:

-   More negative → more anomalous
-   Higher/positive → more normal

The score is useful for understanding how strongly a record is
considered unusual.

------------------------------------------------------------------------

## 📊 Model Evaluation

The project compares Isolation Forest predictions against the
statistical reference labels using:

### Accuracy

Measures the proportion of correctly classified records.

``` text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Among records predicted as anomalies, precision measures how many match
the reference anomalies.

``` text
Precision =
True Positives / (True Positives + False Positives)
```

### Recall

Among reference anomalies, recall measures how many were detected.

``` text
Recall =
True Positives / (True Positives + False Negatives)
```

### F1 Score

F1 combines precision and recall.

``` text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

For this project, F1-score is used to select the parameter configuration
that has the strongest agreement with the reference labels.

------------------------------------------------------------------------

## 🔧 Parameter Tuning

Several Isolation Forest configurations are tested by changing
parameters such as:

-   `n_estimators`
-   `max_samples`
-   `contamination`
-   `max_features`

The results are stored in a Pandas DataFrame.

Example structure:

``` text
n_estimators
max_samples
contamination
max_features
accuracy
precision
recall
f1
```

The configuration with the highest F1-score against the reference labels
is selected for the final model.

> Since the reference labels are generated from the same telemetry data
> using a statistical rule, this tuning process measures agreement with
> that rule rather than optimizing against independently verified
> failure labels.

------------------------------------------------------------------------

## 🚨 Final Anomaly Detection

The final Isolation Forest model generates:

``` text
Final_Anomaly
Final_Anomaly_Binary
Final_Anomaly_Score
Final_Status
```

### Final Status

``` text
1  → Normal
-1 → Anomaly
```

The project then extracts records classified as anomalies.

------------------------------------------------------------------------

## 📋 Anomaly Summary

The notebook calculates:

-   Total records
-   Normal records
-   Anomaly records
-   Anomaly percentage
-   Normal percentage

This provides a high-level overview of the detected machine behavior.

------------------------------------------------------------------------

## 💾 Output

Detected anomalies are exported as a CSV file.

The notebook currently saves the output using:

``` text
/content/detected_network_anomalies.csv
```

### Suggested filename correction

Because this is a **machine sensor** project rather than a network
traffic project, a clearer output filename would be:

``` text
/content/detected_machine_anomalies.csv
```

If you change the filename in the notebook, use the corrected name
consistently in the README and project folder.

------------------------------------------------------------------------

## 📁 Suggested Project Structure

``` text
Machine-Sensor-Anomaly-Detection/
│
├── Machine_Sensor_Anomaly_Detection_using_Isolation_Forest.ipynb
├── machine_sensor_telemetry.csv
├── detected_machine_anomalies.csv
├── README.md
└── requirements.txt
```

For GitHub, avoid committing very large datasets if they exceed
repository limits. You can instead provide a dataset source or a small
sample dataset.

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone the Repository

``` bash
git clone <your-repository-url>
cd Machine-Sensor-Anomaly-Detection
```

### 2. Install Dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Open the Notebook

``` bash
jupyter notebook
```

Then open:

``` text
Machine_Sensor_Anomaly_Detection_using_Isolation_Forest.ipynb
```

### 4. Add the Dataset

Place:

``` text
machine_sensor_telemetry.csv
```

in the expected location or update the `pd.read_csv()` path in the
notebook.

### 5. Run All Cells

Execute the notebook from top to bottom.

------------------------------------------------------------------------

## 🧪 Example Interpretation

Suppose a machine normally operates around:

``` text
Temperature → normal range
Vibration   → normal range
Pressure    → normal range
Rotation    → normal range
Power       → normal range
```

But one record contains an unusual combination such as:

``` text
Very high vibration
High temperature
Unusual pressure
Abnormal power consumption
```

Isolation Forest may isolate this record quickly.

The model can then produce:

``` text
Final_Status = Anomaly
```

This does **not automatically mean the machine has failed**. It means
the telemetry pattern is unusual according to the trained anomaly
detector.

------------------------------------------------------------------------

## 💡 Real-World Applications

The same approach can be adapted for:

-   Predictive maintenance
-   Industrial equipment monitoring
-   Manufacturing systems
-   Motor monitoring
-   HVAC equipment monitoring
-   Industrial IoT
-   Sensor fault detection
-   Equipment behavior monitoring

For production use, anomaly predictions should ideally be validated
against maintenance logs, failure events, technician inspections, or
other reliable ground-truth information.

------------------------------------------------------------------------

## ⚠️ Limitations

### 1. No confirmed failure labels

The project does not use verified machine-failure labels. The reference
labels are generated using the 3-sigma statistical rule.

### 2. Reference-label dependency

Evaluation metrics measure agreement with the statistical reference
rule.

### 3. Contamination assumption

Isolation Forest's `contamination` parameter influences how many
observations are treated as anomalies.

### 4. Static dataset

Real industrial systems often contain:

-   Time-series dependencies
-   Sensor drift
-   Seasonal behavior
-   Machine-specific operating ranges
-   Maintenance periods
-   Different machine operating modes

A production system may therefore require additional temporal and
domain-specific analysis.

------------------------------------------------------------------------

## 🚀 Future Improvements

Possible improvements include:

-   Use real machine failure/maintenance labels.
-   Add time-series features.
-   Detect anomalies separately for each machine.
-   Add rolling mean and rolling standard deviation.
-   Compare Isolation Forest with One-Class SVM.
-   Compare with Local Outlier Factor.
-   Experiment with Autoencoders.
-   Build a Streamlit monitoring dashboard.
-   Add real-time sensor anomaly detection.
-   Add anomaly severity levels based on anomaly scores.
-   Store predictions in a database.
-   Add alert notifications for critical anomalies.

------------------------------------------------------------------------

## 🧠 Key Concepts Learned

This project demonstrates practical understanding of:

-   Unsupervised anomaly detection
-   Isolation Forest
-   Feature scaling
-   Statistical outlier detection
-   3-sigma rule
-   Data cleaning
-   EDA
-   Anomaly scores
-   Contamination
-   Hyperparameter tuning
-   Precision and Recall
-   F1-score
-   Confusion Matrix
-   Model evaluation
-   CSV export
-   End-to-end ML workflow

------------------------------------------------------------------------

## 👨‍💻 Author

**Muhammad Aasik**

B.Sc. Artificial Intelligence & Machine Learning Student

**Focus:** Backend Development → AI/ML

------------------------------------------------------------------------

## ⭐ Project Goal

This project was built as part of a practical journey into **Machine
Learning and Anomaly Detection**, with the goal of understanding how
unsupervised algorithms can be applied to real-world machine telemetry
data.
