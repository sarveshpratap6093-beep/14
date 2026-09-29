# Risk-predication-and-alarm-system-for-patients-

Pneumonia Risk Factor Prediction and Alarm System

A prototype machine-learning system for pneumonia risk-factor
prediction and binary alarm detection using patient vital signs.

Project type: Academic / research prototype
ML model: XGBoost Multiclass Classifier
Development environment: Google Colab
Input: Patient vital signs in CSV format
Output: Predicted risk factor, risk probabilities, and alarm
status

1. Project Overview

This repository contains the complete implementation of the pneumonia
risk-factor prediction prototype, including:

The complete Python training and prediction code

The final dataset used to train and test the model

MATLAB-generated patient-vital signals used for the demonstration

Architecture diagrams for the implemented and planned systems

The model was developed and trained using Google Colab. Running the
project in Google Colab is recommended because the original development
and testing were performed there.

The current prototype is designed for a single medical context:
Pneumonia.

The system takes selected patient vital signs as input and predicts one
of four risk-factor classes. A simple binary alarm rule is then applied
to the predicted risk factor.

2. System Objective

The objective is to demonstrate a pipeline in which patient-vital data
can be generated, converted into CSV data, processed by a
machine-learning model, and converted into a risk-factor and alarm
output.

Patient Vital Signals
        |
        v
     CSV Data
        |
        v
Pneumonia Risk Prediction
        |
        +------------------+
        |                  |
        v                  v
   Risk Factor          Alarm

3. Input Parameters

The model uses six patient parameters:

Parameter                  Dataset Column

Age                        age_years
SpO2                       spo2
Heart Rate                 heart_rate_bpm
Respiratory Rate           respiratory_rate_bpm
Systolic Blood Pressure    systolic_bp_mmhg
Diastolic Blood Pressure   diastolic_bp_mmhg

patient_id is used for patient-level dataset splitting and is not
used as a prediction feature.

4. Risk-Factor Classes

Risk Factor Risk Level

          0 LOW
          1 MODERATE
          2 HIGH
          3 VERY HIGH

The model also produces a probability for each of the four classes.

5. Alarm Logic

The prototype uses:

Risk Factor = 0  -> Alarm = 0
Risk Factor >= 1 -> Alarm = 1

Predicted Risk Factor            Alarm

                    0   0 - Not Raised
                    1       1 - Raised
                    2       1 - Raised
                    3       1 - Raised

This alarm rule is prototype logic and is not a clinically validated
emergency-alert system.

6. Machine-Learning Model

The project uses XGBoost for multiclass classification.

Model configuration

Objective: multi:softprob

Number of classes: 4

Number of estimators: 300

Maximum depth: 6

Learning rate: 0.05

Subsample: 0.8

Column subsampling: 0.8

Evaluation metric: mlogloss

Random state: 42

Tree method: hist

Missing-value handling

Missing numerical values are handled using median imputation. The
imputer is fitted only on training data and then applied to test data
and new patient inputs.

7. Dataset Splitting

Because the dataset contains multiple records associated with patients,
the project uses a patient-level train/test split using
GroupShuffleSplit.

Test size: 20%

Random state: 42

Grouping variable: patient_id

This prevents records belonging to the same patient from being
intentionally divided between training and testing sets.

8. Model Evaluation

The training script reports:

Accuracy

Classification report

Precision

Recall

F1-score

Confusion matrix

Feature importance

The previously trained version achieved approximately 99% test
accuracy on the supplied dataset, with an overall macro F1-score of
approximately 0.98.

These values are specific to this dataset and evaluation procedure. They
do not establish clinical validity or real-world diagnostic performance.

9. Expected Output

After patient values are entered, the program produces:

Predicted risk level

Risk-factor value

Probability of each risk class

Alarm value

Alarm status

Example:

Predicted Risk: LOW
Risk Factor: 0

Risk Probabilities:
LOW: 98.27%
MODERATE: 1.71%
HIGH: 0.02%
VERY HIGH: 0.00%

Alarm: 0
ALARM STATUS: NOT RAISED

The exact probabilities depend on the patient input.

10. How to Use

Step 1 - Open Google Colab

Create a new Google Colab notebook.

Step 2 - Upload the project files

Upload:

code.py
pneumonia_dataset.csv

The dataset filename must match the filename specified in code.py, or
the filename in the code must be changed accordingly.

Step 3 - Install required packages

pip install pandas numpy scikit-learn xgboost joblib

Step 4 - Run the Python code

Run code.py. The program will:

Load the dataset

Select input features

Create the target

Perform patient-level train/test splitting

Handle missing values

Train XGBoost

Evaluate the model

Display feature importance

Save model/preprocessing objects

Accept patient values

Predict the risk factor

Generate alarm status

11. MATLAB Signal Demonstration

MATLAB-generated patient-vital signals were used for the final
demonstration.

MATLAB Generated Patient Vitals
             |
             v
        CSV Conversion
             |
             v
   Pneumonia Risk Predictor
             |
             v
     Risk Factor + Alarm

The MATLAB signals can represent continuously generated patient-vital
measurements. These measurements can be converted into CSV data before
being supplied to the Python model.

12. Implemented Pipeline

The implemented prototype contains only the pneumonia context.

Patient Context
      |
      v
Pneumonia Risk Predictor
      ^
      |
Patient Vitals / CSV
      |
      v
Alarm + Risk Factor

Add the implemented architecture image to the repository as:

implemented_pipeline.png

Then use:

![Implemented Pipeline](implemented_pipeline.png)

13. Planned Full Architecture

The original planned architecture was designed to support multiple
medical contexts.

The intended system would contain:

A context classifier

Multiple context-specific risk models

A mechanism for selecting the appropriate risk model

Risk-factor and alarm output

Patient Context
       |
       v
Context Classifier
       |
       +--------------------+
       |                    |
       v                    v
Context-Specific       Other Context
Risk Model             Risk Models
       |
       v
Alarm + Risk Factor

Add the planned architecture image as:

planned_architecture.png

Then use:

![Planned Architecture](planned_architecture.png)

Because the current prototype contains only the pneumonia context, the
context classifier has not been implemented.

14. Repository Structure

Recommended structure:

14/
|
+-- code.py
+-- README.md
+-- pneumonia_dataset.csv
|
+-- implemented_pipeline.png
+-- planned_architecture.png
|
+-- MATLAB/
    +-- MATLAB-generated signal files

When code.py is executed, it can also generate:

pneumonia_xgboost_model.pkl
pneumonia_imputer.pkl
pneumonia_features.pkl

These files are not required if the model is retrained from the supplied
dataset.

15. Technologies Used

Python

XGBoost

Scikit-learn

Pandas

NumPy

Joblib

MATLAB

Google Colab

GitHub

16. Project Files

code.py

Contains the complete machine-learning pipeline:

Dataset loading

Feature selection

Patient-level splitting

Missing-value handling

XGBoost training

Model evaluation

Feature importance

Model saving

Patient prediction

Risk-factor classification

Alarm generation

pneumonia_dataset.csv

The final dataset used for training and testing.

MATLAB files

Contain the generated patient-vital signals used for the demonstration.

Architecture diagrams

Show the implemented prototype and the planned multi-context
architecture.

17. Limitations

This project is a prototype and has important limitations:

It currently supports only the pneumonia context.

The context-classification stage has not been implemented.

The alarm rule is a simple prototype rule.

The model has not been clinically validated.

Reported performance is specific to the supplied dataset and
evaluation procedure.

High test performance on this dataset should not be interpreted as
evidence of real-world clinical performance.

The model should not be used to make medical diagnoses or treatment
decisions.

18. Future Development

Planned development includes:

Multiple disease/context-specific models

A context classifier

Automatic model selection based on context

Continuous real-time data ingestion

Direct integration with MATLAB-generated or sensor-generated signals

Improved validation on independent datasets

More robust real-world evaluation

A user interface for risk and alarm information

19. Academic / Research Purpose

This repository demonstrates the integration of:

Patient Vital Signals
        |
        v
Data Processing
        |
        v
Machine Learning
        |
        v
Risk Classification
        |
        v
Alarm Logic

It is intended for academic, educational, and prototype research
purposes.

20. Safety Notice

This system is not a medical device and has not been clinically
validated.

The predicted risk factor and alarm output are model outputs based on
the supplied dataset and should not be treated as a medical diagnosis,
treatment recommendation, or substitute for professional medical
assessment.
