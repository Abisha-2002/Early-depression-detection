# Sri Lankan Student Dropout Prediction System

This repository contains the implementation for **IT41043 Intelligent Systems - Milestone 2**. The project aims to predict student dropout rates in Sri Lanka using machine learning techniques based on academic performance, household income, and mental health indices.

##  Repository Structure
- `notebooks/`: Contains Jupyter Notebooks for Baseline and Advanced Models.
- `docs/`: Contains generated PDF evaluation reports.
- `datasets/`: Contains the student dropout dataset.

##  Models Implemented
1. **Baseline Model:** Logistic Regression with SMOTE balancing and train-test split.
2. **Proposed Advanced Model:** Random Forest Classifier evaluated using 5-Fold Stratified Cross-Validation.

##  Evaluation Metrics
- Accuracy, Macro-averaged F1-Score, and ROC-AUC Score.

##  System Architecture Diagram
```mermaid
graph TD
    A[Raw Student Data Input] --> B[Preprocessing & ColumnTransformer]
    B --> C[SMOTE Class Balancing]
    C --> D[Train / Stratified K-Fold Split]
    D --> E[Random Forest Classifier Inference]
    E --> F[Output Interpretation & Metrics Evaluation]

##  Installation & Setup
To install the required Python libraries for this project, run:
```bash
pip install -r requirements.txt

##  Group Details
Student 1 [Chackrawarthi Prabodha Imashi Fernando ] - [ITBIN-2313-0033] : Data preprocessing, implementing SMOTE class balancing, and developing the Baseline Logistic Regression model.

Student 2 [Wesly Jeyananthan Abisha ] - [TBIN-2313-0003] : Implementing the Proposed Random Forest model, setting up the Stratified 5-Fold Cross-Validation pipeline, and generating evaluation reports. 

Module: IT41043 Intelligent Systems
