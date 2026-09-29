# Pneumonia Risk Factor Prediction and Alarm System

A prototype machine-learning system for pneumonia risk-factor prediction and binary alarm detection using patient vital signs.

**Project Type:** Academic / Research Prototype  
**ML Model:** XGBoost Multiclass Classifier  
**Development Environment:** Google Colab  
**Input:** Patient vital signs in CSV format  
**Output:** Predicted risk factor, risk probabilities, and alarm status

---

## 1. Project Overview

This repository contains the complete implementation of the pneumonia risk-factor prediction prototype, including:

- Python training and prediction code
- Dataset used for training and testing
- MATLAB-generated patient-vital signals
- Architecture diagrams

The model was developed and tested using Google Colab.

The current prototype is designed for the **pneumonia** medical context. It takes selected patient vital signs as input and predicts one of four risk-factor classes, followed by a binary alarm decision.

---

## 2. System Objective

The objective is to demonstrate a pipeline in which patient-vital data can be processed by a machine-learning model and converted into a risk-factor and alarm output.

```text
Patient Vital Signals
        ↓
     CSV Data
        ↓
Pneumonia Risk Prediction
        ↓
 ┌───────────────┐
 ↓               ↓
Risk Factor     Alarm
