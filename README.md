# Vacation Recommender System and Travel Agent Engine

A machine learning-based travel recommendation system built in **R** for BU CAS MA685. This project predicts destination ratings and generates **Top-5 travel recommendations** by combining multiple models in a **stacked ensemble** framework.

## Overview

Choosing a travel destination depends on many factors, including budget, trip duration, climate preference, travel style, and destination attributes. This project builds a recommendation engine that learns from:

- **User-level features** such as demographics, budget, and travel preferences
- **Destination-level features** such as climate zone, city type, and average temperature
- **Historical trip ratings** from past user travel records

The system supports two recommendation settings:

1. **No user information available**  
   Recommend destinations using aggregate historical ratings.

2. **User information available**  
   Generate personalized Top-5 destination recommendations using predicted ratings from a stacked ensemble model.

## Models Used

The recommendation engine combines predictions from 7 base models:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Means Clustering (KMC)
- LightGBM
- XGBoost

These base learners are combined using a **stacking meta-learner** implemented with **multiple linear regression**, which learns how to weight each model’s prediction.

## Project Structure

```bash
ma685_travel_agent/
├── src/                         # Source datasets and supporting scripts
├── final_project.R              # Main project code
├── TravelAgent_report.Rmd       # R Markdown report
├── TravelAgent_report.pdf       # Final report
├── MA685 Fianl Project.pdf      # Project-related PDF
├── LICENSE
└── README.md
