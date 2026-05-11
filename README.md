## Overview
This project investigates the use of synthetic data as an alternative to real-world datasets in sensitive domains such as healthcare and finance. The goal is to understand how well synthetic data can preserve useful patterns while maintaining fairness across different groups.

We generate synthetic datasets using both statistical and deep learning approaches, including Gaussian Copula and CTGAN. These synthetic datasets are then evaluated using machine learning models such as Logistic Regression and Random Forest.

To assess the quality of synthetic data, we compare model performance and fairness metrics between real and synthetic data using the Train on Synthetic, Test on Real (TSTR) approach. Our analysis focuses on understanding how synthetic data impacts accuracy, fairness, and overall model reliability.

## Datasets
We use two real-world datasets:
- UCI Diabetes Dataset – healthcare data used to predict patient readmission
- German Credit Dataset – financial data used to predict credit risk

## Models

### Generative Models (Synthetic Data)
- synthpop (R package)
- SDV (Python library)
  - Gaussian Copula
  - CTGAN

### Downstream Models (Evaluation)
- Logistic Regression
- Random Forest

## Evaluation Method
We use the TSTR (Train on Synthetic, Test on Real) approach:
- Train model on synthetic data
- Test model on real data

Metrics used:
- Accuracy
- F1 Score
- AUC
- Fairness metrics (FPR, FNR, Precision by group)

## How to Run

1. Open the notebooks in Jupyter Notebook or Google Colab:
   - midtermreportregressionmodel.ipynb
   - RandomForestmodel.ipynb

2. Make sure the dataset files are in the same folder:
   - diabetic_data.csv
   - german.data
   - diabetic_data_synthetic_gen1.csv
   - german_credit_synthetic_gen1.csv

3. Run all cells from top to bottom:
   - Data preparation
   - Model training (Logistic Regression / Random Forest)
   - Evaluation (TSTR + fairness metrics)

4. Check the outputs:
   - Performance metrics (Accuracy, F1, AUC)
   - Fairness tables

## Results
- Synthetic data preserves patterns but not perfectly
- Model performance decreases on synthetic data
- Fairness differences exist across groups

## Conclusion
Synthetic data is a powerful tool that can effectively capture important patterns from real-world data while helping protect sensitive information. In our project, we observed that models trained on synthetic data can still achieve reasonable performance and provide useful insights.

Although there are some differences compared to real data, especially across certain groups, synthetic data remains valuable for experimentation, prototyping, and data sharing. When used carefully and evaluated properly, it can support the development of machine learning models in privacy-sensitive domains.

Overall, synthetic data should be viewed as a complementary approach rather than a complete replacement for real data, with fairness and performance checks playing an important role in its application.

## Team
SynthPopStars
