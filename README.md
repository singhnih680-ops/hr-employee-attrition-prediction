HR Attrition ML Pipeline
An educational classification project using automatically generated synthetic employee records. The goal is to demonstrate a complete, understandable machine-learning workflow for HR attrition prediction.
Business objective
Identify possible future resignations to illustrate supportive retention review. This project does not establish causes of attrition and is not suitable for operational employee decisions.
Dataset
1,000 fictional employee records, plus 20 deliberately duplicated rows before cleaning. Inputs include experience, salary, satisfaction, overtime, absences, department and work mode. Missing values are deliberately introduced. Target: 0 = stayed, 1 = resigned within the following year. No real employee data is used.
ML pipeline step by step
1. Loading: Generate fictional records, save a CSV and load it with Pandas.
2. Inspection: Examine shape, data types, missing values and repeated rows.
3. Cleaning: Remove exact duplicates and validate basic target and feature ranges.
4. Feature selection: Exclude identifiers and keep only information available at prediction time.
5. Splitting: Use an 80/20 stratified training/test split before fitting preprocessing.
6. EDA: Summarise and visualise training data; examine imbalance and descriptive patterns.
7. Imputation: Learn numerical medians and categorical modes on training data.
8. Scaling: Fit StandardScaler on training numbers and reuse it for other inputs.
9. SMOTE: Apply SMOTENC only to training data because inputs contain numbers and categories.
10. Encoding: Learn one-hot categories from original training data and transform each model input.
11. Training: Fit Logistic Regression with and without oversampling.
12. Evaluation: Compare accuracy, precision, recall, F1 and confusion matrices on the same untouched test set.
13. Interpretation: Describe missed resignations, false flags, limitations and human review.
14. Prediction: Reuse every fitted preprocessing component for a new fictional employee.
15. Saving: Export results, cleaned data and fitted preprocessing/model components.
Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, imbalanced-learn, Joblib and Google Colab.
Results
Insert the actual baseline and SMOTENC results from the notebook. Report tradeoffs honestly; do not assume that oversampling improved every metric.
Running the project
Open the notebook in Google Colab, use a CPU runtime and run all cells from top to bottom. The setup cell installs the specified package versions. The dataset is created automatically.
Files
- HR_Attrition_Monday_Complete_Pipeline.ipynb — guided notebook
- hr_attrition_raw.csv — generated teaching data
- hr_attrition_cleaned.csv — deduplicated data; fitted imputation remains training-only
- model_comparison.csv — observed evaluation scores
- hr_attrition_teaching_model.joblib — fitted preprocessing components and classifier
Limitations and responsible use
Synthetic data cannot demonstrate real-world effectiveness. Oversampling creates examples rather than evidence; scores can need calibration. Group associations and coefficients do not show causation. Evaluation here is a teaching comparison, not production model selection. Real deployment requires representative data, validation, privacy controls, fairness review, human oversight and monitoring. Load saved models only from trusted sources.
Learning outcomes
Data loading, cleaning, EDA, leakage-aware preprocessing, imbalance handling, classification, evaluation and business interpretation.
