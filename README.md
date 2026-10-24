# Loan Approval Prediction: Feature Engineering and Data Preprocessing

**AnalystLab Africa Data Science Internship | Week 2 | Dataset: Loan Prediction Problem (Kaggle)**
**Author:** Cheruiyot Nickson Kirui

Preparing a home-loan application dataset for machine learning. The notebook takes the raw Kaggle file from inspection to a documented, leakage-free, machine-learning-ready dataset, and the two reports explain every decision.

## The problem
Dream Housing Finance wants to speed up and standardise home-loan eligibility checks. The target is `Loan_Status` (Y = approved, N = rejected), a binary classification problem with 68.7% approved and 31.3% rejected.

## Headline results
- **614 applications, 13 columns**; 22% of rows had at least one missing value and there were no duplicates.
- **11 new features** were engineered (total income, monthly instalment, repayment-to-income ratio, household size and log versions of skewed money columns).
- **24 candidate features reduced to 8** using redundancy checks, statistical tests, mutual information, random forest importance with a random-noise benchmark, plus documented domain and fairness rules.
- **Credit history is the only strong predictor.** Income and loan-amount features behave like noise on their own but are kept as the core affordability measures.
- Preparation lifts cross-validated ROC-AUC from **0.64 (raw)** to about **0.73 to 0.74**.
- The pipeline is **leakage-free** (everything is fitted on the training rows only) and runs unchanged on the 367 unseen Kaggle test applications.

## Repository structure
```
Week2_Loan_Preprocessing/
├── Week2_Feature_Engineering_Preprocessing.ipynb   # Main notebook (executed, with outputs)
├── data/
│   ├── raw/                  # Original Kaggle files
│   └── processed/
│       ├── loan_cleaned.csv                 # Cleaned dataset (readable, before encoding/scaling)
│       ├── loan_ml_ready.csv                # Machine-learning-ready dataset (all 614 rows + split flag)
│       ├── loan_ml_ready_train.csv          # Training rows (491)
│       ├── loan_ml_ready_holdout.csv        # Hold-out rows (123)
│       ├── loan_kaggle_test_ml_ready.csv    # Unlabelled Kaggle test file, same pipeline
│       ├── preprocessing_parameters.json    # Every fitted value (fill values, caps, scaler, encodings)
│       ├── feature_selection_summary.csv    # Evidence and decision for each of the 24 candidates
│       └── ml_ready_data_dictionary.csv     # Description of every final column
├── reports/
│   ├── Business_Understanding_Report.pdf
│   └── Data_Preprocessing_Report.pdf
├── images/                   # All 14 charts
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Week2_Feature_Engineering_Preprocessing.ipynb    # then Kernel > Restart & Run All
```
The notebook reads from `data/raw/` and writes to `data/processed/` and `images/`.

## Pipeline overview
| Step | What was done |
|---|---|
| Inspection | First 10 rows, structure, types, missing values, duplicates, summary statistics, target balance |
| Split | 80/20 stratified train/hold-out **before** anything is fitted (prevents data leakage) |
| Cleaning | `Dependents` "3+" to 3; credit history to Meets/Fails; clear column names; mode / median / "Unknown" imputation fitted on training rows |
| Feature engineering | total income, EMI, EMI-to-income, household size, income per person, log transforms |
| Outliers | IQR, z-score and boxplots; log transform plus IQR capping (no rows deleted) |
| Encoding | binary 0/1, one-hot (property area, credit history), ordinal (dependents), target N/Y to 0/1 |
| Scaling | `StandardScaler` on continuous features (compared with MinMax and Robust) |
| Selection | correlation heatmap, mutual information, random forest importance vs noise probes, statistical tests, domain and fairness rules |
| Validation | 8 readiness checks, cross-validated comparison, one hold-out check, pipeline re-run on unseen data |

## Key assumptions and decisions
- **Gender and marital status are excluded** from the final dataset: using them in credit decisions can discriminate, and neither is statistically supported here.
- **`Unknown` credit history is kept** as its own category so applicants with no record are not silently treated as having a good history.
- `LoanAmount` is in thousands and `Loan_Amount_Term` is in months (per the dataset description). Incomes are assumed to be monthly; the currency is not stated.
- The target is a **past approval decision**, not a repayment outcome.
- The small sample (614 rows) means results carry wide error bars; use cross-validation rather than a single split.

## Data source
Loan Prediction Problem Dataset on Kaggle (originally an Analytics Vidhya practice problem). Please check the licence on the dataset's Kaggle page before redistributing the raw files.

## Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, scipy (versions in `requirements.txt`).

## Author
Cheruiyot Nickson Kirui | [LinkedIn](#) | #AnalystLabAfrica
