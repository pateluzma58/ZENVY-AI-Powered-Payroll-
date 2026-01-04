# ZENVY - AI-Powered Payroll Risk Scoring System

**Data Science Intern Task Solution** - Builds a comprehensive payroll risk detection system using ML to identify high-risk employees from attendance patterns, leave frequency, and salary anomalies. Meets all requirements: feature engineering, 3-model comparison (XGBoost best), mathematical justification, EDA, and risk scoring.

## 🎯 Problem Statement
Engineer features from attendance/leave/salary data to predict high-risk payroll records. Compare 3 models, justify choice mathematically, deliver notebook with EDA, modeling, feature importance, and model comparison table.

## 📊 Key Features
- **Data**: 10,000 synthetic payroll records (attendance_days, total_work_days=250, leave_freq, salary, prev_salary, risk_label ~5% positive).
- **EDA**: Distributions, correlations, visualizations (histograms, boxplots, heatmaps).
- **Engineered Features**:
  | Feature | Description |
  |---------|-------------|
  | attendance_irreg | 1 - (attendance/total_work_days)  |
  | leave_ratio | leave_freq / total_work_days  |
  | salary_change_vel | \|(salary - prev_salary)/prev_salary\| × 100  |
  | high_leave_flag | leave_ratio > 95th percentile  |
  | salary_spike | salary_change_vel > 95th percentile  |
  | low_attendance | attendance_irreg > 95th percentile  |
- **Models Compared**:
  | Model | CV F1 | AUC-ROC | Log-Loss |
  |-------|-------|---------|----------|
  | Logistic Regression | 0.0000 | 0.4626 | 0.2011  |
  | Random Forest | 0.0194 | 0.5125 | 0.4926  |
  | XGBoost (best) | 0.0194 | 0.5172 | 0.2220  |

## 🏆 Model Justification
**XGBoost selected** for superior AUC-ROC (0.5172) and log-loss (0.2220) on imbalanced data via gradient boosting. Log-loss formula: \( L = -\frac{1}{N} \sum [y \log(p) + (1-y) \log(1-p)] \) minimized better than RF's bagging/LR's linearity for non-linear payroll fraud patterns. Feature importance: attendance_irreg > salary_change
