# NSMP Endometrial Cancer Survival Calculator

A clinical decision support tool for predicting Overall Survival (OS) and Recurrence-Free Survival (RFS) in endometrial cancer patients using Cox Proportional Hazards models trained on the NSMP (No Specific Molecular Profile) endometrial cancer cohort.

## 🎯 Overview

This project provides:
- **Survival Prediction Models**: Two Cox PH models for OS and RFS prediction
- **Interactive Web Application**: Dash-based calculator for real-time survival probability predictions
- **Comprehensive Analysis Pipeline**: Data preprocessing, feature selection, model training, and evaluation tools

## 📊 Model Performance

- **Overall Survival (OS)**: C-index = 0.77
- **Recurrence-Free Survival (RFS)**: C-index = 0.85
- **Training Cohort**: NSMP endometrial cancer patients (n=161)
- **Regularization**: L2 penalizer = 1.0

## 🏗️ Project Structure

```
hackathon/
├── app/
│   ├── dash_app.py              # Main Dash web application
│   └── assets/
│       └── custom.css            # Custom styling
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── modeling.py              # Cox PH model implementation
│   ├── survival_analysis.py     # Survival analysis functions
│   └── validation.py            # Model validation utilities
├── preprocessing/
│   ├── 02_preproccesing.py      # Main preprocessing script
│   ├── 02b_oversampliong.py     # Oversampling pipeline
│   ├── 02c_overs_check.py        # Oversampling validation
│   ├── 03_encoding.py            # Feature encoding
│   └── fix_sparse_features.py    # Sparse feature handling
├── models/
│   ├── cox_model_os.pkl         # Trained OS model
│   └── cox_model_recurrent.pkl  # Trained RFS model
├── corrected_preprocessing_output/  # Preprocessed datasets
├── balanced_survival_output/       # SMOTE-balanced datasets
├── model_evaluation_plots/         # Model evaluation visualizations
├── cv_results/                     # Cross-validation results
├── run_survival_analysis.py        # Model training/validation runner
├── evaluate_survival_models.py     # Model evaluation script
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd hackathon
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application

1. **Start the Dash application:**
   ```bash
   python app/dash_app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8051`

3. **Use the calculator:**
   - Fill in patient clinical features
   - Select prediction model(s) and time horizons
   - Click "Calculate Survival Probability" to see predictions

## 📖 Usage Guide

### Web Application Features

The Dash application provides an intuitive interface for survival prediction:

#### Input Fields

**Common Features (used by both models):**
- **Numeric Variables**: Tumor Size (mm), BMI (kg/m²), Total Sentinel Nodes
- **Tumor Characteristics**: Preoperative/Final Tumor Grade, Final Histology
- **Risk Classification**: Preoperative Risk Group, Final Risk Group, Preoperative Staging
- **Invasion & Pathology**: Myometrial Invasion, Sentinel Node Status, Primary Surgery Type
- **Clinical Factors**: Distant Metastasis, Chemotherapy

**OS-Specific Features:**
- PMS2 Expression (MMR marker)

**RFS-Specific Features:**
- Lymphovascular Invasion
- Radiotherapy Indication
- Surgery Time (minutes)

#### Output

The application provides:
- **Risk Category**: Low, Medium, or High risk classification
- **Risk Score**: Continuous hazard ratio score
- **Survival Curves**: Interactive Kaplan-Meier-style survival probability curves
- **Survival Probabilities**: Predicted survival at selected time points (1-10 years)
- **Feature Contributions**: Bar chart showing which features contribute most to the prediction

### Command-Line Tools

#### Training Models

Train a new Cox PH model:

```bash
python run_survival_analysis.py --mode training \
    --data path/to/data.csv \
    --duration_col os_time \
    --event_col os_event \
    --penalizer 1.0 \
    --save_model models/my_model.pkl
```

#### Validating Models

Validate a trained model:

```bash
python run_survival_analysis.py --mode validation \
    --model_path models/cox_model_os.pkl \
    --data path/to/test_data.csv \
    --prediction_years 1 3 5
```

#### Preprocessing Data

Run the preprocessing pipeline:

```bash
python preprocessing/02_preproccesing.py
```

This will:
- Load and clean the raw Excel data
- Create survival time and event variables
- Perform feature selection
- Generate train/test splits
- Save preprocessed datasets

## 🔬 Model Details

### Overall Survival (OS) Model

**Features (16 total):**
- `final_histology_2.0` (Serous carcinoma indicator)
- `tumor_grade_preop_encoded`
- `distant_metastasis`
- `preoperative_risk_group_encoded`
- `primary_surgery_type_1.0` (Laparoscopic surgery)
- `preoperative_staging_2.0` (Stage II+ indicator)
- `sentinel_node_pathology_4.0` (Macrometastasis)
- `total_sentinel_nodes`
- `final_grade_encoded`
- `chemotherapy`
- `tumor_size`
- `bmi`
- `final_risk_group_encoded`
- `pms2` (PMS2 expression loss)
- `myometrial_invasion_preop_objective_4.0`
- `myometrial_invasion_preop_subjective_2.0`

### Recurrence-Free Survival (RFS) Model

**Features (20 total):**
- `tumor_grade_preop_encoded`
- `preoperative_risk_group_encoded`
- `final_grade_encoded`
- `final_risk_group_encoded`
- `chemotherapy`
- `sentinel_node_pathology_4.0`
- `final_histology_2.0` (Serous)
- `final_histology_9.0` (Carcinosarcoma)
- `lymphovascular_invasion`
- `preoperative_staging_1.0` (Stage I)
- `preoperative_staging_2.0` (Stage II+)
- `myometrial_invasion_encoded`
- `tumor_size`
- `indication_radiotherapy`
- `bmi`
- `surgery_time`
- `distant_metastasis`
- `primary_surgery_type_1.0`
- `myometrial_invasion_preop_subjective_2.0`
- `total_sentinel_nodes`

## 📁 Data Requirements

### Input Data Format

The preprocessing pipeline expects an Excel file with the following key columns:

**Survival Outcomes:**
- Date columns for diagnosis, recurrence, death, and follow-up
- Recurrence and death indicators

**Clinical Features:**
- Demographics: Age, BMI, ASA score
- Tumor characteristics: Histology, grade, size, invasion depth
- Staging: Preoperative and final staging, risk groups
- Molecular markers: p53, POLE, MMR proteins (MSH2, MSH6, PMS2, MLH1)
- Treatment: Surgery type, chemotherapy, radiotherapy
- Pathology: Sentinel node status, lymphovascular invasion

See `key_features.md` for a complete list of features and missing data percentages.

### Output Data Format

Preprocessed datasets include:
- Survival time variables: `os_time`, `rfs_time` (in days)
- Event indicators: `os_event`, `rfs_event` (0=censored, 1=event)
- Encoded and normalized features ready for modeling

## 🛠️ Development

### Code Organization

- **`src/`**: Core modules for data processing, modeling, and validation
- **`app/`**: Web application code
- **Scripts**: Standalone preprocessing and analysis scripts

### Key Modules

- **`src/preprocessing.py`**: Data cleaning, encoding, and feature engineering
- **`src/modeling.py`**: Cox PH model wrapper class
- **`src/survival_analysis.py`**: High-level survival analysis functions
- **`app/dash_app.py`**: Interactive web interface

### Adding New Features

1. Update feature mappings in `app/dash_app.py`:
   - Add to `OS_FEATURES` or `RFS_FEATURES`
   - Add human-readable label to `FEATURE_LABELS`
   - Create input component in `create_common_inputs()` or model-specific functions

2. Update preprocessing in `preprocessing/02_preproccesing.py`:
   - Add feature extraction logic
   - Update encoding/transformation steps

3. Retrain models with new features

## 📊 Evaluation Metrics

Models are evaluated using:
- **Concordance Index (C-index)**: Measures discrimination ability
- **Kaplan-Meier Curves**: Visual assessment of risk stratification
- **Hazard Ratios**: Feature importance and effect sizes
- **Cross-Validation**: Internal validation of model performance

## ⚠️ Clinical Disclaimer

**This tool is for research and educational purposes only. It is not intended for clinical decision-making without professional medical judgment.**

The models are trained on a specific cohort (NSMP endometrial cancer patients) and may not generalize to all patient populations. Always consult with qualified healthcare professionals for clinical decisions.

## 📝 License

This project is intended for research purposes. Please ensure appropriate data use agreements and ethical approvals are in place before using patient data.

## 🤝 Contributing

When contributing to this project:
1. Follow the existing code structure and style
2. Add docstrings to new functions
3. Update this README if adding major features
4. Test changes thoroughly before committing

## 📧 Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated**: December 2025
**Version**: 1.0

