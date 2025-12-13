#!/usr/bin/env python3
"""
Simple Preprocessing Pipeline for Endometrial Cancer Dataset
Identifies 20 most important features for each target variable and creates baseline models

Target Variables:
- recurrence (binary): Use logistic regression
- final_risk_group (ordinal 1-5): Use ordinal regression
- disease_free (categorical): Use logistic regression

Requirements:
pip install pandas numpy scikit-learn matplotlib seaborn

Author: Preprocessing Pipeline
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from scipy.stats import pearsonr, spearmanr
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class EndometrialCancerPreprocessor:
    def __init__(self, excel_path, output_dir="preprocessing_output"):
        """Initialize preprocessor with Excel file path."""
        self.excel_path = excel_path
        self.df = None
        self.df_english = None
        self.target_features = {}
        self.models = {}
        self.scalers = {}
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
        
        # Variable mapping (same as before)
        self.variable_mapping = {
            'codigo_participante': 'patient_code',
            'edad': 'age_at_diagnosis',
            'imc': 'bmi',
            'asa': 'asa_score',
            'f_diag': 'pathological_diagnosis_date',
            'tipo_histologico': 'histological_subtype_preop',
            'Grado': 'tumor_grade_preop',
            'valor_de_ca125': 'ca125_value',
            'ecotv_infiltsub': 'myometrial_invasion_preop_subjective',
            'ecotv_infiltobj': 'myometrial_invasion_preop_objective',
            'metasta_distan': 'distant_metastasis',
            'grupo_riesgo': 'preoperative_risk_group',
            'estadiaje_pre_i': 'preoperative_staging',
            'tto_NA': 'neoadjuvant_treatment',
            'ciclos_tto_NAdj': 'neoadjuvant_cycles',
            'tto_1_quirugico': 'primary_surgery_type',
            'tiempo_qx': 'surgery_time',
            'perdida_hem_cc': 'blood_loss_cc',
            'dias_de_ingreso': 'hospital_stay_days',
            'histo_defin': 'final_histology',
            'grado_histologi': 'final_grade',
            'tamano_tumoral': 'tumor_size',
            'infiltracion_mi': 'myometrial_invasion',
            'afectacion_linf': 'lymphovascular_invasion',
            'AP_centinela_pelvico': 'sentinel_node_pathology',
            'AP_ganPelv': 'pelvic_nodes_pathology',
            'AP_glanPaor': 'paraaortic_nodes_pathology',
            'n_total_GC': 'total_sentinel_nodes',
            'n_GC_Afect': 'affected_sentinel_nodes',
            'recep_est_porcent': 'estrogen_receptor_percentage',
            'rece_de_Ppor': 'progesterone_receptor_percentage',
            'p53_molecular': 'p53_molecular',
            'p53_ihq': 'p53_immunohistochemistry',
            'mut_pole': 'pole_mutation',
            'msh2': 'msh2',
            'msh6': 'msh6',
            'pms2': 'pms2',
            'mlh1': 'mlh1',
            'FIGO2023': 'figo_stage_2023',
            'grupo_de_riesgo_definitivo': 'final_risk_group',
            'Tributaria_a_Radioterapia': 'indication_radiotherapy',
            'rdt': 'radiotherapy_given',
            'bqt': 'brachytherapy',
            'qt': 'chemotherapy',
            'recidiva': 'recurrence',
            'libre_enferm': 'disease_free'
        }
        
    def load_and_prepare_data(self):
        """Load Excel file and apply variable mapping."""
        print("Loading and preparing data...")
        
        try:
            self.df = pd.read_excel(self.excel_path)
            print(f"✓ Data loaded: {self.df.shape[0]} patients, {self.df.shape[1]} variables")
            
            # Apply variable mapping
            self.df_english = self.df.copy()
            available_mappings = {k: v for k, v in self.variable_mapping.items() if k in self.df.columns}
            self.df_english = self.df_english.rename(columns=available_mappings)
            print(f"✓ Renamed {len(available_mappings)} variables to English")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def identify_usable_features(self, max_missing_pct=50):
        """Identify features with acceptable missing data percentage."""
        print(f"\nIdentifying features with <{max_missing_pct}% missing data...")
        
        # Calculate missing percentages
        missing_pct = (self.df_english.isnull().sum() / len(self.df_english) * 100)
        
        # Get usable features (excluding patient ID and target variables)
        usable_features = missing_pct[missing_pct < max_missing_pct].index.tolist()
        
        # Remove patient identifier and target variables
        exclude_vars = ['patient_code', 'recurrence', 'final_risk_group', 'disease_free']
        usable_features = [f for f in usable_features if f not in exclude_vars]
        
        # Separate numeric and categorical features
        numeric_features = self.df_english[usable_features].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df_english[usable_features].select_dtypes(include=['object']).columns.tolist()
        
        print(f"✓ Found {len(usable_features)} usable features:")
        print(f"  - Numeric: {len(numeric_features)}")
        print(f"  - Categorical: {len(categorical_features)}")
        
        return numeric_features, categorical_features
    
    def find_top_features_for_target(self, target_var, numeric_features, categorical_features, k=20):
        """Find top k features most correlated with target variable."""
        print(f"\nFinding top {k} features for {target_var}...")
        
        if target_var not in self.df_english.columns:
            print(f"✗ Target variable {target_var} not found")
            return []
        
        # Get target data (remove missing values)
        target_data = self.df_english[target_var].dropna()
        feature_correlations = []
        
        # Calculate correlations for numeric features
        for feature in numeric_features:
            feature_data = self.df_english[feature]
            
            # Get common indices (both target and feature available)
            common_idx = target_data.index.intersection(feature_data.dropna().index)
            
            if len(common_idx) > 10:  # Need at least 10 samples
                target_subset = target_data[common_idx]
                feature_subset = feature_data[common_idx]
                
                try:
                    # Use Spearman correlation (handles ordinal data better)
                    corr, p_value = spearmanr(feature_subset, target_subset)
                    if not np.isnan(corr):
                        feature_correlations.append({
                            'feature': feature,
                            'correlation': abs(corr),
                            'p_value': p_value,
                            'type': 'numeric',
                            'sample_size': len(common_idx)
                        })
                except:
                    continue
        
        # For categorical features, use chi-square or ANOVA
        for feature in categorical_features:
            feature_data = self.df_english[feature]
            common_idx = target_data.index.intersection(feature_data.dropna().index)
            
            if len(common_idx) > 10:
                try:
                    # Convert categorical to numeric codes for correlation
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature_data[common_idx])
                    target_subset = target_data[common_idx]
                    
                    corr, p_value = spearmanr(feature_encoded, target_subset)
                    if not np.isnan(corr):
                        feature_correlations.append({
                            'feature': feature,
                            'correlation': abs(corr),
                            'p_value': p_value,
                            'type': 'categorical',
                            'sample_size': len(common_idx)
                        })
                except:
                    continue
        
        # Sort by correlation strength and get top k
        feature_correlations.sort(key=lambda x: x['correlation'], reverse=True)
        top_features = feature_correlations[:k]
        
        # Store results
        self.target_features[target_var] = top_features
        
        print(f"✓ Top {len(top_features)} features for {target_var}:")
        for i, feat in enumerate(top_features[:10], 1):
            print(f"  {i:2d}. {feat['feature']:<30} | Corr: {feat['correlation']:.3f} | p: {feat['p_value']:.3f}")
        if len(top_features) > 10:
            print(f"     ... and {len(top_features) - 10} more")
        
        return top_features
    
    def prepare_modeling_data(self, target_var):
        """Prepare data for modeling with selected features."""
        if target_var not in self.target_features:
            print(f"✗ No features selected for {target_var}")
            return None, None, None, None
        
        # Get selected features
        selected_features = [f['feature'] for f in self.target_features[target_var]]
        
        # Prepare dataset with target and features
        model_data = self.df_english[[target_var] + selected_features].copy()
        
        # Remove rows where target is missing
        model_data = model_data.dropna(subset=[target_var])
        
        # Handle missing values in features (simple imputation)
        numeric_cols = model_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_var)  # Exclude target
        
        # Impute numeric features with median
        for col in numeric_cols:
            if model_data[col].isnull().any():
                model_data[col].fillna(model_data[col].median(), inplace=True)
        
        # Encode categorical features
        categorical_cols = model_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if model_data[col].isnull().any():
                model_data[col].fillna(model_data[col].mode()[0], inplace=True)
            
            le = LabelEncoder()
            model_data[col] = le.fit_transform(model_data[col])
        
        # Split features and target
        X = model_data.drop(target_var, axis=1)
        y = model_data[target_var]
        
        print(f"✓ Prepared modeling data for {target_var}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, selected_features, model_data
    
    def create_baseline_model(self, target_var):
        """Create and train baseline model."""
        X, y, features, _ = self.prepare_modeling_data(target_var)
        
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[target_var] = scaler
        
        # Choose model based on target variable type
        if target_var == 'recurrence':
            # Binary classification
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluation
            print(f"\n{target_var.upper()} - Binary Classification Results:")
            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        elif target_var == 'final_risk_group':
            # Treat as regression (ordinal)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluation
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"\n{target_var.upper()} - Regression Results:")
            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            print(f"R² Score: {r2:.3f}")
            print(f"MSE: {mse:.3f}")
            print(f"RMSE: {np.sqrt(mse):.3f}")
            
        elif target_var == 'disease_free':
            # Multi-class classification
            model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluation
            print(f"\n{target_var.upper()} - Multi-class Classification Results:")
            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        # Store model
        self.models[target_var] = {
            'model': model,
            'features': features,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return model
    
    def create_feature_importance_plot(self, target_var):
        """Create feature importance visualization."""
        if target_var not in self.target_features:
            return
        
        # Get top features data
        features_data = self.target_features[target_var][:15]  # Top 15 for visualization
        
        features = [f['feature'] for f in features_data]
        correlations = [f['correlation'] for f in features_data]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, correlations, color='steelblue', alpha=0.7)
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Absolute Correlation with Target')
        plt.title(f'Top 15 Features Correlated with {target_var.replace("_", " ").title()}')
        plt.grid(axis='x', alpha=0.3)
        
        # Add correlation values on bars
        for i, v in enumerate(correlations):
            plt.text(v + max(correlations) * 0.01, i, f'{v:.3f}', va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{target_var}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_preprocessing_results(self):
        """Save preprocessing results to files."""
        print(f"\nSaving results to {self.output_dir}...")
        
        # Save feature selection results
        with open(self.output_dir / 'feature_selection_results.txt', 'w', encoding='utf-8') as f:
            f.write("ENDOMETRIAL CANCER - FEATURE SELECTION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for target_var, features_data in self.target_features.items():
                f.write(f"TARGET: {target_var.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write("Rank | Feature | Correlation | P-value | Type | Sample Size\n")
                f.write("-" * 70 + "\n")
                
                for i, feat in enumerate(features_data, 1):
                    f.write(f"{i:4d} | {feat['feature']:<25} | {feat['correlation']:11.3f} | "
                           f"{feat['p_value']:7.3f} | {feat['type']:<11} | {feat['sample_size']:4d}\n")
                f.write("\n")
        
        # Save selected features as CSV for easy import
        for target_var, features_data in self.target_features.items():
            features_df = pd.DataFrame(features_data)
            features_df.to_csv(self.output_dir / f'{target_var}_top_features.csv', index=False)
        
        print("✓ Results saved successfully")
    
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline."""
        print("ENDOMETRIAL CANCER - PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Load data
        if not self.load_and_prepare_data():
            return False
        
        # Identify usable features
        numeric_features, categorical_features = self.identify_usable_features()
        
        # Define target variables
        target_variables = ['recurrence', 'final_risk_group', 'disease_free']
        available_targets = [t for t in target_variables if t in self.df_english.columns]
        
        print(f"\nTarget variables found: {available_targets}")
        
        # Process each target variable
        for target_var in available_targets:
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_var.upper()}")
            print(f"{'='*60}")
            
            # Find top features
            self.find_top_features_for_target(target_var, numeric_features, categorical_features)
            
            # Create baseline model
            self.create_baseline_model(target_var)
            
            # Create visualization
            self.create_feature_importance_plot(target_var)
        
        # Save results
        self.save_preprocessing_results()
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"\nGenerated files in {self.output_dir}:")
        print("  - feature_selection_results.txt")
        print("  - [target]_top_features.csv (for each target)")
        print("  - [target]_feature_importance.png (for each target)")
        print("\nBaseline model performance printed above.")
        print("Ready for advanced modeling by your team!")
        
        return True

# Main execution
if __name__ == "__main__":
    # CHANGE THIS PATH TO MATCH YOUR EXCEL FILE LOCATION
    excel_file_path = "IQ_Cancer_Endometrio_merged_NMSP.xlsx"
    
    # Create output directory
    output_directory = "preprocessing_output"
    
    # Initialize and run preprocessor
    preprocessor = EndometrialCancerPreprocessor(excel_file_path, output_directory)
    success = preprocessor.run_preprocessing_pipeline()
    
    if success:
        print(f"\n✓ Preprocessing complete! Check {output_directory}/ for results.")
        print("\n📊 SUMMARY FOR YOUR TEAM:")
        print("  - Feature selection completed for all target variables")
        print("  - Baseline models trained and evaluated")
        print("  - Top 20 features identified for each target")
        print("  - Data ready for advanced modeling techniques")
    else:
        print("\n✗ Preprocessing failed. Check error messages above.")