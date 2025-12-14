#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr
import warnings
from pathlib import Path
from datetime import datetime

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    SURVIVAL_AVAILABLE = True
except ImportError:
    print("⚠️  lifelines not installed. Install with: pip install lifelines")
    SURVIVAL_AVAILABLE = False

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

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
        
    
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
    
    def create_survival_outcomes(self):
        """Create survival outcomes for recurrence-free survival and overall survival."""
        print("\nCreating survival outcomes...")
        
        # Handle mixed date column formats more robustly
        date_columns = {
            'pathological_diagnosis_date': 'f_diag',
            'recurrence_date': 'fecha_de_recidi', 
            'death_date': 'f_muerte',
            'last_followup_date': 'Ultima_fecha'
        }
        
        # Convert and parse date columns with different formats
        for english_name, spanish_name in date_columns.items():
            if spanish_name in self.df_english.columns:
                print(f"  Processing {english_name} ({spanish_name})...")
                
                if spanish_name in ['f_diag', 'Ultima_fecha']:
                
                    self.df_english[english_name] = pd.to_datetime(self.df_english[spanish_name], errors='coerce')
                else:
                   
                    self.df_english[english_name] = pd.to_datetime(self.df_english[spanish_name], 
                                                                  format='%d/%m/%Y', errors='coerce')
        
        # Create recurrence-free survival outcome
        self.create_recurrence_free_survival()
        
        # Create overall survival outcome  
        self.create_overall_survival()
        
        return True
    
    def create_recurrence_free_survival(self):
        """Create recurrence-free survival time and event variables."""
        print("  Creating recurrence-free survival...")
        
      
        self.df_english['rfs_time'] = np.nan
        self.df_english['rfs_event'] = 0
        
        for idx, row in self.df_english.iterrows():
            diagnosis_date = row['pathological_diagnosis_date']
            recurrence_date = row.get('recurrence_date', pd.NaT)
            last_followup = row.get('last_followup_date', pd.NaT)
            recurrence_status = row.get('recurrence', 0)
            
            if pd.isna(diagnosis_date):
                continue
                
            
            if recurrence_status > 0 and not pd.isna(recurrence_date):
                time_to_event = (recurrence_date - diagnosis_date).days
                if time_to_event > 0:  # Valid time
                    self.df_english.at[idx, 'rfs_time'] = time_to_event
                    self.df_english.at[idx, 'rfs_event'] = 1
            
        
            elif not pd.isna(last_followup):
                time_to_censoring = (last_followup - diagnosis_date).days
                if time_to_censoring > 0:  # Valid time
                    self.df_english.at[idx, 'rfs_time'] = time_to_censoring
                    self.df_english.at[idx, 'rfs_event'] = 0
        
       
        valid_rfs = self.df_english['rfs_time'].notna() & (self.df_english['rfs_time'] > 0)
        
        print(f"    ✓ RFS created for {valid_rfs.sum()}/{len(self.df_english)} patients")
        print(f"    Events: {self.df_english[valid_rfs]['rfs_event'].sum()}")
        print(f"    Median follow-up: {self.df_english[valid_rfs]['rfs_time'].median():.0f} days")
    
    def create_overall_survival(self):
        """Create overall survival time and event variables."""
        print("  Creating overall survival...")
        
        
        self.df_english['os_time'] = np.nan
        self.df_english['os_event'] = 0
        
        for idx, row in self.df_english.iterrows():
            diagnosis_date = row['pathological_diagnosis_date']
            death_date = row.get('death_date', pd.NaT)
            last_followup = row.get('last_followup_date', pd.NaT)
            
           
            died = False
            if not pd.isna(death_date):
                died = True
            elif 'current_patient_status' in row and row['current_patient_status'] in [2, 3]:  # Common death codes
                died = True
                
            if pd.isna(diagnosis_date):
                continue
            
            if died and not pd.isna(death_date):
                time_to_event = (death_date - diagnosis_date).days
                if time_to_event > 0:  
                    self.df_english.at[idx, 'os_time'] = time_to_event
                    self.df_english.at[idx, 'os_event'] = 1
            
         
            elif not pd.isna(last_followup):
                time_to_censoring = (last_followup - diagnosis_date).days
                if time_to_censoring > 0:  
                    self.df_english.at[idx, 'os_time'] = time_to_censoring
                    self.df_english.at[idx, 'os_event'] = 0
        
        valid_os = self.df_english['os_time'].notna() & (self.df_english['os_time'] > 0)
        
        print(f"    ✓ OS created for {valid_os.sum()}/{len(self.df_english)} patients")
        print(f"    Events: {self.df_english[valid_os]['os_event'].sum()}")
        print(f"    Median follow-up: {self.df_english[valid_os]['os_time'].median():.0f} days")
    
    def find_top_features_for_survival(self, outcome_type, numeric_features, categorical_features, k=20):
        """Find top k features most associated with survival outcomes."""
        print(f"\nFinding top {k} features for {outcome_type}...")
        
        if outcome_type == 'recurrence_free_survival':
            time_col = 'rfs_time'
            event_col = 'rfs_event'
        elif outcome_type == 'overall_survival':
            time_col = 'os_time'
            event_col = 'os_event'
        else:
            print(f"✗ Unknown outcome type: {outcome_type}")
            return []
        
        if time_col not in self.df_english.columns or event_col not in self.df_english.columns:
            print(f"✗ Survival variables {time_col}/{event_col} not found")
            return []
        
       
        survival_mask = (self.df_english[time_col].notna() & 
                        self.df_english[event_col].notna() &
                        (self.df_english[time_col] > 0))
        survival_data = self.df_english[survival_mask]
        
        if len(survival_data) < 20:
            print(f"✗ Insufficient survival data: {len(survival_data)} patients")
            return []
        
        print(f"  Using {len(survival_data)} patients with complete survival data")
        
        feature_associations = []
        
        # Calculate associations for numeric features
        for feature in numeric_features:
            if feature in [time_col, event_col]:
                continue
                
            feature_data = survival_data[feature].dropna()
            common_idx = survival_data.index.intersection(feature_data.index)
            
            if len(common_idx) > 10:
                try:
                    # Correlation with survival time (for prognostic effect)
                    time_corr, time_p = spearmanr(survival_data.loc[common_idx, feature], 
                                                 survival_data.loc[common_idx, time_col])
                    
                    # Association with event occurrence
                    event_corr, event_p = spearmanr(survival_data.loc[common_idx, feature], 
                                                   survival_data.loc[common_idx, event_col])
                    
                    # Combined association score (average absolute correlations)
                    combined_score = (abs(time_corr) + abs(event_corr)) / 2
                    
                    if not np.isnan(combined_score):
                        feature_associations.append({
                            'feature': feature,
                            'association_score': combined_score,
                            'time_corr': time_corr,
                            'event_corr': event_corr,
                            'time_p': time_p,
                            'event_p': event_p,
                            'type': 'numeric',
                            'sample_size': len(common_idx)
                        })
                except:
                    continue
        
        # Calculate associations for categorical features
        for feature in categorical_features:
            feature_data = survival_data[feature].dropna()
            common_idx = survival_data.index.intersection(feature_data.index)
            
            if len(common_idx) > 10:
                try:
                    # Encode categorical variable
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature_data[common_idx])
                    
                   
                    time_corr, time_p = spearmanr(feature_encoded, 
                                                 survival_data.loc[common_idx, time_col])
                    event_corr, event_p = spearmanr(feature_encoded, 
                                                   survival_data.loc[common_idx, event_col])
                    
                    combined_score = (abs(time_corr) + abs(event_corr)) / 2
                    
                    if not np.isnan(combined_score):
                        feature_associations.append({
                            'feature': feature,
                            'association_score': combined_score,
                            'time_corr': time_corr,
                            'event_corr': event_corr,
                            'time_p': time_p,
                            'event_p': event_p,
                            'type': 'categorical',
                            'sample_size': len(common_idx)
                        })
                except:
                    continue
        
        # Sort by association strength and get top k
        feature_associations.sort(key=lambda x: x['association_score'], reverse=True)
        top_features = feature_associations[:k]
    
        self.target_features[outcome_type] = top_features
        
        print(f"✓ Top {len(top_features)} features for {outcome_type}:")
        for i, feat in enumerate(top_features[:10], 1):
            print(f"  {i:2d}. {feat['feature']:<30} | Score: {feat['association_score']:.3f} | "
                  f"Time r: {feat['time_corr']:5.3f} | Event r: {feat['event_corr']:5.3f}")
        if len(top_features) > 10:
            print(f"     ... and {len(top_features) - 10} more")
        
        return top_features
    
    def prepare_survival_data(self, outcome_type):
        """Prepare data for survival modeling with selected features."""
        if outcome_type not in self.target_features:
            print(f"✗ No features selected for {outcome_type}")
            return None, None, None, None
        
        # Get selected features
        selected_features = [f['feature'] for f in self.target_features[outcome_type]]
    
        if outcome_type == 'recurrence_free_survival':
            time_col = 'rfs_time'
            event_col = 'rfs_event'
        elif outcome_type == 'overall_survival':
            time_col = 'os_time'
            event_col = 'os_event'
        else:
            print(f"✗ Unknown outcome type: {outcome_type}")
            return None, None, None, None
        
        # Prepare dataset with survival outcome and features
        survival_data = self.df_english[[time_col, event_col] + selected_features].copy()
        
        # Remove rows where survival data is missing
        survival_data = survival_data.dropna(subset=[time_col, event_col])
        survival_data = survival_data[survival_data[time_col] > 0]  # Positive survival times only
        
        # Handle missing values in features (simple imputation)
        numeric_cols = survival_data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove survival columns from features
        numeric_cols = [c for c in numeric_cols if c not in [time_col, event_col]]
        
        # Impute numeric features with median
        for col in numeric_cols:
            if survival_data[col].isnull().any():
                survival_data[col].fillna(survival_data[col].median(), inplace=True)
        
  
        categorical_cols = survival_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if survival_data[col].isnull().any():
                survival_data[col].fillna(survival_data[col].mode()[0], inplace=True)
            
            le = LabelEncoder()
            survival_data[col] = le.fit_transform(survival_data[col])
        
        
        X = survival_data.drop([time_col, event_col], axis=1)
        survival_outcome = survival_data[[time_col, event_col]]
        
        print(f"✓ Prepared survival data for {outcome_type}: {X.shape[0]} samples, {X.shape[1]} features")
     
        survival_data.to_csv(self.output_dir / f'{outcome_type}_cleaned_survival_data.csv', index=False)
        
        return X, survival_outcome, selected_features, survival_data
    
    def create_survival_model(self, outcome_type):
        """Create and train survival model."""
        if not SURVIVAL_AVAILABLE:
            print("✗ Survival analysis libraries not available. Install lifelines: pip install lifelines")
            return None
        
        X, survival_outcome, features, full_data = self.prepare_survival_data(outcome_type)
        
        if X is None:
            return None
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col = 'rfs_time'
            event_col = 'rfs_event'
        else:  # overall_survival
            time_col = 'os_time'
            event_col = 'os_event'
        
        # Split data for validation
        train_idx, test_idx = train_test_split(full_data.index, test_size=0.3, random_state=42)
        train_data = full_data.loc[train_idx].copy()
        test_data = full_data.loc[test_idx].copy()
        
        # Scale features
        scaler = StandardScaler()
        feature_cols = [c for c in full_data.columns if c not in [time_col, event_col]]
        
        train_data_scaled = train_data.copy()
        test_data_scaled = test_data.copy()
        
        train_data_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
        test_data_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
        
        # Store scaler
        self.scalers[outcome_type] = scaler
        
        try:
            # Fit Cox Proportional Hazards model
            cph = CoxPHFitter(penalizer=0.1)  # Add penalty term
            cph.fit(train_data_scaled, duration_col=time_col, event_col=event_col)
            
            # Predictions and evaluation
            c_index_train = cph.concordance_index_
            c_index_test = concordance_index(test_data[time_col], 
                                           -cph.predict_partial_hazard(test_data_scaled), 
                                           test_data[event_col])
            
            print(f"\n{outcome_type.upper().replace('_', ' ')} - Cox Regression Results:")
            print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")
            print(f"Events in training: {train_data[event_col].sum()}/{len(train_data)}")
            print(f"Events in testing: {test_data[event_col].sum()}/{len(test_data)}")
            print(f"C-index (training): {c_index_train:.3f}")
            print(f"C-index (testing): {c_index_test:.3f}")
            
            # Show top significant features
            hazard_ratios = cph.hazard_ratios_.sort_values(ascending=False)
            p_values = cph.summary['p']
            
            print(f"\nTop 5 Most Significant Features (p < 0.05):")
            sig_features = p_values[p_values < 0.05].sort_values()
            for feature in sig_features.head().index:
                hr = hazard_ratios[feature]
                p_val = p_values[feature]
                direction = "↑ Risk" if hr > 1 else "↓ Risk"
                print(f"  {feature:<25} | HR: {hr:5.2f} | p: {p_val:6.3f} | {direction}")
            
            # Store model
            self.models[outcome_type] = {
                'model': cph,
                'features': features,
                'train_data': train_data,
                'test_data': test_data,
                'c_index_train': c_index_train,
                'c_index_test': c_index_test
            }
            
            return cph
            
        except Exception as e:
            print(f"✗ Error fitting survival model: {e}")
            return None
    
    def create_survival_feature_importance_plot(self, outcome_type):
        """Create feature importance visualization for survival outcomes."""
        if outcome_type not in self.target_features:
            return
        
        # Get top features data
        features_data = self.target_features[outcome_type][:15] 
        
        features = [f['feature'] for f in features_data]
        association_scores = [f['association_score'] for f in features_data]
  
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, association_scores, color='darkred', alpha=0.7)
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Survival Association Score')
        plt.title(f'Top 15 Features for {outcome_type.replace("_", " ").title()}')
        plt.grid(axis='x', alpha=0.3)
        
 
        for i, v in enumerate(association_scores):
            plt.text(v + max(association_scores) * 0.01, i, f'{v:.3f}', va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{outcome_type}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {outcome_type}_feature_importance.png")
    
    def save_survival_results(self):
        """Save survival analysis preprocessing results to files."""
        print(f"\nSaving survival analysis results to {self.output_dir}...")
        
        
        with open(self.output_dir / 'survival_analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write("ENDOMETRIAL CANCER - SURVIVAL ANALYSIS FEATURE SELECTION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Challenge Outcomes: Recurrence-free Survival & Overall Survival\n\n")
            
            if hasattr(self, 'target_features') and self.target_features:
                for outcome, features_data in self.target_features.items():
                    f.write(f"SURVIVAL OUTCOME: {outcome.upper().replace('_', ' ')}\n")
                    f.write("-" * 40 + "\n")
                    f.write("Rank | Feature | Association | Time Corr | Event Corr | Time p | Event p | Type | N\n")
                    f.write("-" * 90 + "\n")
                    
                    for i, feat in enumerate(features_data, 1):
                        f.write(f"{i:4d} | {feat['feature']:<20} | {feat['association_score']:11.3f} | "
                               f"{feat['time_corr']:9.3f} | {feat['event_corr']:10.3f} | "
                               f"{feat['time_p']:6.3f} | {feat['event_p']:7.3f} | "
                               f"{feat['type']:<11} | {feat['sample_size']:3d}\n")
                    f.write("\n")
            else:
                f.write("No features selected - check survival outcome creation.\n\n")
            
           
            if hasattr(self, 'models') and self.models:
                f.write("BASELINE COX REGRESSION PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                for outcome, model_info in self.models.items():
                    if model_info:
                        f.write(f"{outcome.replace('_', ' ').title()}:\n")
                        f.write(f"  C-index (train): {model_info['c_index_train']:.3f}\n")
                        f.write(f"  C-index (test):  {model_info['c_index_test']:.3f}\n")
                        
                       
                        if outcome == 'recurrence_free_survival':
                            event_col = 'rfs_event'
                        elif outcome == 'overall_survival':
                            event_col = 'os_event'
                        else:
                            event_col = f"{outcome}_event"
                        
                        if event_col in model_info['train_data'].columns:
                            f.write(f"  Training events: {model_info['train_data'][event_col].sum()}\n")
                            f.write(f"  Testing events:  {model_info['test_data'][event_col].sum()}\n")
                        f.write("\n")
        
        if hasattr(self, 'target_features') and self.target_features:
            for outcome, features_data in self.target_features.items():
                features_df = pd.DataFrame(features_data)
                features_df.to_csv(self.output_dir / f'{outcome}_top_features.csv', index=False)
        
        print("✓ Survival analysis results saved successfully")
    
    def run_survival_preprocessing_pipeline(self):
        """Run the complete survival analysis preprocessing pipeline."""
        print("ENDOMETRIAL CANCER - SURVIVAL ANALYSIS PREPROCESSING")
        print("=" * 60)
        
   
        if not self.load_and_prepare_data():
            return False
        
        if not self.create_survival_outcomes():
            return False
        

        numeric_features, categorical_features = self.identify_usable_features()
      
        survival_outcomes = ['recurrence_free_survival', 'overall_survival']
        
        print(f"\nSurvival outcomes for analysis: {survival_outcomes}")
        
        # Process each survival outcome
        for outcome in survival_outcomes:
            print(f"\n{'='*70}")
            print(f"PROCESSING SURVIVAL OUTCOME: {outcome.upper().replace('_', ' ')}")
            print(f"{'='*70}")
            
            # Find top features
            self.find_top_features_for_survival(outcome, numeric_features, categorical_features)
            
            # Create survival model
            self.create_survival_model(outcome)
            
            # Create visualization
            self.create_survival_feature_importance_plot(outcome)
        
   
        self.save_survival_results()
        
        print(f"\n{'='*70}")
        print("SURVIVAL ANALYSIS PREPROCESSING COMPLETED!")
        print(f"{'='*70}")
        print(f"\nGenerated files in {self.output_dir}:")
        print("  - survival_analysis_results.txt")
        print("  - [outcome]_top_features.csv (for each survival outcome)")
        print("  - [outcome]_feature_importance.png (for each outcome)")
        print("  - [outcome]_cleaned_survival_data.csv (ready for modeling)")
        print("\nCox regression baseline models trained and evaluated.")
   
        return True

if __name__ == "__main__":
    print("ENDOMETRIAL CANCER SURVIVAL ANALYSIS PREPROCESSING")
    print("=" * 60)
    print("Challenge Requirements:")
    print("  1. Recurrence-free survival analysis")
    print("  2. Overall survival analysis") 
    print("  3. Time-to-event modeling with Cox regression")
    print("=" * 60)
    
  
    excel_file_path = "IQ_Cancer_Endometrio_merged_NMSP.xlsx"
    
   
    output_directory = "survival_preprocessing_output"
    
    preprocessor = EndometrialCancerPreprocessor(excel_file_path, output_directory)
    success = preprocessor.run_survival_preprocessing_pipeline()
    
    if success:
        print(f"\n✓ Survival preprocessing complete! Check {output_directory}/ for results.")
        print("\n📊 SUMMARY FOR YOUR SURVIVAL ANALYSIS TEAM:")
        print("  ✅ Recurrence-free survival outcomes created")
        print("  ✅ Overall survival outcomes created") 
        print("  ✅ Top 20 features identified for each survival endpoint")
        print("  ✅ Cox regression baseline models trained")
        print("  ✅ Cleaned survival datasets ready for advanced modeling")
    else:
        print("\n✗ Survival preprocessing failed. Check error messages above.")
        print("\n💡 TROUBLESHOOTING:")
        if not SURVIVAL_AVAILABLE:
            print("  • Install survival libraries: pip install lifelines scikit-survival")
        print("  • Check date column formats in your dataset")
        print("  • Verify patient follow-up data completeness")