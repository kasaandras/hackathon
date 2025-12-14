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
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    SURVIVAL_AVAILABLE = True
except ImportError:
    print("⚠️  lifelines not installed. Install with: pip install lifelines")
    SURVIVAL_AVAILABLE = False

# VIF calculation
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except ImportError:
    print("⚠️  statsmodels not installed. Install with: pip install statsmodels")
    VIF_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class ImprovedSurvivalPreprocessor:
    def __init__(self, excel_path, output_dir="corrected_preprocessing_output"):
        """Initialize preprocessor with Excel file path."""
        self.excel_path = excel_path
        self.df = None
        self.df_english = None
        self.encoded_data = {}
        self.feature_info = {}
        self.selected_features = {}
        self.models = {}
        
        # Create output directory
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
        
        # Define variable types (YOU MAY NEED TO ADJUST THESE)
        self.continuous_vars = [
            'age_at_diagnosis', 'bmi', 'ca125_value', 'surgery_time',
            'blood_loss_cc', 'hospital_stay_days', 'tumor_size',
            'total_sentinel_nodes', 'affected_sentinel_nodes',
            'estrogen_receptor_percentage', 'progesterone_receptor_percentage'
        ]
        
        self.ordinal_vars = {
            'tumor_grade_preop': {
                'G1': 1, 'G2': 2, 'G3': 3,  # Text values
                1: 1, 2: 2, 3: 3, 1.0: 1, 2.0: 2, 3.0: 3  # Numeric values
            },
            'final_grade': {
                'G1': 1, 'G2': 2, 'G3': 3,
                1: 1, 2: 2, 3: 3, 1.0: 1, 2.0: 2, 3.0: 3
            },
            'asa_score': {
                1: 1, 2: 2, 3: 3, 4: 4,
                1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4
            },
            'preoperative_risk_group': {
                # Text values
                'Low': 1, 'Intermediate': 2, 'High': 3,
                'Bajo': 1, 'Intermedio': 2, 'Alto': 3,
                'low': 1, 'intermediate': 2, 'high': 3,
                # Numeric values (already in data)
                1: 1, 2: 2, 3: 3,
                1.0: 1, 2.0: 2, 3.0: 3
            },
            'final_risk_group': {
                # Your data has 1-5 scale
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5,
                # Text values just in case
                'Low': 1, 'Intermediate': 3, 'High': 5,
                'Bajo': 1, 'Intermedio': 3, 'Alto': 5
            },
            'myometrial_invasion': {
                # Numeric values (0-3 scale in your data)
                0: 0, 1: 1, 2: 2, 3: 3,
                0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3,
                # Text values just in case
                'None': 0, 'No': 0, '<50%': 1, '<50': 1,
                '≥50%': 2, '>=50': 2, 'Yes': 1
            }
        }
        
        self.binary_vars = [
            'distant_metastasis', 'neoadjuvant_treatment', 'lymphovascular_invasion',
            'indication_radiotherapy', 'radiotherapy_given', 'brachytherapy',
            'chemotherapy', 'p53_molecular', 'p53_immunohistochemistry',
            'pole_mutation', 'msh2', 'msh6', 'pms2', 'mlh1'
        ]
        
        self.nominal_vars = [
            'histological_subtype_preop', 'final_histology',
            'primary_surgery_type', 'preoperative_staging', 'figo_stage_2023',
            'myometrial_invasion_preop_subjective', 'myometrial_invasion_preop_objective',
            'sentinel_node_pathology', 'pelvic_nodes_pathology', 'paraaortic_nodes_pathology'
        ]
    
    def load_and_prepare_data(self):
        """Load Excel file and apply variable mapping."""
        print("\n" + "="*70)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*70)
        
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
    
    def create_survival_outcomes(self):
        """Create survival outcomes."""
        print("\n" + "="*70)
        print("STEP 2: CREATING SURVIVAL OUTCOMES")
        print("="*70)
        
        # Date columns
        date_columns = {
            'pathological_diagnosis_date': 'f_diag',
            'recurrence_date': 'fecha_de_recidi', 
            'death_date': 'f_muerte',
            'last_followup_date': 'Ultima_fecha'
        }
        
        # Convert dates
        for english_name, spanish_name in date_columns.items():
            if spanish_name in self.df_english.columns:
                if spanish_name in ['f_diag', 'Ultima_fecha']:
                    self.df_english[english_name] = pd.to_datetime(self.df_english[spanish_name], errors='coerce')
                else:
                    self.df_english[english_name] = pd.to_datetime(self.df_english[spanish_name], 
                                                                  format='%d/%m/%Y', errors='coerce')
        
        # Create RFS
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
                if time_to_event > 0:
                    self.df_english.at[idx, 'rfs_time'] = time_to_event
                    self.df_english.at[idx, 'rfs_event'] = 1
            elif not pd.isna(last_followup):
                time_to_censoring = (last_followup - diagnosis_date).days
                if time_to_censoring > 0:
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
            
            died = not pd.isna(death_date)
            
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
        
        # Report
        valid_rfs = self.df_english['rfs_time'].notna() & (self.df_english['rfs_time'] > 0)
        valid_os = self.df_english['os_time'].notna() & (self.df_english['os_time'] > 0)
        
        print(f"\n✓ RFS created: {valid_rfs.sum()}/{len(self.df_english)} patients")
        print(f"  Events: {self.df_english[valid_rfs]['rfs_event'].sum()}")
        print(f"  Median follow-up: {self.df_english[valid_rfs]['rfs_time'].median():.0f} days")
        
        print(f"\n✓ OS created: {valid_os.sum()}/{len(self.df_english)} patients")
        print(f"  Events: {self.df_english[valid_os]['os_event'].sum()}")
        print(f"  Median follow-up: {self.df_english[valid_os]['os_time'].median():.0f} days")
        
        return True
    
    def encode_variables(self, outcome_type):
        """Properly encode all variables BEFORE feature selection."""
        print("\n" + "="*70)
        print(f"STEP 3: ENCODING VARIABLES FOR {outcome_type.upper()}")
        print("="*70)
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col, event_col = 'rfs_time', 'rfs_event'
        else:
            time_col, event_col = 'os_time', 'os_event'
        
        # Get valid survival data
        survival_mask = (self.df_english[time_col].notna() & 
                        self.df_english[event_col].notna() &
                        (self.df_english[time_col] > 0))
        df_work = self.df_english[survival_mask].copy()
        
        print(f"\n✓ Working with {len(df_work)} patients with complete survival data")
        
        # Exclude columns
        exclude_cols = ['patient_code', 'recurrence', 'disease_free', 'final_risk_group',
                       'pathological_diagnosis_date', 'recurrence_date', 'death_date',
                       'last_followup_date', 'rfs_time', 'rfs_event', 'os_time', 'os_event']
        
        # Start with empty encoded dataframe
        df_encoded = pd.DataFrame()
        encoding_report = []
        
        # 1. CONTINUOUS VARIABLES - No encoding needed
        print("\n[1/4] Processing continuous variables...")
        continuous_present = [v for v in self.continuous_vars if v in df_work.columns]
        for var in continuous_present:
            if df_work[var].notna().sum() > 0:
                df_encoded[var] = df_work[var]
                encoding_report.append({
                    'variable': var,
                    'type': 'continuous',
                    'action': 'kept as-is',
                    'n_features': 1,
                    'missing_pct': df_work[var].isna().sum() / len(df_work) * 100
                })
        print(f"  ✓ Kept {len(continuous_present)} continuous variables")
        
        # 2. BINARY VARIABLES - Ensure 0/1 numeric
        print("\n[2/4] Processing binary variables...")
        binary_present = [v for v in self.binary_vars if v in df_work.columns]
        for var in binary_present:
            if df_work[var].notna().sum() > 0:
                # Convert to 0/1
                unique_vals = df_work[var].dropna().unique()
                if len(unique_vals) <= 2:
                    df_encoded[var] = df_work[var].astype(float)
                    encoding_report.append({
                        'variable': var,
                        'type': 'binary',
                        'action': 'kept as 0/1',
                        'n_features': 1,
                        'missing_pct': df_work[var].isna().sum() / len(df_work) * 100
                    })
        print(f"  ✓ Kept {len(binary_present)} binary variables")
        
        # 3. ORDINAL VARIABLES - Label encoding
        print("\n[3/4] Processing ordinal variables...")
        ordinal_count = 0
        for var, mapping in self.ordinal_vars.items():
            if var in df_work.columns and df_work[var].notna().sum() > 0:
                try:
                    # Show what values exist before mapping
                    unique_vals = df_work[var].dropna().unique()
                    print(f"\n  Processing: {var}")
                    print(f"    Unique values in data: {unique_vals[:10]}")  # Show first 10
                    
                    df_encoded[f'{var}_encoded'] = df_work[var].map(mapping)
                    
                    # Check if mapping worked
                    n_before = df_work[var].notna().sum()
                    n_after = df_encoded[f'{var}_encoded'].notna().sum()
                    
                    if n_after < n_before:
                        unmapped = n_before - n_after
                        print(f"    ⚠️  Warning: {unmapped} values failed to map!")
                        print(f"    Before mapping: {n_before} non-null")
                        print(f"    After mapping: {n_after} non-null")
                        # Show unmapped values
                        unmapped_vals = df_work[df_work[var].notna() & df_encoded[f'{var}_encoded'].isna()][var].unique()
                        print(f"    Unmapped values: {unmapped_vals}")
                    else:
                        print(f"    ✓ Successfully mapped {n_after} values")
                    
                    ordinal_count += 1
                    encoding_report.append({
                        'variable': var,
                        'type': 'ordinal',
                        'action': f'label encoded ({n_after} mapped)',
                        'n_features': 1,
                        'missing_pct': df_work[var].isna().sum() / len(df_work) * 100
                    })
                except Exception as e:
                    print(f"    ✗ Error encoding {var}: {e}")
        print(f"\n  ✓ Encoded {ordinal_count} ordinal variables")
        
        # 4. NOMINAL VARIABLES - One-hot encoding with drop_first=True
        print("\n[4/4] Processing nominal variables...")
        nominal_present = [v for v in self.nominal_vars if v in df_work.columns]
        nominal_to_encode = []
        
        for var in nominal_present:
            if df_work[var].notna().sum() > 0:
                n_unique = df_work[var].nunique()
                missing_pct = df_work[var].isna().sum() / len(df_work) * 100
                
                # Only encode if reasonable number of categories
                if 2 <= n_unique <= 10 and missing_pct < 50:
                    nominal_to_encode.append(var)
                    encoding_report.append({
                        'variable': var,
                        'type': 'nominal',
                        'action': f'one-hot encoded ({n_unique-1} features)',
                        'n_features': n_unique - 1,  # drop_first=True
                        'missing_pct': missing_pct
                    })
        
        if nominal_to_encode:
            # Add nominal variables to dataframe temporarily
            for var in nominal_to_encode:
                df_encoded[var] = df_work[var]
            
            # One-hot encode with drop_first=True (prevents dummy variable trap!)
            df_encoded = pd.get_dummies(df_encoded, columns=nominal_to_encode, 
                                       drop_first=True, dummy_na=False)
            
            # Convert boolean columns to int (0/1 instead of True/False)
            print(f"  ✓ Converting boolean dummy variables to 0/1...")
            bool_cols = df_encoded.select_dtypes(include=['bool']).columns
            if len(bool_cols) > 0:
                for col in bool_cols:
                    df_encoded[col] = df_encoded[col].astype(int)
                print(f"    Converted {len(bool_cols)} boolean columns to integer")
        
        print(f"  ✓ One-hot encoded {len(nominal_to_encode)} nominal variables")
        print(f"  ✓ Created {len(df_encoded.columns)} total features after encoding")
        
        # Handle missing values
        print("\n[IMPUTATION] Handling missing values...")
        for col in df_encoded.columns:
            if df_encoded[col].isna().any():
                if df_encoded[col].dtype in ['float64', 'int64']:
                    df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
                else:
                    df_encoded[col].fillna(df_encoded[col].mode()[0], inplace=True)
        
        # Remove columns that are ALL NaN (failed encoding)
        print("\n[CLEANUP] Checking for completely empty features...")
        all_nan_cols = []
        for col in df_encoded.columns:
            if col not in [time_col, event_col]:
                if df_encoded[col].isna().all():
                    all_nan_cols.append(col)
        
        if all_nan_cols:
            print(f"  ⚠️  Removing {len(all_nan_cols)} features with all NaN values:")
            for col in all_nan_cols:
                print(f"    • {col}")
                df_encoded.drop(col, axis=1, inplace=True)
        else:
            print("  ✓ No all-NaN features found")
        
        # Add survival outcomes
        df_encoded[time_col] = df_work[time_col].values
        df_encoded[event_col] = df_work[event_col].values
        
        # Store
        self.encoded_data[outcome_type] = df_encoded
        self.feature_info[outcome_type] = pd.DataFrame(encoding_report)
        
        # Save encoding report
        report_path = self.output_dir / f'{outcome_type}_encoding_report.csv'
        self.feature_info[outcome_type].to_csv(report_path, index=False)
        print(f"\n✓ Encoding report saved: {report_path.name}")
        
        return df_encoded
    
    def check_multicollinearity(self, df_encoded, outcome_type):
        """Check for multicollinearity using VIF and correlation."""
        print("\n" + "="*70)
        print(f"STEP 4: CHECKING MULTICOLLINEARITY FOR {outcome_type.upper()}")
        print("="*70)
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col, event_col = 'rfs_time', 'rfs_event'
        else:
            time_col, event_col = 'os_time', 'os_event'
        
        # Get feature columns
        feature_cols = [c for c in df_encoded.columns if c not in [time_col, event_col]]
        X = df_encoded[feature_cols]
        
        print(f"\n📊 Analyzing {len(feature_cols)} features...")
        
        # 1. Correlation matrix
        print("\n[1/2] Calculating feature correlation matrix...")
        corr_matrix = X.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature1': corr_matrix.columns[i],
                        'Feature2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.7):")
            for pair in high_corr_pairs[:10]:
                print(f"  • {pair['Feature1']:<35} ↔ {pair['Feature2']:<35} | r = {pair['Correlation']:>6.3f}")
            if len(high_corr_pairs) > 10:
                print(f"  ... and {len(high_corr_pairs)-10} more")
        else:
            print("\n✓ No highly correlated pairs found (|r| > 0.7)")
        
        # Save correlation matrix plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'{outcome_type.replace("_", " ").title()} - Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{outcome_type}_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {outcome_type}_correlation_matrix.png")
        
        # 2. VIF calculation
        if VIF_AVAILABLE and len(X) > len(feature_cols):
            print("\n[2/2] Calculating VIF (Variance Inflation Factor)...")
            try:
                vif_data = []
                for i, col in enumerate(feature_cols):
                    vif = variance_inflation_factor(X.values, i)
                    vif_data.append({'Feature': col, 'VIF': vif})
                
                vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
                
                # Report high VIF features
                high_vif = vif_df[vif_df['VIF'] > 10]
                if len(high_vif) > 0:
                    print(f"\n⚠️  Found {len(high_vif)} features with VIF > 10 (severe multicollinearity):")
                    for _, row in high_vif.head(15).iterrows():
                        severity = "🚨 SEVERE" if row['VIF'] > 100 else "⚠️  HIGH"
                        print(f"  {severity} | {row['Feature']:<40} | VIF = {row['VIF']:>8.2f}")
                    if len(high_vif) > 15:
                        print(f"  ... and {len(high_vif)-15} more")
                else:
                    print("\n✓ All features have VIF < 10 (acceptable multicollinearity)")
                
                # Save VIF report
                vif_df.to_csv(self.output_dir / f'{outcome_type}_VIF_report.csv', index=False)
                print(f"\n✓ VIF report saved: {outcome_type}_VIF_report.csv")
                
                return high_corr_pairs, vif_df
                
            except Exception as e:
                print(f"\n⚠️  VIF calculation failed: {e}")
                return high_corr_pairs, None
        else:
            print("\n⚠️  VIF calculation skipped (statsmodels not available or insufficient data)")
            return high_corr_pairs, None
    
    def remove_multicollinear_features(self, df_encoded, outcome_type, vif_threshold=10, corr_threshold=0.85):
        """Remove features with severe multicollinearity."""
        print("\n" + "="*70)
        print(f"STEP 5: REMOVING MULTICOLLINEAR FEATURES FOR {outcome_type.upper()}")
        print("="*70)
        print(f"Thresholds: VIF > {vif_threshold}, |Correlation| > {corr_threshold}")
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col, event_col = 'rfs_time', 'rfs_event'
        else:
            time_col, event_col = 'os_time', 'os_event'
        
        feature_cols = [c for c in df_encoded.columns if c not in [time_col, event_col]]
        X = df_encoded[feature_cols].copy()
        
        features_to_remove = set()
        
        # Strategy 1: Remove based on VIF
        if VIF_AVAILABLE:
            print("\n[Strategy 1] Removing features based on VIF...")
            iteration = 0
            max_iterations = 20
            
            while iteration < max_iterations:
                iteration += 1
                current_features = [f for f in feature_cols if f not in features_to_remove]
                
                if len(current_features) <= 1:
                    break
                
                X_current = X[current_features]
                
                try:
                    vif_values = []
                    for i, col in enumerate(current_features):
                        vif = variance_inflation_factor(X_current.values, i)
                        vif_values.append((col, vif))
                    
                    max_vif_feature, max_vif = max(vif_values, key=lambda x: x[1])
                    
                    if max_vif > vif_threshold:
                        features_to_remove.add(max_vif_feature)
                        print(f"  Iteration {iteration}: Removed {max_vif_feature} (VIF = {max_vif:.2f})")
                    else:
                        print(f"  Iteration {iteration}: All VIF values acceptable (max = {max_vif:.2f})")
                        break
                except:
                    print(f"  VIF calculation failed at iteration {iteration}")
                    break
            
            if features_to_remove:
                print(f"\n  ✓ Removed {len(features_to_remove)} features based on VIF")
        
        # Strategy 2: Remove highly correlated pairs (keep one with higher survival association)
        print("\n[Strategy 2] Removing one feature from highly correlated pairs...")
        remaining_features = [f for f in feature_cols if f not in features_to_remove]
        
        if len(remaining_features) > 1:
            corr_matrix = X[remaining_features].corr()
            
            # Calculate survival association for tie-breaking
            survival_assoc = {}
            for feat in remaining_features:
                try:
                    corr_time, _ = spearmanr(X[feat], df_encoded[time_col])
                    corr_event, _ = spearmanr(X[feat], df_encoded[event_col])
                    survival_assoc[feat] = abs(corr_time) + abs(corr_event)
                except:
                    survival_assoc[feat] = 0
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    
                    if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                        # Remove the one with weaker survival association
                        if survival_assoc[feat1] >= survival_assoc[feat2]:
                            if feat2 not in features_to_remove:
                                features_to_remove.add(feat2)
                                print(f"  Removed {feat2} (corr with {feat1} = {corr_matrix.iloc[i, j]:.3f})")
                        else:
                            if feat1 not in features_to_remove:
                                features_to_remove.add(feat1)
                                print(f"  Removed {feat1} (corr with {feat2} = {corr_matrix.iloc[i, j]:.3f})")
        
        # Create cleaned dataset
        final_features = [f for f in feature_cols if f not in features_to_remove]
        df_cleaned = df_encoded[final_features + [time_col, event_col]].copy()
        
        print(f"\n📊 Summary:")
        print(f"  Original features: {len(feature_cols)}")
        print(f"  Removed features: {len(features_to_remove)}")
        print(f"  Remaining features: {len(final_features)}")
        
        # Save removed features list
        if features_to_remove:
            removed_df = pd.DataFrame({
                'removed_feature': list(features_to_remove),
                'reason': 'High VIF or correlation'
            })
            removed_df.to_csv(self.output_dir / f'{outcome_type}_removed_features.csv', index=False)
            print(f"\n✓ Removed features list saved: {outcome_type}_removed_features.csv")
        
        return df_cleaned, final_features
    
    def select_top_survival_features(self, df_cleaned, outcome_type, k=20):
        """Select top k features based on survival association."""
        print("\n" + "="*70)
        print(f"STEP 6: SELECTING TOP {k} SURVIVAL-ASSOCIATED FEATURES FOR {outcome_type.upper()}")
        print("="*70)
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col, event_col = 'rfs_time', 'rfs_event'
        else:
            time_col, event_col = 'os_time', 'os_event'
        
        feature_cols = [c for c in df_cleaned.columns if c not in [time_col, event_col]]
        
        print(f"\n📊 Analyzing {len(feature_cols)} features...")
        
        # Calculate survival associations
        feature_scores = []
        for feat in feature_cols:
            try:
                # Correlation with survival time
                time_corr, time_p = spearmanr(df_cleaned[feat], df_cleaned[time_col])
                
                # Correlation with event
                event_corr, event_p = spearmanr(df_cleaned[feat], df_cleaned[event_col])
                
                # Combined score
                combined_score = abs(time_corr) + abs(event_corr)
                
                feature_scores.append({
                    'feature': feat,
                    'survival_score': combined_score,
                    'time_corr': time_corr,
                    'time_p': time_p,
                    'event_corr': event_corr,
                    'event_p': event_p
                })
            except:
                continue
        
        # Sort by survival score
        feature_scores.sort(key=lambda x: x['survival_score'], reverse=True)
        
        # Select top k
        top_features = feature_scores[:min(k, len(feature_scores))]
        top_feature_names = [f['feature'] for f in top_features]
        
        # Report
        print(f"\n✓ Top {len(top_features)} features selected:\n")
        print(f"  {'Rank':<6} {'Feature':<40} {'Score':<10} {'Time r':<10} {'Event r':<10}")
        print("  " + "─"*80)
        for i, feat_info in enumerate(top_features[:15], 1):
            print(f"  {i:<6} {feat_info['feature']:<40} "
                  f"{feat_info['survival_score']:<10.3f} "
                  f"{feat_info['time_corr']:<10.3f} "
                  f"{feat_info['event_corr']:<10.3f}")
        
        if len(top_features) > 15:
            print(f"  ... and {len(top_features)-15} more")
        
        # Save feature selection report
        feature_df = pd.DataFrame(top_features)
        feature_df.to_csv(self.output_dir / f'{outcome_type}_selected_features.csv', index=False)
        print(f"\n✓ Feature selection report saved: {outcome_type}_selected_features.csv")
        
        # Create final dataset
        df_final = df_cleaned[top_feature_names + [time_col, event_col]].copy()
        
        # Store
        self.selected_features[outcome_type] = top_feature_names
        
        return df_final, top_feature_names
    
    def create_baseline_cox_model(self, df_final, outcome_type):
        """Create baseline Cox regression model."""
        if not SURVIVAL_AVAILABLE:
            print("\n⚠️  Skipping Cox model (lifelines not installed)")
            return None
        
        print("\n" + "="*70)
        print(f"STEP 7: BASELINE COX REGRESSION FOR {outcome_type.upper()}")
        print("="*70)
        
        # Determine survival columns
        if outcome_type == 'recurrence_free_survival':
            time_col, event_col = 'rfs_time', 'rfs_event'
        else:
            time_col, event_col = 'os_time', 'os_event'
        
        # Check for NaN values BEFORE anything else
        print("\n[Pre-check] Checking for NaN values...")
        nan_counts = df_final.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        
        if len(cols_with_nans) > 0:
            print(f"\n⚠️  Found NaN values in {len(cols_with_nans)} columns:")
            for col, count in cols_with_nans.items():
                print(f"    {col}: {count} NaNs ({count/len(df_final)*100:.1f}%)")
            
            # Fill NaN values
            print("\n  Imputing NaN values...")
            for col in cols_with_nans.index:
                if col not in [time_col, event_col]:
                    if df_final[col].dtype in ['float64', 'int64']:
                        df_final[col].fillna(df_final[col].median(), inplace=True)
                    else:
                        df_final[col].fillna(0, inplace=True)
            print("  ✓ NaN values imputed")
        else:
            print("  ✓ No NaN values found")
        
        # Split data
        train_idx, test_idx = train_test_split(df_final.index, test_size=0.3, random_state=42)
        train_data = df_final.loc[train_idx].copy()
        test_data = df_final.loc[test_idx].copy()
        
        # Scale features
        feature_cols = [c for c in df_final.columns if c not in [time_col, event_col]]
        scaler = StandardScaler()
        
        train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
        test_data[feature_cols] = scaler.transform(test_data[feature_cols])
        
        # Final NaN check after scaling
        if train_data.isnull().any().any():
            print("\n⚠️  NaN values detected after scaling - replacing with 0")
            train_data.fillna(0, inplace=True)
        if test_data.isnull().any().any():
            test_data.fillna(0, inplace=True)
        
        try:
            # Fit Cox model with regularization to handle any remaining multicollinearity
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_data, duration_col=time_col, event_col=event_col)
            
            # Evaluate
            c_index_train = cph.concordance_index_
            c_index_test = concordance_index(test_data[time_col], 
                                           -cph.predict_partial_hazard(test_data), 
                                           test_data[event_col])
            
            print(f"\n✓ Cox Regression Results:")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Testing samples: {len(test_data)}")
            print(f"  Training events: {train_data[event_col].sum()}/{len(train_data)}")
            print(f"  Testing events: {test_data[event_col].sum()}/{len(test_data)}")
            print(f"  C-index (train): {c_index_train:.3f}")
            print(f"  C-index (test): {c_index_test:.3f}")
            
            # Show top significant features
            print(f"\n  Top 5 Most Significant Features (p < 0.05):")
            sig_features = cph.summary[cph.summary['p'] < 0.05].sort_values('p')
            for feat in sig_features.head().index:
                hr = cph.hazard_ratios_[feat]
                p_val = cph.summary.loc[feat, 'p']
                direction = "↑ Risk" if hr > 1 else "↓ Risk"
                print(f"    {feat:<35} | HR: {hr:>6.2f} | p: {p_val:>8.4f} | {direction}")
            
            # Store model
            self.models[outcome_type] = {
                'model': cph,
                'c_index_train': c_index_train,
                'c_index_test': c_index_test,
                'scaler': scaler
            }
            
            return cph
            
        except Exception as e:
            print(f"\n✗ Error fitting Cox model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_final_datasets(self, df_final, outcome_type):
        """Save the final cleaned dataset."""
        print("\n" + "="*70)
        print(f"STEP 8: SAVING FINAL DATASET FOR {outcome_type.upper()}")
        print("="*70)
        
        # Final data type cleanup - ensure everything is numeric
        print("\n[Data Type Check]")
        print(f"  Checking data types before saving...")
        
        # Convert any remaining boolean columns to int
        bool_cols = df_final.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            print(f"  ⚠️  Found {len(bool_cols)} boolean columns - converting to 0/1")
            for col in bool_cols:
                df_final[col] = df_final[col].astype(int)
        
        # Check for object (string) columns (shouldn't happen, but just in case)
        obj_cols = df_final.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            print(f"  ⚠️  Warning: Found {len(obj_cols)} non-numeric columns:")
            for col in obj_cols:
                print(f"    • {col}: {df_final[col].dtype}")
        else:
            print(f"  ✓ All features are numeric")
        
        # Show final data types summary
        print(f"\n  Final data types:")
        dtype_counts = df_final.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"    {str(dtype):<15} {count:>3} columns")
        
        output_file = self.output_dir / f'{outcome_type}_final_cleaned.csv'
        df_final.to_csv(output_file, index=False)
        
        print(f"\n✓ Final dataset saved: {output_file.name}")
        print(f"  Samples: {len(df_final)}")
        print(f"  Features: {len(df_final.columns) - 2}")  # Excluding time and event
        print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    def run_corrected_pipeline(self, vif_threshold=10, corr_threshold=0.85, k_features=20):
        """Run the complete corrected preprocessing pipeline."""
        print("\n" + "="*70)
        print("CORRECTED SURVIVAL ANALYSIS PREPROCESSING PIPELINE")
        print("="*70)
        print(f"VIF threshold: {vif_threshold}")
        print(f"Correlation threshold: {corr_threshold}")
        print(f"Target features per outcome: {k_features}")
        
        # Step 1: Load data
        if not self.load_and_prepare_data():
            return False
        
        # Step 2: Create survival outcomes
        if not self.create_survival_outcomes():
            return False
        
        # Process each survival outcome
        for outcome_type in ['recurrence_free_survival', 'overall_survival']:
            print("\n" + "🔹"*35)
            print(f"PROCESSING: {outcome_type.upper().replace('_', ' ')}")
            print("🔹"*35)
            
            # Step 3: Encode variables
            df_encoded = self.encode_variables(outcome_type)
            
            # Step 4: Check multicollinearity
            high_corr, vif_df = self.check_multicollinearity(df_encoded, outcome_type)
            
            # Step 5: Remove multicollinear features
            df_cleaned, remaining_features = self.remove_multicollinear_features(
                df_encoded, outcome_type, vif_threshold, corr_threshold
            )
            
            # Step 6: Select top features
            df_final, selected_features = self.select_top_survival_features(
                df_cleaned, outcome_type, k_features
            )
            
            # Step 7: Baseline Cox model
            self.create_baseline_cox_model(df_final, outcome_type)
            
            # Step 8: Save final dataset
            self.save_final_datasets(df_final, outcome_type)
        
   
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
    print("="*70)
    print("CORRECTED SURVIVAL ANALYSIS PREPROCESSING")
    print("With Proper Encoding and Multicollinearity Handling")
    print("="*70)
    
    # Configuration
    excel_file_path = "IQ_Cancer_Endometrio_merged_NMSP.xlsx"
    output_directory = "corrected_preprocessing_output"
    
    # Thresholds
    VIF_THRESHOLD = 10      # Remove features with VIF > 10
    CORR_THRESHOLD = 0.85   # Remove one from pairs with |r| > 0.85
    K_FEATURES = 20         # Select top 20 features per outcome
    
    # Initialize and run
    preprocessor = ImprovedSurvivalPreprocessor(excel_file_path, output_directory)
    success = preprocessor.run_corrected_pipeline(
        vif_threshold=VIF_THRESHOLD,
        corr_threshold=CORR_THRESHOLD,
        k_features=K_FEATURES
    )
    
    if success:
        print(f"\n✅ SUCCESS! Check {output_directory}/ for all outputs.")
    else:
        print("\n✗ Pipeline failed. Check error messages above.")
