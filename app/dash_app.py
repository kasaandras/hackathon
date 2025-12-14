import sys
from pathlib import Path

_APP_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _APP_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib

# APP INITIALIZATION


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "NSMP Endometrial Cancer Survival Calculator"


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Fix dropdown text truncation */
            .Select-menu-outer {
                min-width: 100% !important;
            }
            .Select-option {
                white-space: normal !important;
                padding: 8px 12px !important;
                line-height: 1.4 !important;
                word-wrap: break-word !important;
            }
            .Select-value-label {
                white-space: normal !important;
                word-wrap: break-word !important;
            }
            .Select-control {
                min-height: 38px !important;
            }
            .Select-input {
                min-height: 36px !important;
            }
            /* Ensure dropdown columns are aligned */
            .row > [class*="col-"] {
                display: flex !important;
                flex-direction: column !important;
                align-items: stretch !important;
            }
            .row > [class*="col-"] > .form-label {
                margin-bottom: 8px !important;
            }
            .row > [class*="col-"] > .mb-3 {
                flex: 1 1 auto !important;
            }
            /* Prevent row wrapping for dropdown rows */
            .row {
                flex-wrap: wrap !important;
            }
            /* Ensure three columns fit in one row on medium+ screens */
            @media (min-width: 768px) {
                .row > .col-md-4 {
                    flex: 0 0 33.333333% !important;
                    max-width: 33.333333% !important;
                }
            }
            /* Align radio buttons with dropdowns */
            .form-check-inline {
                display: inline-flex !important;
                align-items: center !important;
                margin-right: 1rem !important;
            }
            /* Ensure consistent label heights */
            .col-md-4 .form-label {
                min-height: 24px !important;
                margin-bottom: 8px !important;
                display: flex !important;
                align-items: center !important;
            }
            /* Align radio button containers with dropdown height */
            .col-md-4 > div:has(input[type="radio"]) {
                min-height: 38px !important;
                display: flex !important;
                align-items: center !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

MODEL_DIR = _PROJECT_DIR / "models"
DATA_DIR = _PROJECT_DIR / "corrected_preprocessing_output"

# Load models
try:
    os_model = joblib.load(MODEL_DIR / "cox_model_os.pkl")
    rfs_model = joblib.load(MODEL_DIR / "cox_model_recurrent.pkl")
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    os_model = None
    rfs_model = None
    MODELS_LOADED = False

# Load training data for statistics (medians, quantiles)
try:
    os_train_data = pd.read_csv(DATA_DIR / "overall_survival_final_cleaned_no_sparse.csv")
    rfs_train_data = pd.read_csv(DATA_DIR / "recurrence_free_survival_final_cleaned_no_sparse.csv")
    
   
    OS_MEDIANS = os_train_data.drop(columns=['os_time', 'os_event']).median().to_dict()
    RFS_MEDIANS = rfs_train_data.drop(columns=['rfs_time', 'rfs_event']).median().to_dict()
    
   
    if MODELS_LOADED:
        os_features = os_model.feature_names
        rfs_features = rfs_model.feature_names
      
        os_risk_scores = os_model.predict_risk(os_train_data[os_features])
        rfs_risk_scores = rfs_model.predict_risk(rfs_train_data[rfs_features])
        
        OS_RISK_QUANTILES = {
            'low': np.percentile(os_risk_scores, 33),
            'high': np.percentile(os_risk_scores, 67)
        }
        RFS_RISK_QUANTILES = {
            'low': np.percentile(rfs_risk_scores, 33),
            'high': np.percentile(rfs_risk_scores, 67)
        }
    else:
        OS_RISK_QUANTILES = {'low': 0.5, 'high': 2.0}
        RFS_RISK_QUANTILES = {'low': 0.5, 'high': 2.0}
        
except Exception as e:
    print(f"Warning: Could not load training data: {e}")
    OS_MEDIANS = {}
    RFS_MEDIANS = {}
    OS_RISK_QUANTILES = {'low': 0.5, 'high': 2.0}
    RFS_RISK_QUANTILES = {'low': 0.5, 'high': 2.0}

OS_FEATURES = [
    'final_histology_2.0', 'tumor_grade_preop_encoded', 'distant_metastasis',
    'preoperative_risk_group_encoded', 'primary_surgery_type_1.0', 'preoperative_staging_2.0',
    'sentinel_node_pathology_4.0', 'total_sentinel_nodes', 'final_grade_encoded',
    'chemotherapy', 'tumor_size', 'bmi', 'final_risk_group_encoded', 'pms2',
    'myometrial_invasion_preop_objective_4.0', 'myometrial_invasion_preop_subjective_2.0'
]

RFS_FEATURES = [
    'tumor_grade_preop_encoded', 'preoperative_risk_group_encoded', 'final_grade_encoded',
    'final_risk_group_encoded', 'chemotherapy', 'sentinel_node_pathology_4.0',
    'final_histology_2.0', 'lymphovascular_invasion', 'preoperative_staging_2.0',
    'myometrial_invasion_encoded', 'tumor_size', 'indication_radiotherapy', 'bmi',
    'surgery_time', 'distant_metastasis', 'primary_surgery_type_1.0',
    'myometrial_invasion_preop_subjective_2.0', 'total_sentinel_nodes',
    'preoperative_staging_1.0', 'final_histology_9.0'
]

FEATURE_LABELS = {
    'tumor_size': 'Tumor Size (cm)',
    'bmi': 'BMI (kg/m²)',
    'total_sentinel_nodes': 'Total Sentinel Nodes',
    'surgery_time': 'Surgery Time (min)',
    'tumor_grade_preop_encoded': 'Preoperative Tumor Grade',
    'final_grade_encoded': 'Final Tumor Grade',
    'preoperative_risk_group_encoded': 'Preoperative Risk Group',
    'final_risk_group_encoded': 'Final Risk Group',
    'primary_surgery_type_1.0': 'Primary Surgery (Yes)',
    'preoperative_staging_1.0': 'Preop Staging I',
    'preoperative_staging_2.0': 'Preop Staging II+',
    'sentinel_node_pathology_4.0': 'Sentinel Node Positive',
    'myometrial_invasion_preop_subjective_2.0': 'Deep Myometrial Invasion (Subjective)',
    'myometrial_invasion_preop_objective_4.0': 'Deep Myometrial Invasion (Objective)',
    'myometrial_invasion_encoded': 'Myometrial Invasion',
    'final_histology_2.0': 'Histology: Serous',
    'final_histology_9.0': 'Histology: Carcinosarcoma',
    'distant_metastasis': 'Distant Metastasis',
    'chemotherapy': 'Chemotherapy',
    'lymphovascular_invasion': 'Lymphovascular Invasion',
    'indication_radiotherapy': 'Radiotherapy Indicated',
    'pms2': 'PMS2 Expression Loss',
}

COLORS = {
    "background": "#f8f9fa",
    "card": "#ffffff",
    "accent": "#0d9488",
    "accent_secondary": "#7c3aed",
    "text": "#1f2937",
    "text_muted": "#6b7280",
    "border": "#e5e7eb",
    "low_risk": "#059669",
    "medium_risk": "#d97706",
    "high_risk": "#dc2626",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "24px",
    "marginBottom": "20px",
    "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.08)",
}

HEADER_STYLE = {
    "color": COLORS["accent"],
    "fontFamily": "'Fira Code', monospace",
    "fontWeight": "600",
    "marginBottom": "20px",
    "fontSize": "1.1rem",
}


TUMOR_GRADES = [
    {"label": "G1 - Well differentiated", "value": 1},
    {"label": "G2 - Moderately differentiated", "value": 2},
    {"label": "G3 - Poorly differentiated", "value": 3},
]

PREOP_RISK_GROUPS = [
    {"label": "Low Risk", "value": 1},
    {"label": "Intermediate Risk", "value": 2},
    {"label": "High Risk", "value": 3},
]

FINAL_RISK_GROUPS = [
    {"label": "Low Risk", "value": 1},
    {"label": "Intermediate Risk", "value": 2},
    {"label": "High-Intermediate Risk", "value": 3},
    {"label": "High Risk", "value": 4},
    {"label": "Advanced", "value": 5},
]

HISTOLOGY_OPTIONS = [
    {"label": "Atypical Hyperplasia", "value": "atypical_hyperplasia"},
    {"label": "Endometrioid Carcinoma", "value": "endometrioid"},
    {"label": "Serous Carcinoma", "value": "serous"},
    {"label": "Clear Cell Carcinoma", "value": "clear_cell"},
    {"label": "Undifferentiated Carcinoma", "value": "undifferentiated"},
    {"label": "Mixed Carcinoma", "value": "mixed"},
    {"label": "Squamous Cell Carcinoma", "value": "squamous"},
    {"label": "Carcinosarcoma", "value": "carcinosarcoma"},
    {"label": "Leiomyosarcoma", "value": "leiomyosarcoma"},
    {"label": "Endometrial Stromal Sarcoma", "value": "stromal_sarcoma"},
    {"label": "Undifferentiated Sarcoma", "value": "undiff_sarcoma"},
    {"label": "Adenosarcoma", "value": "adenosarcoma"},
    {"label": "Others", "value": "others"},
]


STAGING_OPTIONS = [
    {"label": "Stage I", "value": 1},
    {"label": "Stage II", "value": 2},
    {"label": "Stage III-IV", "value": 3},
]

MYOMETRIAL_INVASION_OPTIONS = [
    {"label": "Not Applicable", "value": 0},
    {"label": "< 50% invasion", "value": 1},
    {"label": "> 50% invasion", "value": 2},
    {"label": "Not Assessable", "value": 3},
]

SURGERY_TYPE_OPTIONS = [
    {"label": "No", "value": 0},
    {"label": "Yes", "value": 1},
]

SENTINEL_NODE_OPTIONS = [
    {"label": "Negative", "value": 0},
    {"label": "Isolated Tumor Cells (ITC)", "value": 1},
    {"label": "Micrometastasis", "value": 2},
    {"label": "Macrometastasis", "value": 3},
    {"label": "pNx (Not Assessed)", "value": 4},
]


PMS2_OPTIONS = [
    {"label": "Normal", "value": 0},
    {"label": "Abnormal", "value": 1},
    {"label": "Not Performed", "value": 2},
]

BINARY_OPTIONS = [
    {"label": "No", "value": 0},
    {"label": "Yes", "value": 1},
]

def create_numeric_input(id_name, label, placeholder, min_val=None, max_val=None, step=1, tooltip=None):
    """Create a numeric input with optional tooltip"""
    label_content = [label]
    if tooltip:
        label_content.append(html.I(className="bi bi-info-circle ms-1", id=f"{id_name}-info", 
                                    style={"cursor": "pointer", "fontSize": "0.8rem", "color": COLORS["text_muted"]}))
    
    components = [
        html.Label(label_content, className="form-label", 
                   style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        dbc.Input(
            id=id_name,
            type="number",
            placeholder=placeholder,
            min=min_val,
            max=max_val,
            step=step,
            className="mb-3",
        ),
    ]
    
    if tooltip:
        components.append(dbc.Tooltip(tooltip, target=f"{id_name}-info", placement="top"))
    
    return dbc.Col(components, md=6, lg=4)


def create_dropdown(id_name, label, options, placeholder="Select...", tooltip=None):

    if tooltip:
        label_element = html.Div([
            html.Span(label, style={"marginRight": "4px"}),
            html.I(className="bi bi-info-circle", id=f"{id_name}-info",
                  style={"cursor": "pointer", "fontSize": "0.8rem", "color": COLORS["text_muted"]}),
        ], className="form-label", style={
            "color": COLORS["text_muted"], 
            "fontSize": "0.85rem",
            "height": "24px",  
            "lineHeight": "24px",  
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "8px",
        })
    else:
        label_element = html.Label(
            label, 
            className="form-label",
            style={
                "color": COLORS["text_muted"], 
                "fontSize": "0.85rem",
                "height": "24px",  
                "lineHeight": "24px",
                "display": "block",
                "marginBottom": "8px",
            }
        )
    
    components = [
        label_element,
        dcc.Dropdown(
            id=id_name,
            options=options,
            placeholder=placeholder,
            className="mb-3",
            style={
                "width": "100%",
            },
            searchable=False,
        ),
    ]
    
    if tooltip:
        components.append(dbc.Tooltip(tooltip, target=f"{id_name}-info", placement="top"))
    
   
    return dbc.Col(components, md=4, className="mb-3")


def create_binary_toggle(id_name, label, tooltip=None):
    """Create a Yes/No toggle"""
   
    if tooltip:
        label_element = html.Div([
            html.Span(label, style={"marginRight": "4px"}),
            html.I(className="bi bi-info-circle", id=f"{id_name}-info",
                  style={"cursor": "pointer", "fontSize": "0.8rem", "color": COLORS["text_muted"]}),
        ], className="form-label", style={
            "color": COLORS["text_muted"], 
            "fontSize": "0.85rem",
            "height": "24px", 
            "lineHeight": "24px", 
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "8px",
        })
    else:
        label_element = html.Label(
            label, 
            className="form-label",
            style={
                "color": COLORS["text_muted"], 
                "fontSize": "0.85rem",
                "height": "24px",  
                "lineHeight": "24px",
                "display": "block",
                "marginBottom": "8px",
            }
        )
    
    components = [
        label_element,
        dbc.RadioItems(
            id=id_name,
            options=BINARY_OPTIONS,
            value=0,
            inline=True,
            className="mb-3",
        ),
    ]
    
    if tooltip:
        components.append(dbc.Tooltip(tooltip, target=f"{id_name}-info", placement="top"))
    
    return dbc.Col(components, md=4, className="mb-3")


def create_common_inputs():
    """Create common input fields for both models"""
    return html.Div([
        html.H5("📊 Common Features", style=HEADER_STYLE),
      
        html.H6("Numeric Variables", style={"color": COLORS["text"], "marginBottom": "15px", "marginTop": "10px"}),
        dbc.Row([
            create_numeric_input("tumor_size", "Tumor Size", "e.g., 3.5", 0, 30, 0.1, "Tumor diameter in centimeters"),
            create_numeric_input("bmi", "BMI", "e.g., 28.5", 10, 80, 0.1, "Body Mass Index (kg/m²)"),
            create_numeric_input("total_sentinel_nodes", "Sentinel Nodes", "e.g., 3", 0, 50, 1, "Total sentinel lymph nodes examined"),
        ]),
        
    
        html.H6("Tumor Characteristics", style={"color": COLORS["text"], "marginBottom": "15px", "marginTop": "20px"}),
        dbc.Row([
            create_dropdown("tumor_grade_preop", "Preop Tumor Grade", TUMOR_GRADES, tooltip="Grade assessed before surgery"),
            create_dropdown("final_grade", "Final Tumor Grade", TUMOR_GRADES, tooltip="Grade from final pathology"),
            create_dropdown("histology", "Final Histology", HISTOLOGY_OPTIONS, tooltip="Histological subtype"),
        ]),
       
        html.H6("Risk Classification", style={"color": COLORS["text"], "marginBottom": "15px", "marginTop": "20px"}),
        dbc.Row([
            create_dropdown("preop_risk_group", "Preoperative Risk Group", PREOP_RISK_GROUPS),
            create_dropdown("final_risk_group", "Final Risk Group", FINAL_RISK_GROUPS),
            create_dropdown("preop_staging", "Preoperative Staging", STAGING_OPTIONS),
        ]),
    
        html.H6("Invasion & Pathology", style={"color": COLORS["text"], "marginBottom": "15px", "marginTop": "20px"}),
        dbc.Row([
            create_dropdown("myometrial_invasion", "Myometrial Invasion", MYOMETRIAL_INVASION_OPTIONS,
                           tooltip="Depth of tumor invasion into the uterine wall"),
            create_dropdown("sentinel_node_status", "Sentinel Node Status", SENTINEL_NODE_OPTIONS),
            create_dropdown("surgery_type", "Primary Surgery Type", SURGERY_TYPE_OPTIONS),
        ], className="align-items-stretch", style={"marginBottom": "1rem"}),
        
        html.H6("Clinical Factors", style={"color": COLORS["text"], "marginBottom": "15px", "marginTop": "20px"}),
        dbc.Row([
            create_binary_toggle("distant_metastasis", "Distant Metastasis", "Presence of distant metastatic disease"),
            create_binary_toggle("chemotherapy", "Chemotherapy", "Chemotherapy administered"),
        ]),
    ], style=CARD_STYLE)


def create_os_specific_inputs():
    """Create OS-specific input fields"""
    return html.Div([
        html.H5("🔬 OS-Specific Features", style=HEADER_STYLE),
        dbc.Row([
            create_dropdown("pms2", "PMS2 Expression", PMS2_OPTIONS, 
                           tooltip="PMS2 immunohistochemistry result (MMR marker)"),
        ]),
        html.P("These features are only used for Overall Survival prediction.", 
               style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "fontStyle": "italic"}),
    ], style=CARD_STYLE)


def create_rfs_specific_inputs():
    """Create RFS-specific input fields"""
    return html.Div([
        html.H5("🔬 RFS-Specific Features", style=HEADER_STYLE),
        dbc.Row([
            create_binary_toggle("lymphovascular_invasion", "LVS Invasion", "Lymphovascular Invasion"),
            create_binary_toggle("indication_radiotherapy", "Radiotherapy Indicated"),
            create_numeric_input("surgery_time", "Surgery Time", "e.g., 120", 0, 600, 1, "Duration of surgery in minutes"),
        ], className="align-items-stretch", style={"marginBottom": "1rem"}),
        html.P("These features are only used for Recurrence-Free Survival prediction.",
               style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "fontStyle": "italic"}),
    ], style=CARD_STYLE)


def create_prediction_controls():
    """Create prediction control section"""
    return html.Div([
        html.H5("⚙️ Prediction Settings", style=HEADER_STYLE),
        dbc.Row([
            dbc.Col([
                html.Label("Model Selection", className="form-label",
                          style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id="model-selector",
                    options=[
                        {"label": "Overall Survival (OS)", "value": "os"},
                        {"label": "Recurrence-Free Survival (RFS)", "value": "rfs"},
                        {"label": "Both Models", "value": "both"},
                    ],
                    value="both",
                    className="mb-3",
                ),
            ], md=6),
            dbc.Col([
                html.Label("Prediction Horizons (Years)", className="form-label",
                          style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id="prediction-years",
                    options=[{"label": f"{i} Year{'s' if i > 1 else ''}", "value": i} for i in range(1, 11)],
                    value=[1, 3, 5],
                    multi=True,
                    className="mb-3",
                ),
            ], md=6),
        ]),
    ], style=CARD_STYLE)



def create_header():
    """Create app header"""
    return html.Div([
        html.Div([
            html.H1([
                html.Span("◆ ", style={"color": COLORS["accent"]}),
                "NSMP Endometrial Cancer",
                html.Br(),
                html.Span("Survival Calculator", style={"color": COLORS["accent"]}),
            ], style={
                "fontFamily": "'Fira Code', monospace",
                "fontWeight": "700",
                "fontSize": "2rem",
                "color": COLORS["text"],
                "marginBottom": "10px",
            }),
            html.P([
                "Cox Proportional Hazards Model • ",
                html.Span("C-index: ", style={"fontWeight": "600"}),
                "OS 0.77 | RFS 0.85",
            ], style={"color": COLORS["text_muted"], "fontSize": "0.95rem"}),
        ], style={"textAlign": "center", "padding": "30px 20px"}),
    ], style={
        "background": f"linear-gradient(135deg, {COLORS['background']} 0%, {COLORS['card']} 100%)",
        "borderBottom": f"2px solid {COLORS['accent']}",
        "marginBottom": "20px",
    })


def create_results_section():
    """Create results display section - content populated by callback"""
    return html.Div(id="results-content")


app.layout = html.Div([
    
    dcc.Store(id="results-visible-store", data=False),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="bi bi-exclamation-triangle-fill me-2", style={"color": "#dc2626"}),
            "Missing Required Fields"
        ]), close_button=True),
        dbc.ModalBody(id="error-modal-body"),
        dbc.ModalFooter(
            dbc.Button("OK", id="close-error-modal", className="ms-auto", color="primary")
        ),
    ], id="error-modal", is_open=False, centered=True),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="bi bi-shield-lock-fill me-2", style={"color": COLORS["accent"]}),
            "Ethical Considerations & Privacy"
        ]), close_button=True),
        dbc.ModalBody([
            html.Div([
                html.H5([
                    html.I(className="bi bi-lock-fill me-2"),
                    "Data Privacy & Security"
                ], style={"color": COLORS["accent"], "marginTop": "10px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([
                        html.Strong("No Data Storage: "),
                        "This application does not store, log, or transmit any patient data. All calculations are performed locally in your browser session."
                    ]),
                    html.Li([
                        html.Strong("Session-Based Processing: "),
                        "Patient inputs exist only during your active session and are cleared when you close the browser."
                    ]),
                    html.Li([
                        html.Strong("No Server-Side Storage: "),
                        "No patient data is saved to disk, databases, or external services."
                    ]),
                    html.Li([
                        html.Strong("Compliance: "),
                        "When deployed in clinical settings, ensure HIPAA/GDPR compliance with appropriate security safeguards."
                    ]),
                ], style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
                
                html.H5([
                    html.I(className="bi bi-eye-fill me-2"),
                    "Model Explainability & Transparency"
                ], style={"color": COLORS["accent"], "marginTop": "25px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([
                        html.Strong("Interpretable Model: "),
                        "Uses Cox Proportional Hazards model - a transparent statistical approach, not a 'black box'."
                    ]),
                    html.Li([
                        html.Strong("Feature Contributions: "),
                        "The application shows which clinical variables contribute most to each prediction and how (increased/decreased risk)."
                    ]),
                    html.Li([
                        html.Strong("Hazard Ratios: "),
                        "Model coefficients represent clinically meaningful hazard ratios showing effect sizes."
                    ]),
                    html.Li([
                        html.Strong("Transparency: "),
                        "All model features are clinically meaningful variables with clear interpretations."
                    ]),
                ], style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
                
                html.H5([
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    "Understanding Predictions"
                ], style={"color": COLORS["accent"], "marginTop": "25px", "marginBottom": "15px"}),
                html.P([
                    "The ", html.Strong("Feature Contributions chart"), " shows how each input variable affects the prediction:",
                ], style={"fontSize": "0.9rem"}),
                html.Ul([
                    html.Li("Positive contributions (red bars) indicate factors that increase risk"),
                    html.Li("Negative contributions (green bars) indicate factors that decrease risk"),
                    html.Li("Bar length shows the magnitude of each feature's impact"),
                    html.Li("The sum of all contributions determines the overall risk score"),
                ], style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
                
                html.H5([
                    html.I(className="bi bi-exclamation-circle-fill me-2"),
                    "Limitations & Bias"
                ], style={"color": COLORS["accent"], "marginTop": "25px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([
                        html.Strong("Cohort Limitations: "),
                        "Models trained on NSMP endometrial cancer patients (n=161) may not generalize to all populations."
                    ]),
                    html.Li([
                        html.Strong("Probabilistic Predictions: "),
                        "Predictions are probability estimates, not certainties. Individual outcomes may vary."
                    ]),
                    html.Li([
                        html.Strong("External Validation: "),
                        "Models should be validated on independent cohorts before clinical deployment."
                    ]),
                    html.Li([
                        html.Strong("Clinical Judgment: "),
                        "Always combine predictions with clinical expertise, patient preferences, and guidelines."
                    ]),
                ], style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-ethics-modal", className="ms-auto", color="primary")
        ),
    ], id="ethics-modal", is_open=False, centered=True, size="lg"),
    
    create_header(),
    
    dbc.Container([
        dbc.Row([
            
            dbc.Col([
                create_prediction_controls(),
                create_common_inputs(),
                
                
                html.Div(id="os-inputs-container", children=[create_os_specific_inputs()]),
                html.Div(id="rfs-inputs-container", children=[create_rfs_specific_inputs()]),
        
                html.Div([
                    dbc.Button(
                        "Calculate Survival Probability",
                        id="calculate-btn",
                        size="lg",
                        style={
                            "backgroundColor": COLORS["accent"],
                            "border": "none",
                            "borderRadius": "8px",
                            "fontWeight": "600",
                            "padding": "15px 40px",
                            "width": "100%",
                        },
                    ),
                ], style={"marginTop": "20px", "marginBottom": "20px"}),
                
            ], id="input-column", lg=8, md=10, className="mx-auto"),
            
            
            dbc.Col([
                create_results_section(),
            ], id="results-column", lg=7, md=12, style={"display": "none"}),
        ], id="main-content-row", justify="center"),
        

        html.Div([
            html.Hr(style={"borderColor": COLORS["border"]}),
            html.Div([
                dbc.Alert([
                    html.H6([html.I(className="bi bi-exclamation-triangle-fill me-2"), "Clinical Disclaimer"], 
                            style={"marginBottom": "5px", "fontSize": "0.9rem"}),
                    html.P([
                        "This tool is for ", html.Strong("research and educational purposes only"),
                        ". Not intended for clinical decision-making without professional medical judgment."
                    ], className="mb-0", style={"fontSize": "0.8rem"}),
                ], color="warning", style={"display": "inline-block", "maxWidth": "600px", "padding": "10px 15px"}),
            ], className="text-center", style={"marginBottom": "15px"}),
            html.Div([
                dbc.Button([
                    html.I(className="bi bi-shield-lock-fill me-2"),
                    "Privacy & Explainability"
                ], id="open-ethics-modal", outline=True, color="secondary", size="sm"),
            ], className="text-center"),
        ]),
        
    ], fluid=True, style={"maxWidth": "1400px"}),
    
], style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "paddingBottom": "40px"})


def get_risk_category(risk_score, quantiles):
    """Determine risk category based on quantiles"""
    if risk_score <= quantiles['low']:
        return "Low", COLORS["low_risk"]
    elif risk_score <= quantiles['high']:
        return "Medium", COLORS["medium_risk"]
    else:
        return "High", COLORS["high_risk"]


def prepare_os_features(inputs):
    """Prepare feature vector for OS model"""
    features = {}
    
   
    features['tumor_size'] = inputs.get('tumor_size', OS_MEDIANS.get('tumor_size', 3.0))
    features['bmi'] = inputs.get('bmi', OS_MEDIANS.get('bmi', 28))
    features['total_sentinel_nodes'] = inputs.get('total_sentinel_nodes', OS_MEDIANS.get('total_sentinel_nodes', 2))
    
    features['tumor_grade_preop_encoded'] = inputs.get('tumor_grade_preop', 2)
    features['final_grade_encoded'] = inputs.get('final_grade', 2)
    features['preoperative_risk_group_encoded'] = inputs.get('preop_risk_group', 2)
    features['final_risk_group_encoded'] = inputs.get('final_risk_group', 2)

    features['distant_metastasis'] = inputs.get('distant_metastasis', 0)
    features['chemotherapy'] = inputs.get('chemotherapy', 0)
    
    pms2_val = inputs.get('pms2', 0)
    features['pms2'] = 1 if pms2_val == 1 else 0
    
    histology = inputs.get('histology', 'endometrioid')
    features['final_histology_2.0'] = 1 if histology == 'serous' else 0
    
    features['primary_surgery_type_1.0'] = inputs.get('surgery_type', 0)
    
    
    preop_stage = inputs.get('preop_staging', 1)
    features['preoperative_staging_2.0'] = 1 if preop_stage >= 2 else 0
    
   
    sentinel_val = inputs.get('sentinel_node_status', 0)
    features['sentinel_node_pathology_4.0'] = 1 if sentinel_val == 3 else 0
    
  
    myom_inv = inputs.get('myometrial_invasion', 0)
    features['myometrial_invasion_preop_subjective_2.0'] = 1 if myom_inv == 2 else 0
    features['myometrial_invasion_preop_objective_4.0'] = 1 if myom_inv == 2 else 0
    
    return features


def prepare_rfs_features(inputs):
    """Prepare feature vector for RFS model"""
    features = {}
    
    # Numeric
    features['tumor_size'] = inputs.get('tumor_size', RFS_MEDIANS.get('tumor_size', 3.0))
    features['bmi'] = inputs.get('bmi', RFS_MEDIANS.get('bmi', 28))
    features['total_sentinel_nodes'] = inputs.get('total_sentinel_nodes', RFS_MEDIANS.get('total_sentinel_nodes', 2))
    features['surgery_time'] = inputs.get('surgery_time', RFS_MEDIANS.get('surgery_time', 120))
    
    # Encoded grades and risk groups
    features['tumor_grade_preop_encoded'] = inputs.get('tumor_grade_preop', 2)
    features['final_grade_encoded'] = inputs.get('final_grade', 2)
    features['preoperative_risk_group_encoded'] = inputs.get('preop_risk_group', 2)
    features['final_risk_group_encoded'] = inputs.get('final_risk_group', 2)
    
    # Binary
    features['distant_metastasis'] = inputs.get('distant_metastasis', 0)
    features['chemotherapy'] = inputs.get('chemotherapy', 0)
    features['lymphovascular_invasion'] = inputs.get('lymphovascular_invasion', 0)
    features['indication_radiotherapy'] = inputs.get('indication_radiotherapy', 0)
    
  
    myom_inv = inputs.get('myometrial_invasion', 0)
    features['myometrial_invasion_encoded'] = 1 if myom_inv == 2 else 0
    
    # One-hot encoded features
    histology = inputs.get('histology', 'endometrioid')
    features['final_histology_2.0'] = 1 if histology == 'serous' else 0
    features['final_histology_9.0'] = 1 if histology == 'carcinosarcoma' else 0
    
    features['primary_surgery_type_1.0'] = inputs.get('surgery_type', 0)
    
  
    preop_stage = inputs.get('preop_staging', 1)
    features['preoperative_staging_1.0'] = 1 if preop_stage == 1 else 0  
    features['preoperative_staging_2.0'] = 1 if preop_stage >= 2 else 0  
    
   
    sentinel_val = inputs.get('sentinel_node_status', 0)
    features['sentinel_node_pathology_4.0'] = 1 if sentinel_val == 3 else 0
    
   
    features['myometrial_invasion_preop_subjective_2.0'] = 1 if myom_inv == 2 else 0
    
    return features


def create_feature_contribution_chart(model, features_dict, model_name):
    """Create horizontal bar chart showing feature contributions"""
    if model is None:
        return go.Figure()
    
 
    coefs = model.model.params_
    feature_names = model.feature_names
    
    # Calculate contributions: β * x
    contributions = []
    for feat in feature_names:
        try:
            coef = coefs.loc[feat] if feat in coefs.index else 0.0
        except (KeyError, IndexError):
            coef = 0.0
        

        value = features_dict.get(feat)
        if value is None:
            value = 0.0
        else:
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
        
        contrib = coef * value
        
        contributions.append({
            'feature': FEATURE_LABELS.get(feat, feat),
            'contribution': contrib,
            'coef': coef,
            'value': value
        })
    
    # Sort by absolute contribution
    contributions = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)[:10]

    fig = go.Figure()
    
    colors = [COLORS["high_risk"] if c['contribution'] > 0 else COLORS["low_risk"] for c in contributions]
    
    fig.add_trace(go.Bar(
        y=[c['feature'] for c in contributions],
        x=[c['contribution'] for c in contributions],
        orientation='h',
        marker_color=colors,
        text=[f"{c['contribution']:.3f}" for c in contributions],
        textposition='outside',
    ))
    
    fig.update_layout(
        title=f"{model_name} Feature Contributions (β × value)",
        xaxis_title="Contribution to Log-Hazard",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        margin=dict(l=20, r=20, t=50, b=40),
        height=350,
    )

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["text_muted"])
    
    return fig



@callback(
    [Output("os-inputs-container", "style"),
     Output("rfs-inputs-container", "style")],
    Input("model-selector", "value"),
)
def toggle_model_inputs(model_selection):
    """Show/hide model-specific inputs based on selection"""
    if model_selection == "os":
        return {"display": "block"}, {"display": "none"}
    elif model_selection == "rfs":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "block"}


@callback(
    Output("error-modal", "is_open"),
    Output("close-error-modal", "n_clicks"),
    Input("close-error-modal", "n_clicks"),
    State("error-modal", "is_open"),
    prevent_initial_call=True,
)
def close_error_modal(n_clicks, is_open):
    """Close the error modal when OK is clicked"""
    return False, None


@callback(
    Output("ethics-modal", "is_open"),
    [Input("open-ethics-modal", "n_clicks"),
     Input("close-ethics-modal", "n_clicks")],
    State("ethics-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_ethics_modal(open_clicks, close_clicks, is_open):
    """Toggle the ethics modal"""
    if open_clicks or close_clicks:
        return not is_open
    return is_open


@callback(
    [Output("results-column", "style"),
     Output("input-column", "lg"),
     Output("input-column", "md"),
     Output("input-column", "className"),
     Output("results-content", "children"),
     Output("error-modal", "is_open", allow_duplicate=True),
     Output("error-modal-body", "children")],
    Input("calculate-btn", "n_clicks"),
    [State("model-selector", "value"),
     State("prediction-years", "value"),
     # Common inputs
     State("tumor_size", "value"),
     State("bmi", "value"),
     State("total_sentinel_nodes", "value"),
     State("tumor_grade_preop", "value"),
     State("final_grade", "value"),
     State("histology", "value"),
     State("preop_risk_group", "value"),
     State("final_risk_group", "value"),
     State("preop_staging", "value"),
     State("myometrial_invasion", "value"),
     State("sentinel_node_status", "value"),
     State("surgery_type", "value"),
     State("distant_metastasis", "value"),
     State("chemotherapy", "value"),
     # OS-specific
     State("pms2", "value"),
     # RFS-specific
     State("lymphovascular_invasion", "value"),
     State("indication_radiotherapy", "value"),
     State("surgery_time", "value"),
    ],
    prevent_initial_call=True,
)
def calculate_predictions(n_clicks, model_selection, prediction_years,
                          tumor_size, bmi, total_sentinel_nodes,
                          tumor_grade_preop, final_grade, histology,
                          preop_risk_group, final_risk_group, preop_staging,
                          myometrial_invasion, sentinel_node_status, surgery_type,
                          distant_metastasis, chemotherapy,
                          pms2,
                          lymphovascular_invasion,
                          indication_radiotherapy, surgery_time):
    """Main prediction callback"""
    
  
    required_fields = {
        'Preop Tumor Grade': tumor_grade_preop,
        'Final Tumor Grade': final_grade,
        'Preoperative Risk Group': preop_risk_group,
        'Final Risk Group': final_risk_group,
    }
    
    missing_fields = [name for name, value in required_fields.items() if value is None]
    
    if missing_fields:
        
        error_body = html.Div([
            html.P([
                "Please fill in the following required fields before calculating:"
            ], style={"marginBottom": "15px"}),
            html.Ul([
                html.Li(field, style={"color": COLORS["high_risk"], "fontWeight": "500"}) 
                for field in missing_fields
            ]),
            html.P([
                "Other fields are optional and will be imputed with training medians if left empty."
            ], style={"color": COLORS["text_muted"], "fontSize": "0.9rem", "marginTop": "15px"}),
        ])
        
        return (
            dash.no_update,  
            dash.no_update,  
            dash.no_update,  
            dash.no_update, 
            dash.no_update,  
            True,           
            error_body,    
        )
    
    # Default prediction years
    if not prediction_years:
        prediction_years = [1, 2, 3, 5]
    prediction_years = sorted(prediction_years)
    
    # Collect all inputs
    inputs = {
        'tumor_size': tumor_size,
        'bmi': bmi,
        'total_sentinel_nodes': total_sentinel_nodes,
        'tumor_grade_preop': tumor_grade_preop,
        'final_grade': final_grade,
        'histology': histology,
        'preop_risk_group': preop_risk_group,
        'final_risk_group': final_risk_group,
        'preop_staging': preop_staging,
        'myometrial_invasion': myometrial_invasion,
        'sentinel_node_status': sentinel_node_status,
        'surgery_type': surgery_type,
        'distant_metastasis': distant_metastasis,
        'chemotherapy': chemotherapy,
        'pms2': pms2,
        'lymphovascular_invasion': lymphovascular_invasion,
        'indication_radiotherapy': indication_radiotherapy,
        'surgery_time': surgery_time,
    }
    

    imputed_features = []
    alerts = []
    
    
    if not MODELS_LOADED:
        alerts.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            "Models not loaded. Showing placeholder predictions."
        ], color="warning"))
    
    os_results = None
    rfs_results = None
    
    
    if model_selection in ["os", "both"] and os_model is not None:
        os_features = prepare_os_features(inputs)
        
  
        for feat, val in os_features.items():
            if inputs.get(feat.replace('_encoded', '').replace('_2.0', '').replace('_4.0', '').replace('_1.0', '')) is None:
                if feat in ['tumor_size', 'bmi', 'total_sentinel_nodes']:
                    imputed_features.append(FEATURE_LABELS.get(feat, feat))
        
        
        os_df = pd.DataFrame([os_features])
        os_df = os_df[os_model.feature_names]  
        os_risk_score = os_model.predict_risk(os_df)[0]
        os_risk_cat, os_risk_color = get_risk_category(os_risk_score, OS_RISK_QUANTILES)
        
       
        os_surv_func = os_model.predict_survival_function(os_df)
        
        os_results = {
            'risk_score': os_risk_score,
            'risk_category': os_risk_cat,
            'risk_color': os_risk_color,
            'survival_function': os_surv_func,
            'features': os_features,
            'probabilities': {}
        }
        
        
        for year in prediction_years:
            days = year * 365  # Convert years to days
            times = os_surv_func.index.values
            closest_idx = np.argmin(np.abs(times - days))
            os_results['probabilities'][year] = os_surv_func.iloc[closest_idx, 0]
    
   
    if model_selection in ["rfs", "both"] and rfs_model is not None:
        rfs_features = prepare_rfs_features(inputs)
        
        
        rfs_df = pd.DataFrame([rfs_features])
        rfs_df = rfs_df[rfs_model.feature_names]  
        
        # Calculate risk score (using model method to ensure scaling is applied)
        rfs_risk_score = rfs_model.predict_risk(rfs_df)[0]
        rfs_risk_cat, rfs_risk_color = get_risk_category(rfs_risk_score, RFS_RISK_QUANTILES)
        
        # Calculate survival function (using model method to ensure scaling is applied)
        rfs_surv_func = rfs_model.predict_survival_function(rfs_df)
        
        rfs_results = {
            'risk_score': rfs_risk_score,
            'risk_category': rfs_risk_cat,
            'risk_color': rfs_risk_color,
            'survival_function': rfs_surv_func,
            'features': rfs_features,
            'probabilities': {}
        }
        
        # Get survival probabilities at selected years
        
        for year in prediction_years:
            days = year * 365  # Convert years to days
            times = rfs_surv_func.index.values
            closest_idx = np.argmin(np.abs(times - days))
            rfs_results['probabilities'][year] = rfs_surv_func.iloc[closest_idx, 0]
    
 
    

    if imputed_features:
        alerts.append(dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            f"Missing values imputed with training medians: {', '.join(imputed_features)}"
        ], color="info", className="py-2"))

    risk_cards = []
    if os_results:
        risk_cards.append(
            dbc.Col([
                html.Div([
                    html.H6("Overall Survival", style={"color": COLORS["text_muted"], "marginBottom": "5px"}),
                    html.Div([
                        html.Span(os_results['risk_category'], style={
                            "fontSize": "1.5rem",
                            "fontWeight": "700",
                            "color": os_results['risk_color'],
                        }),
                        html.Span(" Risk", style={"color": COLORS["text"], "fontSize": "1rem"}),
                    ]),
                    html.P(f"Score: {os_results['risk_score']:.2f}", 
                           style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "0"}),
                ], style={
                    "backgroundColor": COLORS["background"],
                    "borderRadius": "8px",
                    "padding": "15px",
                    "borderLeft": f"4px solid {os_results['risk_color']}",
                })
            ], md=6)
        )
    
    if rfs_results:
        risk_cards.append(
            dbc.Col([
                html.Div([
                    html.H6("Recurrence-Free Survival", style={"color": COLORS["text_muted"], "marginBottom": "5px"}),
                    html.Div([
                        html.Span(rfs_results['risk_category'], style={
                            "fontSize": "1.5rem",
                            "fontWeight": "700",
                            "color": rfs_results['risk_color'],
                        }),
                        html.Span(" Risk", style={"color": COLORS["text"], "fontSize": "1rem"}),
                    ]),
                    html.P(f"Score: {rfs_results['risk_score']:.2f}",
                           style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "0"}),
                ], style={
                    "backgroundColor": COLORS["background"],
                    "borderRadius": "8px",
                    "padding": "15px",
                    "borderLeft": f"4px solid {rfs_results['risk_color']}",
                })
            ], md=6)
        )
    
    risk_cards_row = dbc.Row(risk_cards, className="mb-3") if risk_cards else html.Div()
    
   
    fig = go.Figure()
    
    max_years = max(prediction_years) + 1
    max_days = max_years * 365  
    if os_results:
        surv_func = os_results['survival_function']
        times_days = surv_func.index.values
        times_years = times_days / 365  # Convert days to years for display
        probs = surv_func.values.flatten()
        
      
        mask = times_days <= max_days
        
        fig.add_trace(go.Scatter(
            x=times_years[mask],
            y=probs[mask],
            mode='lines',
            name='Overall Survival',
            line=dict(color=COLORS["accent"], width=3),
        ))
        
      
        fig.add_trace(go.Scatter(
            x=list(os_results['probabilities'].keys()),
            y=list(os_results['probabilities'].values()),
            mode='markers',
            name='OS Timepoints',
            marker=dict(color=COLORS["accent"], size=12, symbol='diamond'),
            showlegend=False,
        ))
    
    if rfs_results:
        surv_func = rfs_results['survival_function']
        times_days = surv_func.index.values
        times_years = times_days / 365  # Convert days to years for display
        probs = surv_func.values.flatten()
        
        mask = times_days <= max_days
        
        fig.add_trace(go.Scatter(
            x=times_years[mask],
            y=probs[mask],
            mode='lines',
            name='Recurrence-Free Survival',
            line=dict(color=COLORS["accent_secondary"], width=3),
        ))
        
        fig.add_trace(go.Scatter(
            x=list(rfs_results['probabilities'].keys()),
            y=list(rfs_results['probabilities'].values()),
            mode='markers',
            name='RFS Timepoints',
            marker=dict(color=COLORS["accent_secondary"], size=12, symbol='diamond'),
            showlegend=False,
        ))
    
    fig.update_layout(
        title="Predicted Survival Curves",
        xaxis_title="Years from Diagnosis",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05], tickformat='.0%'),
        xaxis=dict(range=[0, max_years]),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=60, r=20, t=80, b=50),
    )
    
    
    table_rows = [html.Tr([
        html.Th("Year"),
        *([html.Th("OS Probability")] if os_results else []),
        *([html.Th("RFS Probability")] if rfs_results else []),
    ])]
    
    for year in prediction_years:
        cells = [html.Td(f"{year} Year{'s' if year > 1 else ''}")]
        if os_results:
            prob = os_results['probabilities'].get(year, 0)
            cells.append(html.Td(f"{prob*100:.1f}%", style={"fontWeight": "600", "color": COLORS["accent"]}))
        if rfs_results:
            prob = rfs_results['probabilities'].get(year, 0)
            cells.append(html.Td(f"{prob*100:.1f}%", style={"fontWeight": "600", "color": COLORS["accent_secondary"]}))
        table_rows.append(html.Tr(cells))
    
    survival_table = html.Div([
        html.H5("Survival Probabilities", style={"color": COLORS["text"], "marginTop": "20px", "marginBottom": "10px"}),
        dbc.Table([html.Tbody(table_rows)], bordered=True, hover=True, size="sm",
                  style={"backgroundColor": COLORS["card"]}),
    ])
   
    if os_results and model_selection in ["os", "both"]:
        contrib_fig = create_feature_contribution_chart(os_model, os_results['features'], "OS")
    elif rfs_results:
        contrib_fig = create_feature_contribution_chart(rfs_model, rfs_results['features'], "RFS")
    else:
        contrib_fig = go.Figure()
    
   
    results_content = html.Div([
        html.H4("📈 Prediction Results", style=HEADER_STYLE),
        
      
        html.Div(alerts),

        risk_cards_row,
        
        html.Div([
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "400px"}),
        ]),
        
        
        survival_table,
        
        html.Div([
            html.Div([
                html.H5("Feature Contributions", style={"color": COLORS["text"], "marginTop": "20px", "marginBottom": "5px"}),
                html.P([
                    html.I(className="bi bi-info-circle me-2", style={"color": COLORS["text_muted"]}),
                    html.Small(
                        "This chart shows how each clinical variable contributes to the prediction. "
                        "Positive values (red) increase risk; negative values (green) decrease risk. "
                        "The model uses Cox Proportional Hazards, providing interpretable, transparent predictions.",
                        style={"color": COLORS["text_muted"], "fontStyle": "italic"}
                    ),
                ], style={"marginBottom": "10px"}),
            ]),
            dcc.Graph(figure=contrib_fig, config={"displayModeBar": False}, style={"height": "350px"}),
        ]),
        
    ], style=CARD_STYLE)
    
    return (
        {"display": "block"},  
        5,  
        12,  
        "",  
        results_content, 
        False,  
        "",     
    )


if __name__ == "__main__":
    app.run(debug=True, port=8051)
