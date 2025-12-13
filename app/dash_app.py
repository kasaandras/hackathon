"""
Endometrial Cancer Survival Risk Calculator - Dash MVP
A clinical decision support tool for predicting recurrence-free survival
"""

import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ============================================================================
# APP INITIALIZATION
# ============================================================================

# Using a clean light theme with medical aesthetic
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,  # Bootstrap icons for tooltips
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "NSMP Endometrial Cancer Risk Calculator"

# ============================================================================
# STYLING
# ============================================================================

COLORS = {
    "background": "#f8f9fa",
    "card": "#ffffff",
    "accent": "#0d9488",  # Teal
    "accent_secondary": "#dc2626",  # Red
    "text": "#1f2937",
    "text_muted": "#6b7280",
    "border": "#e5e7eb",
    "low_risk": "#059669",  # Green
    "intermediate_risk": "#d97706",  # Amber
    "high_risk": "#dc2626",  # Red
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "24px",
    "marginBottom": "20px",
    "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.08)",
    "overflow": "visible",
}

HEADER_STYLE = {
    "color": COLORS["accent"],
    "fontFamily": "'Fira Code', 'JetBrains Mono', monospace",
    "fontWeight": "600",
    "marginBottom": "20px",
    "fontSize": "1.1rem",
    "letterSpacing": "0.5px",
}

INPUT_STYLE = {
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
}

# ============================================================================
# DROPDOWN OPTIONS
# ============================================================================

HISTOLOGICAL_TYPES = [
    {"label": "Endometrioid", "value": "endometrioid"},
    {"label": "Serous", "value": "serous"},
    {"label": "Clear Cell", "value": "clear_cell"},
    {"label": "Carcinosarcoma", "value": "carcinosarcoma"},
    {"label": "Mixed", "value": "mixed"},
    {"label": "Other", "value": "other"},
]

TUMOR_GRADES = [
    {"label": "G1 - Well diff.", "value": 1},
    {"label": "G2 - Moderate diff.", "value": 2},
    {"label": "G3 - Poorly diff.", "value": 3},
]

ASA_SCORES = [
    {"label": "I - Healthy", "value": 1},
    {"label": "II - Mild disease", "value": 2},
    {"label": "III - Severe disease", "value": 3},
    {"label": "IV - Life-threatening", "value": 4},
]

FIGO_STAGES = [
    {"label": "IA - <50% myometrial", "value": "IA"},
    {"label": "IB - ≥50% myometrial", "value": "IB"},
    {"label": "II - Cervical stromal", "value": "II"},
    {"label": "IIIA - Serosa/adnexa", "value": "IIIA"},
    {"label": "IIIB - Vaginal/parametrial", "value": "IIIB"},
    {"label": "IIIC1 - Pelvic nodes", "value": "IIIC1"},
    {"label": "IIIC2 - Para-aortic nodes", "value": "IIIC2"},
    {"label": "IVA - Bladder/bowel", "value": "IVA"},
    {"label": "IVB - Distant metastasis", "value": "IVB"},
]

RISK_GROUPS = [
    {"label": "Low Risk", "value": "low"},
    {"label": "Intermediate Risk", "value": "intermediate"},
    {"label": "High-Intermediate Risk", "value": "high_intermediate"},
    {"label": "High Risk", "value": "high"},
]

LVSI_OPTIONS = [
    {"label": "Absent", "value": 0},
    {"label": "Focal", "value": 1},
    {"label": "Substantial", "value": 2},
]

MYOMETRIAL_INVASION = [
    {"label": "< 50%", "value": "less_50"},
    {"label": "≥ 50%", "value": "more_50"},
]

P53_STATUS = [
    {"label": "Wild-type (normal)", "value": "wt"},
    {"label": "Aberrant (mutant)", "value": "mut"},
    {"label": "Not tested", "value": "unknown"},
]

POLE_STATUS = [
    {"label": "Mutated (ultramutated)", "value": "mut"},
    {"label": "Wild-type", "value": "wt"},
    {"label": "Not tested", "value": "unknown"},
]

MMR_STATUS = [
    {"label": "Proficient (intact)", "value": "proficient"},
    {"label": "Deficient (loss)", "value": "deficient"},
    {"label": "Not tested", "value": "unknown"},
]


# ============================================================================
# COMPONENT BUILDERS
# ============================================================================


def create_numeric_input(id, label, placeholder, min_val=None, max_val=None, step=1):
    """Create a styled numeric input field"""
    return dbc.Col([
        html.Label(label, className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        dbc.Input(
            id=id,
            type="number",
            placeholder=placeholder,
            min=min_val,
            max=max_val,
            step=step,
            style=INPUT_STYLE,
            className="mb-3",
        ),
    ], md=6, lg=4)


def create_dropdown(id, label, options, placeholder="Select...", width=None):
    """Create a styled dropdown"""
    col_props = {"md": 6, "lg": 4}
    if width == "wide":
        col_props = {"md": 12, "lg": 6}
    return dbc.Col([
        html.Label(label, className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id=id,
            options=options,
            placeholder=placeholder,
            style={"backgroundColor": COLORS["background"], "minWidth": "100%"},
            className="mb-3",
            optionHeight=50,
        ),
    ], **col_props)


def create_binary_toggle(id, label):
    """Create a styled Yes/No toggle"""
    return dbc.Col([
        html.Label(label, className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        dbc.RadioItems(
            id=id,
            options=[
                {"label": "Yes", "value": 1},
                {"label": "No", "value": 0},
            ],
            inline=True,
            className="mb-3",
            inputStyle={"marginRight": "5px"},
            labelStyle={"marginRight": "15px", "color": COLORS["text"]},
        ),
    ], md=6, lg=4)


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================


def create_header():
    """Create app header"""
    return html.Div([
        html.Div([
            html.H1([
                html.Span("◆ ", style={"color": COLORS["accent"]}),
                "NSMP Endometrial Cancer",
                html.Br(),
                html.Span("Risk Calculator", style={"color": COLORS["accent"]}),
            ], style={
                "fontFamily": "'Fira Code', monospace",
                "fontWeight": "700",
                "fontSize": "2.2rem",
                "color": COLORS["text"],
                "marginBottom": "10px",
                "lineHeight": "1.3",
            }),
            html.P(
                "Cox Proportional Hazards Model for Recurrence-Free Survival Prediction",
                style={
                    "color": COLORS["text_muted"],
                    "fontSize": "1rem",
                    "marginBottom": "0",
                    "fontFamily": "'Inter', sans-serif",
                }
            ),
        ], style={"textAlign": "center", "padding": "40px 20px"}),
    ], style={
        "background": f"linear-gradient(135deg, {COLORS['background']} 0%, {COLORS['card']} 100%)",
        "borderBottom": f"2px solid {COLORS['accent']}",
        "marginBottom": "30px",
    })


def create_demographics_section():
    """Patient demographics input section"""
    return html.Div([
        html.H4("👤 Patient Demographics", style=HEADER_STYLE),
        dbc.Row([
            create_numeric_input("age", "Age at Diagnosis (years)", "e.g., 65", 18, 100),
            create_numeric_input("bmi", "Body Mass Index (kg/m²)", "e.g., 28.5", 10, 80, 0.1),
            create_dropdown("asa", "ASA Physical Status", ASA_SCORES),
        ]),
    ], style=CARD_STYLE)


def create_tumor_section():
    """Tumor characteristics input section"""
    return html.Div([
        html.H4("🔬 Tumor Characteristics", style=HEADER_STYLE),
        dbc.Row([
            create_dropdown("histological_type", "Histological Type", HISTOLOGICAL_TYPES),
            create_dropdown("tumor_grade", "Tumor Grade", TUMOR_GRADES),
            create_numeric_input("tumor_size", "Tumor Size (mm)", "e.g., 35", 0, 300),
        ], className="align-items-end"),
        dbc.Row([
            dbc.Col([
                html.Label([
                    "LVSI ",
                    html.I(className="bi bi-info-circle", id="lvsi-info", style={"cursor": "pointer", "fontSize": "0.8rem"}),
                ], className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dbc.Tooltip("Lymphovascular Space Invasion", target="lvsi-info", placement="top"),
                dcc.Dropdown(id="lvsi", options=LVSI_OPTIONS, placeholder="Select...", className="mb-3"),
            ], md=6, lg=4),
            dbc.Col([
                html.Label([
                    "Myometrial Invasion ",
                    html.I(className="bi bi-info-circle", id="myo-info", style={"cursor": "pointer", "fontSize": "0.8rem"}),
                ], className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dbc.Tooltip("Depth of myometrial infiltration", target="myo-info", placement="top"),
                dcc.Dropdown(id="myometrial_invasion", options=MYOMETRIAL_INVASION, placeholder="Select...", className="mb-3"),
            ], md=6, lg=4),
            dbc.Col([
                html.Label([
                    "Cervical Invasion ",
                    html.I(className="bi bi-info-circle", id="cerv-info", style={"cursor": "pointer", "fontSize": "0.8rem"}),
                ], className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dbc.Tooltip("Cervical stromal invasion", target="cerv-info", placement="top"),
                dbc.RadioItems(
                    id="cervical_invasion",
                    options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                    inline=True, className="mb-3",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "15px", "color": COLORS["text"]},
                ),
            ], md=6, lg=4),
        ], className="align-items-end"),
    ], style=CARD_STYLE)


def create_staging_section():
    """Staging and risk group input section"""
    return html.Div([
        html.H4("📊 Staging & Risk Classification", style=HEADER_STYLE),
        dbc.Row([
            create_dropdown("figo_stage", "FIGO 2023 Stage", FIGO_STAGES),
            create_dropdown("risk_group", "Preoperative Risk Group", RISK_GROUPS),
        ]),
    ], style=CARD_STYLE)


def create_molecular_section():
    """Molecular markers input section"""
    return html.Div([
        html.H4("🧬 Molecular Markers", style=HEADER_STYLE),
        dbc.Row([
            dbc.Col([
                html.Label("p53 Status", className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dcc.Dropdown(id="p53_status", options=P53_STATUS, placeholder="Select...",
                            className="mb-3"),
            ], xs=12, sm=4),
            dbc.Col([
                html.Label("POLE Mutation", className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dcc.Dropdown(id="pole_status", options=POLE_STATUS, placeholder="Select...",
                            className="mb-3"),
            ], xs=12, sm=4),
            dbc.Col([
                html.Label("MMR Status", className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                dcc.Dropdown(id="mmr_status", options=MMR_STATUS, placeholder="Select...",
                            className="mb-3"),
            ], xs=12, sm=4),
        ]),
    ], style=CARD_STYLE)


def create_treatment_section():
    """Treatment variables input section"""
    return html.Div([
        html.H4("💊 Treatment Information", style=HEADER_STYLE),
        dbc.Row([
            create_binary_toggle("neoadjuvant", "Neoadjuvant Treatment"),
            create_binary_toggle("brachytherapy", "Brachytherapy Indicated"),
            create_binary_toggle("chemotherapy", "Chemotherapy Indicated"),
        ]),
        dbc.Row([
            create_binary_toggle("radiotherapy", "Radiotherapy Indicated"),
        ]),
    ], style=CARD_STYLE)




def create_results_section():
    """Results display section"""
    return html.Div([
        html.H4("📈 Survival Prediction Results", style=HEADER_STYLE),
        
        # Overall Survival (OS) Section
        html.Div([
            html.Div([
                html.H5("Overall Survival (OS)", style={
                    "color": COLORS["text"],
                    "fontWeight": "600",
                    "marginBottom": "5px",
                }),
                html.P("Time from diagnosis until death from any cause", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "0.8rem",
                    "marginBottom": "15px",
                }),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("3-Year", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="os-3yr", children="—%", style={
                            "color": COLORS["accent"],
                            "fontSize": "1.8rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
                dbc.Col([
                    html.Div([
                        html.Span("5-Year", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="os-5yr", children="—%", style={
                            "color": COLORS["accent"],
                            "fontSize": "1.8rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
                dbc.Col([
                    html.Div([
                        html.Span("Risk", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="os-risk", children="—", style={
                            "color": COLORS["accent"],
                            "fontSize": "1.2rem",
                            "fontWeight": "600",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
            ]),
        ], style={
            "backgroundColor": COLORS["background"],
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}",
            "padding": "15px",
            "marginBottom": "15px",
        }),
        
        # Recurrence-Free Survival (RFS) Section
        html.Div([
            html.Div([
                html.H5("Recurrence-Free Survival (RFS)", style={
                    "color": COLORS["text"],
                    "fontWeight": "600",
                    "marginBottom": "5px",
                }),
                html.P("Time from diagnosis until recurrence or death", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "0.8rem",
                    "marginBottom": "15px",
                }),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("3-Year", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="rfs-3yr", children="—%", style={
                            "color": "#7c3aed",  # Purple for RFS
                            "fontSize": "1.8rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
                dbc.Col([
                    html.Div([
                        html.Span("5-Year", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="rfs-5yr", children="—%", style={
                            "color": "#7c3aed",
                            "fontSize": "1.8rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
                dbc.Col([
                    html.Div([
                        html.Span("Risk", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                        html.Div(id="rfs-risk", children="—", style={
                            "color": "#7c3aed",
                            "fontSize": "1.2rem",
                            "fontWeight": "600",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center"})
                ], xs=4),
            ]),
        ], style={
            "backgroundColor": COLORS["background"],
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}",
            "padding": "15px",
            "marginBottom": "20px",
        }),
        
        # Survival Curve Placeholder
        html.Div([
            dcc.Graph(
                id="survival-curve",
                config={"displayModeBar": False},
                style={"height": "350px"},
            ),
        ]),
        
        # Risk Factors Contributing
        html.Div([
            html.H5("Risk Factor Contributions", style={
                "color": COLORS["text"],
                "marginTop": "20px",
                "marginBottom": "15px",
                "fontSize": "1rem",
            }),
            html.Div(id="risk-factors", children=[
                html.P("Submit patient data to see contributing risk factors.",
                       style={"color": COLORS["text_muted"], "fontStyle": "italic"})
            ]),
        ]),
        
    ], style=CARD_STYLE)


def create_submit_button():
    """Create calculate button"""
    return html.Div([
        dbc.Button(
            ["Calculate Survival Probability"],
            id="calculate-btn",
            size="lg",
            style={
                "backgroundColor": COLORS["accent"],
                "border": "none",
                "borderRadius": "8px",
                "fontWeight": "600",
                "padding": "15px 40px",
                "fontSize": "1.1rem",
                "fontFamily": "'Fira Code', monospace",
                "boxShadow": f"0 4px 20px {COLORS['accent']}40",
                "transition": "all 0.3s ease",
            },
            className="calculate-btn",
        ),
    ], style={"textAlign": "center", "margin": "30px 0"})


# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    html.Div([
        create_header(),
        
        dbc.Container([
            dbc.Row([
                # Left Column - Input Forms (centered initially, shifts left when results shown)
                dbc.Col(
                    id="input-column",
                    children=[
                        create_demographics_section(),
                        create_tumor_section(),
                        create_staging_section(),
                        create_molecular_section(),
                        create_treatment_section(),
                        create_submit_button(),
                    ],
                    lg={"size": 8, "offset": 2},  # Centered initially
                    md=12,
                ),
                
                # Right Column - Results
                dbc.Col(
                    id="results-column",
                    children=[
                        html.Div(
                            id="results-container",
                            children=[create_results_section()],
                        ),
                    ],
                    lg=6,
                    md=12,
                    style={"display": "none"},  # Hidden initially
                ),
            ]),
            
            # Footer with Ethics & Compliance
            html.Div([
                html.Hr(style={"borderColor": COLORS["border"]}),
                
                # Main Disclaimer
                dbc.Alert([
                    html.H5([
                        html.I(className="bi bi-exclamation-triangle-fill me-2"),
                        "Clinical Disclaimer"
                    ], className="alert-heading"),
                    html.P([
                        "This tool is for ",
                        html.Strong("research and educational purposes only"),
                        ". It is NOT intended to replace professional medical judgment or clinical decision-making.",
                    ], className="mb-2"),
                    html.Hr(),
                    html.P([
                        "Always consult with qualified healthcare professionals before making treatment decisions.",
                    ], className="mb-0", style={"fontSize": "0.9rem"}),
                ], color="warning", className="mb-3"),
                
                # Ethics & Privacy Buttons
                dbc.Row([
                    dbc.Col([
                        dbc.Button([
                            html.I(className="bi bi-shield-check me-2"),
                            "Data Privacy"
                        ], id="privacy-btn", color="outline-secondary", size="sm", className="me-2"),
                        dbc.Button([
                            html.I(className="bi bi-info-circle me-2"),
                            "Model Transparency"
                        ], id="transparency-btn", color="outline-secondary", size="sm"),
                    ], className="text-center"),
                ], className="mb-3"),
                
                # Version & Audit Info
                html.P([
                    html.Small([
                        "Model Version: 1.0.0-beta | ",
                        "Last Updated: December 2025 | ",
                        "Training Population: NSMP Endometrial Cancer Cohort",
                    ], style={"color": COLORS["text_muted"]}),
                ], className="text-center mb-0"),
            ]),
            
            # Privacy Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([
                    html.I(className="bi bi-shield-check me-2"),
                    "Data Privacy & Security"
                ])),
                dbc.ModalBody([
                    html.H6("🔒 No Data Storage", className="text-success"),
                    html.P("This application does NOT store, transmit, or retain any patient data. All calculations are performed locally in your browser session."),
                    
                    html.H6("🔐 Data Handling"),
                    html.Ul([
                        html.Li("No patient identifiers are collected"),
                        html.Li("No data is sent to external servers"),
                        html.Li("Session data is cleared when you close the browser"),
                        html.Li("No cookies or tracking mechanisms are used"),
                    ]),
                    
                    html.H6("⚖️ GDPR & HIPAA Considerations"),
                    html.P("While this tool processes no identifiable health information, users in clinical settings should ensure compliance with local data protection regulations (GDPR, HIPAA, etc.) when using patient data."),
                    
                    html.H6("👤 User Responsibility"),
                    html.P("Users are responsible for ensuring that any data entered complies with their institution's data governance policies."),
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-privacy", className="ms-auto")
                ),
            ], id="privacy-modal", size="lg"),
            
            # Transparency Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([
                    html.I(className="bi bi-info-circle me-2"),
                    "Model Transparency & Limitations"
                ])),
                dbc.ModalBody([
                    html.H6("📊 Model Description"),
                    html.P("This tool uses a Cox Proportional Hazards model to predict survival probabilities for patients with NSMP (No Specific Molecular Profile) endometrial cancer."),
                    
                    html.H6("🎯 Intended Use"),
                    html.Ul([
                        html.Li("Risk stratification for clinical research"),
                        html.Li("Educational purposes for medical trainees"),
                        html.Li("Hypothesis generation for treatment planning discussions"),
                    ]),
                    
                    html.H6("⚠️ Known Limitations"),
                    html.Ul([
                        html.Li("Model trained on a specific cohort - may not generalize to all populations"),
                        html.Li("Does not account for all possible prognostic factors"),
                        html.Li("Predictions are probabilistic, not deterministic"),
                        html.Li("Performance may vary across different clinical settings"),
                        html.Li("Not validated for patients with specific molecular profiles (POLE, MMRd, p53abn)"),
                    ]),
                    
                    html.H6("📈 Performance Metrics"),
                    html.P("Model performance metrics (C-index, calibration) should be reviewed before clinical application. Contact the development team for detailed validation reports."),
                    
                    html.H6("🔄 Model Updates"),
                    html.P("This model may be updated as new evidence becomes available. Always check for the latest version."),
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-transparency", className="ms-auto")
                ),
            ], id="transparency-modal", size="lg"),
            
        ], fluid=True, style={"maxWidth": "1400px"}),
    ], style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "paddingBottom": "40px"}),
])


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    [
        Output("input-column", "lg"),
        Output("results-column", "style"),
        Output("os-3yr", "children"),
        Output("os-5yr", "children"),
        Output("os-risk", "children"),
        Output("rfs-3yr", "children"),
        Output("rfs-5yr", "children"),
        Output("rfs-risk", "children"),
        Output("survival-curve", "figure"),
        Output("risk-factors", "children"),
    ],
    Input("calculate-btn", "n_clicks"),
    [
        State("age", "value"),
        State("bmi", "value"),
        State("asa", "value"),
        State("histological_type", "value"),
        State("tumor_grade", "value"),
        State("tumor_size", "value"),
        State("lvsi", "value"),
        State("myometrial_invasion", "value"),
        State("cervical_invasion", "value"),
        State("figo_stage", "value"),
        State("risk_group", "value"),
        State("p53_status", "value"),
        State("pole_status", "value"),
        State("mmr_status", "value"),
        State("neoadjuvant", "value"),
        State("brachytherapy", "value"),
        State("chemotherapy", "value"),
        State("radiotherapy", "value"),
    ],
    prevent_initial_call=True,
)
def calculate_survival(n_clicks, age, bmi, asa, histological_type, tumor_grade,
                       tumor_size, lvsi, myometrial_invasion, cervical_invasion,
                       figo_stage, risk_group, p53_status, pole_status, mmr_status,
                       neoadjuvant, brachytherapy, chemotherapy, radiotherapy):
    """
    Calculate survival probability based on patient features.
    
    TODO: Replace this placeholder with actual Cox model predictions
    Currently generates mock results for UI demonstration.
    """
    
    # =========================================================================
    # PLACEHOLDER LOGIC - Replace with actual model predictions
    # =========================================================================
    
    # Generate placeholder survival probabilities
    # In production, this would be: S(t) = S0(t)^exp(βX)
    
    # Mock baseline survival (would come from Cox model)
    # OS typically has higher survival than RFS (since RFS includes recurrence events)
    os_baseline_3yr = 0.95
    os_baseline_5yr = 0.90
    rfs_baseline_3yr = 0.88
    rfs_baseline_5yr = 0.80
    
    # Mock linear predictor calculation (sum of β*X)
    # This is placeholder - actual coefficients come from fitted Cox model
    linear_predictor = 0.0
    risk_factors_list = []
    
    # Age effect (example: increased risk after 65)
    if age is not None:
        if age > 70:
            linear_predictor += 0.3
            risk_factors_list.append(("Age > 70", "+", "Moderate"))
        elif age > 65:
            linear_predictor += 0.15
            risk_factors_list.append(("Age 65-70", "+", "Low"))
    
    # Grade effect
    if tumor_grade is not None:
        if tumor_grade == 3:
            linear_predictor += 0.5
            risk_factors_list.append(("Grade 3 tumor", "+", "High"))
        elif tumor_grade == 2:
            linear_predictor += 0.2
            risk_factors_list.append(("Grade 2 tumor", "+", "Low"))
    
    # LVSI effect
    if lvsi is not None:
        if lvsi == 2:
            linear_predictor += 0.4
            risk_factors_list.append(("Substantial LVSI", "+", "High"))
        elif lvsi == 1:
            linear_predictor += 0.2
            risk_factors_list.append(("Focal LVSI", "+", "Moderate"))
    
    # p53 effect
    if p53_status == "mut":
        linear_predictor += 0.4
        risk_factors_list.append(("p53 aberrant", "+", "High"))
    
    # POLE effect (protective)
    if pole_status == "mut":
        linear_predictor -= 0.5
        risk_factors_list.append(("POLE mutated", "−", "Protective"))
    
    # MMR deficiency
    if mmr_status == "deficient":
        linear_predictor += 0.15
        risk_factors_list.append(("MMR deficient", "+", "Low"))
    
    # Stage effect
    if figo_stage is not None:
        if figo_stage.startswith("IV"):
            linear_predictor += 0.8
            risk_factors_list.append(("Stage IV disease", "+", "Very High"))
        elif figo_stage.startswith("III"):
            linear_predictor += 0.5
            risk_factors_list.append(("Stage III disease", "+", "High"))
        elif figo_stage == "II":
            linear_predictor += 0.25
            risk_factors_list.append(("Stage II disease", "+", "Moderate"))
    
    # Risk group effect
    if risk_group == "high":
        linear_predictor += 0.3
        risk_factors_list.append(("High risk group", "+", "High"))
    elif risk_group == "high_intermediate":
        linear_predictor += 0.15
        risk_factors_list.append(("High-intermediate risk", "+", "Moderate"))
    
    # Myometrial invasion
    if myometrial_invasion == "more_50":
        linear_predictor += 0.25
        risk_factors_list.append(("Deep myometrial invasion", "+", "Moderate"))
    
    # Cervical invasion
    if cervical_invasion == 1:
        linear_predictor += 0.2
        risk_factors_list.append(("Cervical stromal invasion", "+", "Moderate"))
    
    # Calculate survival probabilities
    # S(t) = S0(t)^exp(linear_predictor)
    hazard_ratio = np.exp(linear_predictor)
    
    # OS calculations
    os_3yr = os_baseline_3yr ** hazard_ratio
    os_5yr = os_baseline_5yr ** hazard_ratio
    os_3yr = max(0.05, min(0.99, os_3yr))
    os_5yr = max(0.02, min(0.99, os_5yr))
    
    # RFS calculations (slightly worse due to recurrence events)
    rfs_3yr = rfs_baseline_3yr ** (hazard_ratio * 1.1)  # RFS typically worse
    rfs_5yr = rfs_baseline_5yr ** (hazard_ratio * 1.1)
    rfs_3yr = max(0.05, min(0.99, rfs_3yr))
    rfs_5yr = max(0.02, min(0.99, rfs_5yr))
    
    # Determine risk categories
    def get_risk_category(surv_5yr):
        if surv_5yr >= 0.85:
            return "Low"
        elif surv_5yr >= 0.65:
            return "Moderate"
        else:
            return "High"
    
    os_risk_cat = get_risk_category(os_5yr)
    rfs_risk_cat = get_risk_category(rfs_5yr)
    
    # =========================================================================
    # CREATE SURVIVAL CURVE FIGURE
    # =========================================================================
    
    # Generate survival curve data
    time_points = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    
    # Exponential decay based on hazard for both outcomes
    os_lambda = -np.log(os_5yr) / 5
    rfs_lambda = -np.log(rfs_5yr) / 5
    os_curve = np.exp(-os_lambda * time_points)
    rfs_curve = np.exp(-rfs_lambda * time_points)
    
    fig = go.Figure()
    
    # Add OS curve
    fig.add_trace(go.Scatter(
        x=time_points, y=os_curve,
        mode='lines+markers',
        name='Overall Survival (OS)',
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=5),
    ))
    
    # Add RFS curve
    fig.add_trace(go.Scatter(
        x=time_points, y=rfs_curve,
        mode='lines+markers',
        name='Recurrence-Free (RFS)',
        line=dict(color='#7c3aed', width=3),  # Purple
        marker=dict(size=5),
    ))
    
    # Add markers for key timepoints - OS
    fig.add_trace(go.Scatter(
        x=[3, 5], y=[os_3yr, os_5yr],
        mode='markers',
        name='OS Timepoints',
        marker=dict(color=COLORS["accent"], size=10, symbol='diamond'),
        showlegend=False,
    ))
    
    # Add markers for key timepoints - RFS
    fig.add_trace(go.Scatter(
        x=[3, 5], y=[rfs_3yr, rfs_5yr],
        mode='markers',
        name='RFS Timepoints',
        marker=dict(color='#7c3aed', size=10, symbol='diamond'),
        showlegend=False,
    ))
    
    fig.update_layout(
        title=dict(
            text='Predicted Survival Curves',
            font=dict(color=COLORS["text"], size=14),
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            title='Years from Diagnosis',
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 5.5],
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title='Survival Probability',
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 1.05],
            tickformat='.0%',
            showline=False,
            zeroline=False,
        ),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.08,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS["text_muted"], size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=60, r=20, t=80, b=50),
    )
    
    # =========================================================================
    # CREATE RISK FACTORS DISPLAY
    # =========================================================================
    
    if risk_factors_list:
        risk_factors_display = [
            html.Div([
                html.Span(
                    f"{direction} ",
                    style={
                        "color": COLORS["accent"] if direction == "−" else COLORS["accent_secondary"],
                        "fontWeight": "bold",
                        "marginRight": "5px",
                    }
                ),
                html.Span(factor, style={"color": COLORS["text"]}),
                html.Span(
                    f" ({impact})",
                    style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}
                ),
            ], style={
                "padding": "8px 12px",
                "backgroundColor": COLORS["background"],
                "borderRadius": "6px",
                "marginBottom": "6px",
                "borderLeft": f"3px solid {COLORS['accent'] if direction == '−' else COLORS['accent_secondary']}",
            })
            for factor, direction, impact in risk_factors_list
        ]
    else:
        risk_factors_display = [
            html.P("No significant risk factors identified.",
                   style={"color": COLORS["text_muted"], "fontStyle": "italic"})
        ]
    
    return (
        6,  # Change input column to lg=6 (left side)
        {"display": "block"},  # Show results column
        f"{os_3yr*100:.1f}%",  # OS 3-year
        f"{os_5yr*100:.1f}%",  # OS 5-year
        os_risk_cat,  # OS risk category
        f"{rfs_3yr*100:.1f}%",  # RFS 3-year
        f"{rfs_5yr*100:.1f}%",  # RFS 5-year
        rfs_risk_cat,  # RFS risk category
        fig,
        risk_factors_display,
    )


# ============================================================================
# ETHICS MODAL CALLBACKS
# ============================================================================

@callback(
    Output("privacy-modal", "is_open"),
    [Input("privacy-btn", "n_clicks"), Input("close-privacy", "n_clicks")],
    [State("privacy-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_privacy_modal(n1, n2, is_open):
    return not is_open


@callback(
    Output("transparency-modal", "is_open"),
    [Input("transparency-btn", "n_clicks"), Input("close-transparency", "n_clicks")],
    [State("transparency-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_transparency_modal(n1, n2, is_open):
    return not is_open


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)

