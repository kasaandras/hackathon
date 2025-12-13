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

# Using a custom dark theme with medical aesthetic
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "NSMP Endometrial Cancer Risk Calculator"

# ============================================================================
# STYLING
# ============================================================================

COLORS = {
    "background": "#0a0f1a",
    "card": "#121a2e",
    "accent": "#00d4aa",
    "accent_secondary": "#ff6b6b",
    "text": "#e8e8e8",
    "text_muted": "#8892a0",
    "border": "#2a3a5a",
    "low_risk": "#00d4aa",
    "intermediate_risk": "#ffd93d",
    "high_risk": "#ff6b6b",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "24px",
    "marginBottom": "20px",
    "boxShadow": "0 4px 20px rgba(0, 0, 0, 0.3)",
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
    "backgroundColor": COLORS["background"],
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
    {"label": "Grade 1 (Well differentiated)", "value": 1},
    {"label": "Grade 2 (Moderately differentiated)", "value": 2},
    {"label": "Grade 3 (Poorly differentiated)", "value": 3},
]

ASA_SCORES = [
    {"label": "I - Healthy patient", "value": 1},
    {"label": "II - Mild systemic disease", "value": 2},
    {"label": "III - Severe systemic disease", "value": 3},
    {"label": "IV - Life-threatening disease", "value": 4},
]

FIGO_STAGES = [
    {"label": "IA - Tumor confined to uterus, <50% myometrial invasion", "value": "IA"},
    {"label": "IB - Tumor confined to uterus, ≥50% myometrial invasion", "value": "IB"},
    {"label": "II - Cervical stromal invasion", "value": "II"},
    {"label": "IIIA - Tumor invades serosa/adnexa", "value": "IIIA"},
    {"label": "IIIB - Vaginal/parametrial involvement", "value": "IIIB"},
    {"label": "IIIC1 - Pelvic node involvement", "value": "IIIC1"},
    {"label": "IIIC2 - Para-aortic node involvement", "value": "IIIC2"},
    {"label": "IVA - Bladder/bowel mucosa invasion", "value": "IVA"},
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


def create_dropdown(id, label, options, placeholder="Select..."):
    """Create a styled dropdown"""
    return dbc.Col([
        html.Label(label, className="form-label", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id=id,
            options=options,
            placeholder=placeholder,
            style={"backgroundColor": COLORS["background"]},
            className="mb-3 dash-dropdown-dark",
        ),
    ], md=6, lg=4)


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
        ]),
        dbc.Row([
            create_dropdown("lvsi", "Lymphovascular Space Invasion", LVSI_OPTIONS),
            create_dropdown("myometrial_invasion", "Myometrial Infiltration", MYOMETRIAL_INVASION),
            create_binary_toggle("cervical_invasion", "Cervical Stromal Invasion"),
        ]),
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
            create_dropdown("p53_status", "p53 Status", P53_STATUS),
            create_dropdown("pole_status", "POLE Mutation", POLE_STATUS),
            create_dropdown("mmr_status", "MMR Status (MSH2/MSH6/MLH1/PMS2)", MMR_STATUS),
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
        
        # Risk Score Display
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("3-Year Survival", style={
                            "color": COLORS["text_muted"],
                            "fontSize": "0.9rem",
                            "display": "block",
                            "marginBottom": "8px",
                        }),
                        html.Span(id="survival-3yr", children="—%", style={
                            "color": COLORS["accent"],
                            "fontSize": "2.5rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center", "padding": "20px"})
                ], md=4),
                
                dbc.Col([
                    html.Div([
                        html.Span("5-Year Survival", style={
                            "color": COLORS["text_muted"],
                            "fontSize": "0.9rem",
                            "display": "block",
                            "marginBottom": "8px",
                        }),
                        html.Span(id="survival-5yr", children="—%", style={
                            "color": COLORS["accent"],
                            "fontSize": "2.5rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center", "padding": "20px"})
                ], md=4),
                
                dbc.Col([
                    html.Div([
                        html.Span("Risk Category", style={
                            "color": COLORS["text_muted"],
                            "fontSize": "0.9rem",
                            "display": "block",
                            "marginBottom": "8px",
                        }),
                        html.Span(id="risk-category", children="—", style={
                            "color": COLORS["accent"],
                            "fontSize": "1.8rem",
                            "fontWeight": "700",
                            "fontFamily": "'Fira Code', monospace",
                        }),
                    ], style={"textAlign": "center", "padding": "20px"})
                ], md=4),
            ]),
        ], style={
            "backgroundColor": COLORS["background"],
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}",
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
            [html.Span("⚡ "), "Calculate Survival Probability"],
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
                # Left Column - Input Forms
                dbc.Col([
                    create_demographics_section(),
                    create_tumor_section(),
                    create_staging_section(),
                    create_molecular_section(),
                    create_treatment_section(),
                    create_submit_button(),
                ], lg=6),
                
                # Right Column - Results
                dbc.Col([
                    create_results_section(),
                ], lg=6),
            ]),
            
            # Footer
            html.Div([
                html.Hr(style={"borderColor": COLORS["border"]}),
                html.P([
                    "⚠️ ",
                    html.Strong("Clinical Disclaimer: "),
                    "This tool is for research and educational purposes only. ",
                    "Not intended for clinical decision-making without physician review.",
                ], style={
                    "color": COLORS["text_muted"],
                    "fontSize": "0.85rem",
                    "textAlign": "center",
                    "marginTop": "20px",
                }),
            ]),
        ], fluid=True, style={"maxWidth": "1400px"}),
    ], style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "paddingBottom": "40px"}),
])


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    [
        Output("survival-3yr", "children"),
        Output("survival-5yr", "children"),
        Output("risk-category", "children"),
        Output("risk-category", "style"),
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
    baseline_3yr = 0.92
    baseline_5yr = 0.85
    
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
    survival_3yr = baseline_3yr ** hazard_ratio
    survival_5yr = baseline_5yr ** hazard_ratio
    
    # Clamp values
    survival_3yr = max(0.05, min(0.99, survival_3yr))
    survival_5yr = max(0.02, min(0.99, survival_5yr))
    
    # Determine risk category
    if survival_5yr >= 0.85:
        risk_category = "Low Risk"
        risk_color = COLORS["low_risk"]
    elif survival_5yr >= 0.65:
        risk_category = "Intermediate"
        risk_color = COLORS["intermediate_risk"]
    else:
        risk_category = "High Risk"
        risk_color = COLORS["high_risk"]
    
    # =========================================================================
    # CREATE SURVIVAL CURVE FIGURE
    # =========================================================================
    
    # Generate survival curve data
    time_points = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    
    # Exponential decay based on hazard
    lambda_param = -np.log(survival_5yr) / 5
    survival_curve = np.exp(-lambda_param * time_points)
    
    # Comparison curves (average low and high risk)
    low_risk_curve = 0.95 ** (time_points / 5 * 0.8)
    high_risk_curve = 0.95 ** (time_points / 5 * 3)
    
    fig = go.Figure()
    
    # Add reference curves
    fig.add_trace(go.Scatter(
        x=time_points, y=low_risk_curve,
        mode='lines',
        name='Low Risk (Reference)',
        line=dict(color=COLORS["low_risk"], width=1, dash='dot'),
        opacity=0.5,
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=high_risk_curve,
        mode='lines',
        name='High Risk (Reference)',
        line=dict(color=COLORS["high_risk"], width=1, dash='dot'),
        opacity=0.5,
    ))
    
    # Add patient curve
    fig.add_trace(go.Scatter(
        x=time_points, y=survival_curve,
        mode='lines+markers',
        name='This Patient',
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=6),
    ))
    
    # Add markers for 3yr and 5yr
    fig.add_trace(go.Scatter(
        x=[3, 5], y=[survival_3yr, survival_5yr],
        mode='markers+text',
        name='Key Timepoints',
        marker=dict(color=COLORS["accent"], size=12, symbol='diamond'),
        text=[f'{survival_3yr*100:.1f}%', f'{survival_5yr*100:.1f}%'],
        textposition='top center',
        textfont=dict(color=COLORS["text"]),
        showlegend=False,
    ))
    
    fig.update_layout(
        title=dict(
            text='Predicted Recurrence-Free Survival Curve',
            font=dict(color=COLORS["text"], size=14),
        ),
        xaxis=dict(
            title='Years from Diagnosis',
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 5.5],
        ),
        yaxis=dict(
            title='Survival Probability',
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 1.05],
            tickformat='.0%',
        ),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["card"],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color=COLORS["text_muted"], size=10),
        ),
        margin=dict(l=50, r=20, t=60, b=50),
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
    
    # Return all outputs
    risk_category_style = {
        "color": risk_color,
        "fontSize": "1.8rem",
        "fontWeight": "700",
        "fontFamily": "'Fira Code', monospace",
    }
    
    return (
        f"{survival_3yr*100:.1f}%",
        f"{survival_5yr*100:.1f}%",
        risk_category,
        risk_category_style,
        fig,
        risk_factors_display,
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)

