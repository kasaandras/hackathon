# Ethical Considerations for NSMP Endometrial Cancer Risk Calculator

## Overview

This document outlines the ethical framework guiding the development and deployment of the NSMP Endometrial Cancer Survival Risk Calculator. We are committed to responsible AI development in healthcare.

---

## 1. Data Privacy & Security

### 1.1 No Data Collection
- **No patient identifiers** are collected, stored, or transmitted
- All calculations are performed **client-side** in the user's browser
- No server-side logging of patient information
- Session data is **not persisted** after browser closure

### 1.2 Compliance Considerations
- Users in clinical settings must ensure compliance with:
  - **GDPR** (General Data Protection Regulation) - EU
  - **HIPAA** (Health Insurance Portability and Accountability Act) - US
  - Local data protection regulations

### 1.3 Institutional Responsibility
- Users are responsible for compliance with their institution's data governance policies
- De-identification of patient data before entry is recommended for demonstration purposes

---

## 2. Transparency & Explainability

### 2.1 Model Documentation
- **Algorithm**: Cox Proportional Hazards Model
- **Outcomes**: Overall Survival (OS) and Recurrence-Free Survival (RFS)
- **Training Population**: NSMP endometrial cancer cohort
- **Features**: Clinical, pathological, and molecular variables

### 2.2 Explainable Predictions
- Risk factor contributions are displayed to users
- Clear indication of which factors increase or decrease risk
- Survival curves provide visual interpretation

### 2.3 Uncertainty Communication
- Predictions are probabilistic, not deterministic
- Risk categories (Low/Moderate/High) help interpret results
- Confidence intervals should be included in future versions

---

## 3. Fairness & Bias Mitigation

### 3.1 Training Data Considerations
- Model trained on specific institutional cohort
- **Potential biases**:
  - Geographic/demographic representation
  - Treatment patterns specific to training institution
  - Temporal changes in clinical practice

### 3.2 Generalizability Limitations
- Performance may vary across:
  - Different ethnic/racial populations
  - Different healthcare systems
  - Patients with rare presentations

### 3.3 Ongoing Monitoring
- Regular audits for differential performance across subgroups
- Collection of feedback for continuous improvement
- Periodic revalidation with diverse datasets

---

## 4. Clinical Safety

### 4.1 Intended Use
✅ **Appropriate Uses:**
- Risk stratification for research purposes
- Educational tool for medical trainees
- Decision support to complement clinical judgment
- Hypothesis generation for treatment discussions

❌ **NOT Intended For:**
- Sole basis for treatment decisions
- Replacement of physician judgment
- Diagnosis or screening
- Patients with non-NSMP molecular profiles

### 4.2 Prominent Disclaimers
- Clinical disclaimer displayed prominently
- Tool clearly labeled as decision support
- Physician review explicitly recommended

### 4.3 Known Limitations
- Model has inherent prediction uncertainty
- External validation may be limited
- Does not account for all prognostic factors
- Not validated for:
  - POLE ultramutated tumors
  - MMR-deficient tumors
  - p53-abnormal tumors

---

## 5. Accountability & Governance

### 5.1 Development Oversight
- Multidisciplinary team involvement:
  - Clinical oncologists
  - Data scientists
  - Bioethicists
  - Patient advocates (recommended for future)

### 5.2 Version Control
- All model versions documented
- Changes logged with rationale
- Clear version identification in interface

### 5.3 Feedback Mechanism
- Users can report:
  - Unexpected predictions
  - Technical issues
  - Clinical concerns
- Systematic review of reported issues

### 5.4 Liability Statement
- Tool provides decision **support** only
- Clinical decisions remain physician responsibility
- Developers assume no liability for clinical outcomes
- Users accept terms by using the tool

---

## 6. Informed Use

### 6.1 User Education
- Clear documentation of model methodology
- Training materials for appropriate use
- Limitations explicitly communicated

### 6.2 Patient Communication
- If results are shared with patients:
  - Explain probabilistic nature
  - Discuss limitations
  - Emphasize this is one input among many
  - Document shared decision-making

---

## 7. Continuous Improvement

### 7.1 Performance Monitoring
- Track model performance over time
- Identify drift from training conditions
- Update as evidence evolves

### 7.2 Stakeholder Engagement
- Regular feedback from clinical users
- Patient perspective incorporation
- Alignment with updated guidelines

### 7.3 Regulatory Awareness
- Monitor evolving AI/ML regulations
- Prepare for potential regulatory requirements
- Maintain documentation for audits

---

## 8. Responsible AI Principles

| Principle | Implementation |
|-----------|----------------|
| **Transparency** | Open documentation, explainable predictions |
| **Fairness** | Bias assessment, limitation disclosure |
| **Accountability** | Clear governance, feedback mechanisms |
| **Privacy** | No data collection, client-side processing |
| **Safety** | Clinical disclaimers, decision support framing |
| **Beneficence** | Designed to improve clinical outcomes |
| **Non-maleficence** | Risk communication, limitation awareness |

---

## Contact

For ethical concerns, questions, or feedback about this tool, please contact the development team.

---

*Last Updated: December 2025*
*Version: 1.0.0-beta*

