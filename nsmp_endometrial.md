# Dataset Metadata
NSMP Endometrial Cancer Prognostic Dataset

## Description
This dataset contains clinico-pathological, molecular, treatment, and follow-up variables from patients with NSMP (No Specific Molecular Profile) endometrial cancer. It is intended for risk stratification, recurrence prediction, and survival analysis.

---

## 1. Patient Demographics

| Variable | Description | Type |
|---------|-------------|------|
| fecha_nacimiento | Date of birth | Date |
| edad | Age at diagnosis (years) | Numeric |
| imc | Body Mass Index | Numeric |
| asa | ASA physical status score | Categorical (I–IV) |

## 2. Diagnostic & Preoperative Variables

| Variable | Description | Type |
|---------|-------------|------|
| fecha_diagnostico_ap | Date of pathological diagnosis | Date |
| tipo_histologico_pre | Histological subtype (pre-op) | Categorical |
| grado_pre | Tumor grade (pre-op) | Ordinal |
| infiltracion_miometrial_pre | Myometrial invasion (pre-op) | Categorical |
| metastasis_distancia | Distant metastasis | Binary |
| riesgo_preiq | Preoperative risk group | Categorical |
| estadiaje_preiq | Preoperative staging | Categorical |
| tratamiento_neoadyuvante | Neoadjuvant treatment | Binary |

## 3. Surgical & Pathological Variables

| Variable | Description | Type |
|---------|-------------|------|
| tratamiento_quirurgico_primario | Primary surgery | Categorical |
| tipo_histologico_def | Final histology | Categorical |
| grado_histologico_def | Final grade | Ordinal |
| tamano_tumoral | Tumor size | Numeric |
| afectacion_linfovascular | LVSI | Binary |
| ap_centinela_pelvico | Sentinel node pathology | Categorical |
| ap_ganglios_pelvicos | Pelvic nodes pathology | Categorical |
| ap_ganglios_paraorticos | Para-aortic nodes pathology | Categorical |

## 4. Molecular & Hormonal Variables

| Variable | Description | Type |
|---------|-------------|------|
| receptores_estrogenos | Estrogen receptor status | Binary |
| receptores_progesterona | Progesterone receptor status | Binary |
| estudio_genetico | Molecular/genetic study | Categorical |

## 5. Final Staging & Risk

| Variable | Description | Type |
|---------|-------------|------|
| estadio_figo | Final FIGO stage (2018/2023) | Categorical |
| grupo_riesgo_definitivo | Final risk group | Categorical |
| tratamiento_adyuvante | Indication for adjuvant therapy | Categorical |

## 6. Follow-up & Outcomes

| Variable | Description | Type |
|---------|-------------|------|
| fecha_ultima_visita | Last follow-up date | Date |
| recidiva | Recurrence | Binary |
| numero_recidiva | Number of recurrences | Numeric |
| fecha_recidiva | Date of recurrence | Date |
| lugar_recidiva | Site of recurrence | Categorical |
| tratamiento_recidiva | Treatment for recurrence | Categorical |
| cirugia_recidiva | Surgery for recurrence | Binary |
| reseccion_macroscopica_completa | Complete resection | Binary |
| tratamiento_rt_sistemico | RT or systemic therapy | Categorical |

## 7. Survival Variables

| Variable | Description | Type |
|---------|-------------|------|
| estado_actual_paciente | Current patient status | Categorical |
| causa_muerte | Cause of death | Categorical |
| fecha_muerte | Date of death | Date |
| libre_enfermedad | Disease-free status | Binary |

## Target Variables
Primary targets include:
- grupo_riesgo_definitivo
- recidiva
- libre_enfermedad

Time-to-event targets:
- Time to recurrence
- Overall survival
