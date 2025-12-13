# Key Features for Endometrial Cancer Survival Model

## Target Variables

| Column | Description | Missing % |
|--------|-------------|-----------|
| `recidiva` | Recurrence (binary outcome) | 0% |
| `libre_enferm` | Disease-free status | 0% |
| `diferencia_dias_reci_exit` | Days to recurrence/death (survival time) | 1.2% |

---

## Demographics

| Column | Description | Missing % |
|--------|-------------|-----------|
| `edad` | Age at diagnosis (years) | 0% |
| `imc` | Body Mass Index | 4.9% |
| `asa` | ASA physical status score | 11.7% |

---

## Tumor Characteristics

| Column | Description | Missing % |
|--------|-------------|-----------|
| `tipo_histologico` | Histological type | 0% |
| `Grado` | Tumor grade (preop) | 0% |
| `histo_defin` | Final histology | 7.4% |
| `grado_histologi` | Final histological grade | 11% |
| `tamano_tumoral` | Tumor size (mm) | 19% |
| `afectacion_linf` | Lymphovascular space invasion (LVSI) | 11.7% |
| `infiltracion_mi` | Myometrial infiltration depth | 10.4% |
| `infilt_estr_cervix` | Cervical stromal invasion | 8% |

---

## Staging & Risk Groups

| Column | Description | Missing % |
|--------|-------------|-----------|
| `grupo_riesgo` | Preoperative risk group | 1.2% |
| `estadiaje_pre_i` | Preoperative staging | 1.8% |
| `estadificacion_` | Final staging | 14.7% |
| `FIGO2023` | FIGO 2023 stage | 12.9% |
| `grupo_de_riesgo_definitivo` | Final risk group | 12.3% |

---

## Molecular Markers

| Column | Description | Missing % |
|--------|-------------|-----------|
| `p53_ihq` | p53 immunohistochemistry | 5.5% |
| `p53_molecular` | p53 molecular status | 8% |
| `mut_pole` | POLE mutation | 8.6% |
| `msh2` | MSH2 protein expression | 4.3% |
| `msh6` | MSH6 protein expression | 4.3% |
| `pms2` | PMS2 protein expression | 5.5% |
| `mlh1` | MLH1 protein expression | 6.1% |
| `beta_cateninap` | Beta-catenin status | 7.4% |

---

## Treatment Variables

| Column | Description | Missing % |
|--------|-------------|-----------|
| `tto_NA` | Neoadjuvant treatment | 0% |
| `tto_1_quirugico` | Primary surgical treatment | 2.5% |
| `bqt` | Brachytherapy indication | 5.5% |
| `qt` | Chemotherapy indication | 6.1% |
| `Tributaria_a_Radioterapia` | Radiotherapy indication | 4.9% |

---

## Surgical Variables

| Column | Description | Missing % |
|--------|-------------|-----------|
| `abordajeqx` | Surgical approach | 9.8% |
| `tc_gc` | Sentinel node technique used | 15.3% |
| `AP_centinela_pelvico` | Sentinel node pathology result | 19% |

---

## Python Feature List

```python
selected_features = [
    # Demographics
    'edad', 'imc', 'asa',
    
    # Tumor characteristics
    'tipo_histologico', 'Grado', 'histo_defin', 'grado_histologi',
    'tamano_tumoral', 'afectacion_linf', 'infiltracion_mi', 'infilt_estr_cervix',
    
    # Staging & Risk
    'grupo_riesgo', 'estadiaje_pre_i', 'FIGO2023', 'grupo_de_riesgo_definitivo',
    
    # Molecular
    'p53_ihq', 'p53_molecular', 'mut_pole', 'msh2', 'msh6', 'pms2', 'mlh1', 'beta_cateninap',
    
    # Treatment
    'tto_NA', 'tto_1_quirugico', 'bqt', 'qt', 'Tributaria_a_Radioterapia',
    
    # Surgical
    'abordajeqx', 'tc_gc', 'AP_centinela_pelvico'
]

# Targets
target_event = 'recidiva'
target_time = 'diferencia_dias_reci_exit'
```

---

## Excluded Features (>50% missing or not useful)

- `valor_de_ca125` (87% missing)
- `causa_muerte`, `f_muerte`, `fecha_de_recidi` (>80% missing)
- All `loc_recidiva_*` columns (>80% missing)
- All `tec_*_avan` columns (>90% missing)
- `tabla_de_*` columns (100% missing)
- Date columns (use calculated intervals instead)
- ID columns (`codigo_participante`, `usuario_reg1`)

