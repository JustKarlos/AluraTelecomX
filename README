# 📊 TelecomX — Análisis de Evasión de Clientes (Churn)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C9BE8?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completado-2ECC71?style=for-the-badge)

**Identificación de los factores que impulsan la cancelación de servicios y propuesta de acciones estratégicas para reducirla.**

</div>

---

## 📋 Índice

- [Contexto del Problema](#-contexto-del-problema)
- [Dataset](#-dataset)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Uso](#-instalación-y-uso)
- [Pipeline de Análisis](#-pipeline-de-análisis)
  - [1. Carga de Datos](#1-carga-de-datos)
  - [2. Exploración Inicial](#2-exploración-inicial)
  - [3. Limpieza de Datos](#3-limpieza-de-datos)
  - [4. Transformación](#4-transformación)
  - [5. Análisis Descriptivo](#5-análisis-descriptivo)
  - [6. Visualizaciones](#6-visualizaciones)
- [Hallazgos Principales](#-hallazgos-principales)
- [Conclusiones](#-conclusiones)
- [Recomendaciones Estratégicas](#-recomendaciones-estratégicas)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 🎯 Contexto del Problema

La **evasión de clientes** (*churn*) es uno de los mayores desafíos en la industria de las telecomunicaciones. Adquirir un nuevo cliente puede costar entre **5 y 25 veces más** que retener uno existente.

**TelecomX** enfrenta una tasa de evasión del **26.5%** — más de 1.800 clientes cancelaron sus servicios en el período analizado. Este proyecto busca responder:

> 🔍 ¿Qué perfil tienen los clientes que evaden?  
> 📉 ¿Qué variables están más asociadas a la cancelación?  
> 🚀 ¿Qué acciones concretas puede tomar TelecomX para mejorar la retención?

---

## 📦 Dataset

| Atributo | Detalle |
|---|---|
| **Fuente** | API pública — JSON anidado |
| **Registros originales** | 7.267 clientes |
| **Registros tras limpieza** | 7.043 clientes |
| **Variables** | 21 originales → 22 tras transformación |
| **Variable objetivo** | `Churn` (Evasión: Sí / No) |

### Estructura del JSON

Los datos vienen en formato anidado con 4 sub-objetos por cliente:

```json
{
  "customerID": "0002-ORFBO",
  "Churn": "No",
  "customer": { "gender": "Female", "SeniorCitizen": 0, "tenure": 9, ... },
  "phone":    { "PhoneService": "Yes", "MultipleLines": "No" },
  "internet": { "InternetService": "DSL", "OnlineSecurity": "No", ... },
  "account":  { "Contract": "One year", "Charges": { "Monthly": 65.6, "Total": "593.3" } }
}
```

### Diccionario de Variables

| Variable | Descripción |
|---|---|
| `customerID` | ID único del cliente |
| `Churn` | **Variable objetivo** — si el cliente abandonó la empresa |
| `tenure` | Meses de contrato con la empresa |
| `SeniorCitizen` | Si el cliente tiene 65 años o más (0/1) |
| `Contract` | Tipo de contrato (Mes a Mes / 1 Año / 2 Años) |
| `PaymentMethod` | Método de pago |
| `Charges.Monthly` | Cargo mensual total por todos los servicios |
| `Charges.Total` | Total acumulado gastado por el cliente |
| `InternetService` | Tipo de servicio de internet contratado |

---

## 📁 Estructura del Proyecto

```
telecomx-churn/
│
├── 📓 TelecomX_Informe.ipynb       # Notebook completo con análisis e informe
├── 📄 README.md                    # Este archivo
│
├── src/
│   ├── telecomx_analisis.py        # Pipeline de limpieza y transformación (Pasos 1–6)
│   └── telecomx_analisis_eda.py    # Análisis exploratorio y visualizaciones (Pasos 7–10)
│
├── images/
│   ├── paso7_distribucion_numericas.png
│   ├── paso8_distribucion_evasion.png
│   ├── paso9_evasion_categoricas.png
│   ├── paso10_evasion_numericas.png
│   └── paso10_boxplots_numericas.png
│
└── data/
    └── TelecomX_Data.json          # Dataset fuente
```

---

## ⚙️ Instalación y Uso

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn
```

### Ejecutar el análisis completo

```bash
# Limpieza y transformación de datos
python src/telecomx_analisis.py

# Análisis exploratorio y generación de gráficos
python src/telecomx_analisis_eda.py
```

### O abrir el notebook

```bash
jupyter notebook TelecomX_Informe.ipynb
```

---

## 🔧 Pipeline de Análisis

### 1. Carga de Datos

Los datos se obtienen directamente desde la URL de la API y se aplanan con `pd.json_normalize()`:

```python
import pandas as pd, json, urllib.request

URL = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json"

with urllib.request.urlopen(URL) as response:
    raw_data = json.loads(response.read().decode())

df = pd.json_normalize(raw_data)
# → DataFrame de 7.267 filas × 21 columnas
```

### 2. Exploración Inicial

Se usó `pd.unique()` para inspeccionar tipos y valores únicos en una sola pasada (O(n)):

```python
for col in df.columns:
    nulos  = df[col].isnull().sum()
    vacios = (df[col].astype(str).str.strip() == "").sum()
    unicos = pd.unique(df[col])
    if nulos > 0 or vacios > 0:
        print(f"⚠️  {col}: {nulos} nulos | {vacios} vacíos")
    if len(unicos) <= 10:
        print(f"   Valores únicos: {unicos.tolist()}")
```

**Problemas detectados:**

| Problema | Columna | Cantidad | Acción |
|---|---|---|---|
| Strings vacíos (sin variable objetivo) | `Churn` | 224 filas | Eliminadas |
| Tipo incorrecto (`str` en lugar de `float`) | `Charges.Total` | Toda la columna | `pd.to_numeric()` |
| Valores vacíos no convertibles | `Charges.Total` | 11 filas | Imputados con mediana |
| Filas duplicadas | — | 0 | Sin acción |

### 3. Limpieza de Datos

```python
# Convertir Charges.Total a numérico e imputar nulos
df["account.Charges.Total"] = pd.to_numeric(df["account.Charges.Total"], errors="coerce")
df["account.Charges.Total"] = df["account.Charges.Total"].fillna(df["account.Charges.Total"].median())

# Eliminar filas sin variable objetivo
df = df[df["Churn"].astype(str).str.strip() != ""].reset_index(drop=True)
# → 7.043 filas, 0 nulos
```

### 4. Transformación

```python
# Variables binarias Yes/No → 1/0
mapa = {"Yes": 1, "No": 0}
for col in ["customer.Partner", "customer.Dependents", "phone.PhoneService",
            "account.PaperlessBilling", "Churn"]:
    df[col] = df[col].map(mapa)

# Nueva columna: gasto diario estimado
df["Cuentas_Diarias"] = (df["account.Charges.Monthly"] / 30).round(4)
```

### 5. Análisis Descriptivo

`DataFrame.describe()` con percentiles extendidos más métricas de forma:

| Variable | Media | Mediana | Desv. Std. | Asimetría |
|---|---|---|---|---|
| Meses de Contrato | 32.4 | 29.0 | 24.6 | 0.24 |
| Cargo Mensual ($) | 64.8 | 70.4 | 30.1 | -0.22 |
| Cargo Total ($) | 2,281.9 | 1,394.6 | 2,265.3 | **0.96** |
| Cuentas Diarias ($) | 2.16 | 2.35 | 1.00 | -0.22 |

> ⚠️ `Cargo Total` presenta asimetría positiva alta (0.96): mayoría de clientes con gasto bajo y una minoría con gasto muy elevado.

### 6. Visualizaciones

---

## 📊 Visualizaciones

### Distribución de Variables Numéricas

![Distribución Numéricas](images/paso7_distribucion_numericas.png)

---

### Distribución de la Variable Evasión

![Distribución Evasión](images/paso8_distribucion_evasion.png)

> El dataset presenta un **desbalance de clases**: 73.5% permanece vs 26.5% evade. Considerar técnicas de balanceo (SMOTE, class_weight) para modelos futuros.

---

### Tasa de Evasión por Variable Categórica

![Evasión Categóricas](images/paso9_evasion_categoricas.png)

---

### Distribución de Variables Numéricas por Grupo de Evasión

![Evasión Numéricas KDE](images/paso10_evasion_numericas.png)

![Evasión Numéricas Boxplot](images/paso10_boxplots_numericas.png)

---

## 🔍 Hallazgos Principales

### Perfil del cliente en riesgo de evasión

| Dimensión | Categoría de Mayor Riesgo | Tasa de Evasión |
|---|---|---|
| Método de Pago | Cheque Electrónico | **45.3%** |
| Tipo de Contrato | Mes a Mes | **42.7%** |
| Tipo de Internet | Fibra Óptica | **41.9%** |
| Adulto Mayor | Sí (≥65 años) | **41.7%** |
| Factura Digital | Sí | **33.6%** |
| Género | Similar en ambos | ~26-27% |

### Comparación numérica entre grupos

| Variable | 🟢 Permanece | 🔴 Evadió | Diferencia |
|---|---|---|---|
| Meses de contrato (media) | 37.6 meses | 18.0 meses | **-52%** |
| Cargo mensual (media) | $61.3 | $74.4 | **+21%** |
| Cargo total (media) | $2,553 | $1,532 | **-40%** |

---

## 💡 Conclusiones

**1. La evasión es un fenómeno temprano**
El 50% de los clientes que evaden lo hacen antes de los **10 meses** (mediana). La experiencia durante los primeros meses es crítica.

**2. Fibra Óptica: paradoja precio-valor**
Siendo el servicio más caro, concentra la mayor tasa de evasión (41.9%). Los clientes perciben que el costo no justifica la calidad recibida.

**3. El Cheque Electrónico es la mayor señal de desenganche**
Requiere una decisión activa de pago cada mes (no automático), reflejando bajo compromiso. Tasa de evasión 3× mayor que los métodos automáticos.

**4. Los contratos cortos son la puerta de salida**
La diferencia entre Mes a Mes (42.7%) y Dos Años (2.8%) es de **40 puntos porcentuales** — la variable con mayor poder predictivo.

**5. El género no diferencia**
Femenino (26.9%) y Masculino (26.2%) presentan tasas prácticamente idénticas. Las acciones de retención no deben segmentarse por género.

---

## 🚀 Recomendaciones Estratégicas

| # | Acción | Segmento | Impacto |
|---|---|---|---|
| 1 | **Programa de migración a contratos anuales** — descuentos/upgrades por compromiso | Clientes Mes a Mes | 🔴 Alto |
| 2 | **Onboarding proactivo en los primeros 6 meses** — llamadas, revisiones, ofertas | Clientes nuevos | 🔴 Alto |
| 3 | **Auditoría de calidad Fibra Óptica** — SLAs, compensaciones, paquetes premium | Clientes Fibra Óptica | 🔴 Alto |
| 4 | **Incentivo a pago automático** — descuento permanente por débito automático | Cheque Electrónico | 🟠 Alto |
| 5 | **Canal de atención diferenciado** para adultos mayores | ≥65 años | 🟡 Medio |
| 6 | **Modelo predictivo de churn** (Random Forest / XGBoost) para intervención preventiva | Todos | 🔵 Estratégico |

### Objetivo

> Con las tres primeras acciones focalizadas, es posible reducir la tasa de evasión del **26.5% actual a menos del 15%** en un horizonte de 12 meses.

---

## 🛠️ Tecnologías Utilizadas

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y limpieza de datos, `pd.json_normalize()`, `pd.unique()` |
| `numpy` | Operaciones numéricas |
| `matplotlib` | Visualizaciones base |
| `seaborn` | Gráficos estadísticos (histplot, kdeplot, boxplot) |
| `json` + `urllib` | Carga de datos desde API |

---

<div align="center">

**TelecomX Churn Analysis** — Challenge Data Science LATAM 2025

</div>
