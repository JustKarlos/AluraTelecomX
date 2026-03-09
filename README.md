# AluraTelecomX
# 📊 TelecomX — Informe de Análisis de Evasión de Clientes (Churn)
---
**Autor:** Análisis de Datos  
**Fecha:** 2025  
**Dataset:** TelecomX_Data.json  
**Objetivo:** Identificar los factores que impulsan la evasión de clientes y proponer acciones estratégicas para reducirla.

## 1. 🎯 Introducción
### Contexto del problema

La **evasión de clientes** (*churn*) es uno de los mayores desafíos para las empresas de telecomunicaciones. Adquirir un nuevo cliente cuesta entre **5 y 25 veces más** que retener uno existente, por lo que comprender los patrones de cancelación es una prioridad estratégica de alto impacto.

**TelecomX** enfrenta una tasa de evasión del **26.5%**, lo que representa más de 1.800 clientes que abandonaron el servicio. Este informe busca responder:

- ¿Qué perfil tienen los clientes que evaden?
- ¿Qué variables (contrato, pago, servicio) están más asociadas a la cancelación?
- ¿Qué acciones puede tomar TelecomX para mejorar la retención?

### Fuente de datos

El dataset contiene **7.267 registros** de clientes con 21 variables que incluyen datos demográficos, servicios contratados, tipo de contrato, método de pago y cargos. Fue provisto en formato JSON con estructura anidada (`customer`, `phone`, `internet`, `account`).

---
## 2. ⚙️ Configuración e Importaciones

import pandas as pd
import numpy as np
import json
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import display

sns.set_theme(style='whitegrid', font_scale=1.05)
AZUL  = '#4C9BE8'
ROJO  = '#E8634C'
print('✅ Librerías cargadas correctamente')

---
## 3. 🧹 Limpieza y Tratamiento de Datos
### 3.1 Carga desde la API

Los datos se obtienen directamente desde la URL pública del repositorio en formato JSON y se normalizan con `pd.json_normalize()` para aplanar la estructura anidada.

URL = 'https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json'

# Para entornos sin acceso a red, usar archivo local:
# with open('TelecomX_Data.json') as f: raw_data = json.load(f)

with urllib.request.urlopen(URL) as response:
    raw_data = json.loads(response.read().decode())

df = pd.json_normalize(raw_data)
print(f'✅ Registros cargados: {len(df):,}  |  Columnas: {df.shape[1]}')
df.head(3)

### 3.2 Identificación de Problemas

Se utilizó `pd.unique()` para inspeccionar los valores únicos de cada columna en una sola pasada (más eficiente que `.value_counts()`), permitiendo detectar en simultáneo: tipos de datos incorrectos, strings vacíos y categorías inconsistentes.

| Problema detectado | Descripción |
|---|---|
| `Churn` con valores vacíos | 224 filas sin variable objetivo |
| `account.Charges.Total` tipo `str` | Debería ser numérico (`float64`) |
| `account.Charges.Total` con vacíos | 11 filas con cadena vacía |
| Sin duplicados | ✅ 0 filas duplicadas |

print('🔍 Detección de nulos y vacíos:')
for col in df.columns:
    nulos  = df[col].isnull().sum()
    vacios = (df[col].astype(str).str.strip() == '').sum()
    if nulos > 0 or vacios > 0:
        print(f'   ⚠️  {col:<35} nulos={nulos}  vacíos={vacios}')

print(f'\n🔍 Duplicados: {df.duplicated().sum()}')

print('\n🔍 Tipos de datos y valores únicos:')
for col in df.columns:
    unicos = pd.unique(df[col])
    alerta = ' ⚠️ debería ser numérico' if col == 'account.Charges.Total' else ''
    if len(unicos) <= 10:
        print(f'   {col:<35} {str(df[col].dtype):<10} → {unicos.tolist()}{alerta}')
    else:
        print(f'   {col:<35} {str(df[col].dtype):<10} → {len(unicos)} únicos{alerta}')

### 3.3 Correcciones Aplicadas

1. **`Charges.Total` → numérico**: conversión con `pd.to_numeric(..., errors='coerce')` y los 11 NaN generados imputados con la **mediana** (estrategia robusta ante outliers).
2. **Filas sin `Churn`**: eliminadas 224 filas sin variable objetivo (no aportan al análisis supervisado).
3. **Espacios en blanco**: estandarizados en todas las columnas de texto con `.str.strip()`.
4. **Sin duplicados**: confirmado, no requirió acción.

df_clean = df.copy()

# Charges.Total a numérico
df_clean['account.Charges.Total'] = pd.to_numeric(df_clean['account.Charges.Total'], errors='coerce')
mediana = df_clean['account.Charges.Total'].median()
df_clean['account.Charges.Total'] = df_clean['account.Charges.Total'].fillna(mediana)
print(f'✅ Charges.Total: 11 nulos imputados con mediana = {mediana:.2f}')

# Eliminar filas sin Churn
df_clean = df_clean[df_clean['Churn'].astype(str).str.strip() != ''].reset_index(drop=True)
print(f'✅ Filas con Churn vacío eliminadas. Registros restantes: {len(df_clean):,}')

# Espacios en blanco
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()
print('✅ Espacios estandarizados')

### 3.4 Transformación y Estandarización

- **Variables binarias** (`Yes/No`) → `1/0` para facilitar el análisis cuantitativo.
- **Columnas renombradas** al español para mejorar la legibilidad del informe.
- **Valores categóricos** traducidos al español.
- **Nueva columna `Cuentas_Diarias`**: `Cargo_Mensual / 30`, que permite analizar el gasto diario del cliente.

# Binarias Yes/No → 1/0
mapa_bin = {'Yes': 1, 'No': 0}
for col in ['customer.Partner','customer.Dependents','phone.PhoneService','account.PaperlessBilling']:
    df_clean[col] = df_clean[col].map(mapa_bin)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

# Cuentas_Diarias
df_clean['Cuentas_Diarias'] = (df_clean['account.Charges.Monthly'] / 30).round(4)

# Renombrar
df_clean.rename(columns={
    'customerID':'ID_Cliente','Churn':'Evasion','customer.gender':'Genero',
    'customer.SeniorCitizen':'Adulto_Mayor','customer.Partner':'Tiene_Pareja',
    'customer.Dependents':'Tiene_Dependientes','customer.tenure':'Meses_Contrato',
    'phone.PhoneService':'Servicio_Telefono','phone.MultipleLines':'Multiples_Lineas',
    'internet.InternetService':'Tipo_Internet','internet.OnlineSecurity':'Seguridad_Online',
    'internet.OnlineBackup':'Respaldo_Online','internet.DeviceProtection':'Proteccion_Dispositivo',
    'internet.TechSupport':'Soporte_Tecnico','internet.StreamingTV':'Streaming_TV',
    'internet.StreamingMovies':'Streaming_Peliculas','account.Contract':'Tipo_Contrato',
    'account.PaperlessBilling':'Factura_Digital','account.PaymentMethod':'Metodo_Pago',
    'account.Charges.Monthly':'Cargo_Mensual','account.Charges.Total':'Cargo_Total',
}, inplace=True)

# Traducir categorías
for col, mapa in {
    'Genero':       {'Female':'Femenino','Male':'Masculino'},
    'Tipo_Internet':{'Fiber optic':'Fibra Óptica','No':'Sin Internet'},
    'Tipo_Contrato':{'Month-to-month':'Mes a Mes','One year':'Un Año','Two year':'Dos Años'},
    'Metodo_Pago':  {'Electronic check':'Cheque Electrónico','Mailed check':'Cheque Correo',
                     'Bank transfer (automatic)':'Transferencia Bancaria',
                     'Credit card (automatic)':'Tarjeta de Crédito'},
}.items():
    df_clean[col] = df_clean[col].map(mapa).fillna(df_clean[col])

df_clean['Evasion_Label'] = df_clean['Evasion'].map({1:'Si', 0:'No'})
print(f'✅ Dataset final: {df_clean.shape[0]:,} filas × {df_clean.shape[1]} columnas')
df_clean.head(3)

---
## 4. 🔍 Análisis Exploratorio de Datos

### 4.1 Estadísticas Descriptivas

`DataFrame.describe()` con percentiles extendidos permite ver no solo el centro sino la forma de la distribución. Se complementa con asimetría y curtosis para detectar colas y distribuciones atípicas.

COLS_NUM = ['Meses_Contrato', 'Cargo_Mensual', 'Cargo_Total', 'Cuentas_Diarias']

desc = df_clean[COLS_NUM].describe(percentiles=[.25,.50,.75,.90])
display(desc.round(2))

print('\n📊 Asimetría y Curtosis:')
for col in COLS_NUM:
    print(f'   {col:<20}  asimetría={df_clean[col].skew():.2f}   curtosis={df_clean[col].kurt():.2f}')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribución de Variables Numéricas', fontsize=15, fontweight='bold')

for ax, col in zip(axes.flat, COLS_NUM):
    media, mediana = df_clean[col].mean(), df_clean[col].median()
    sns.histplot(df_clean[col], ax=ax, color=AZUL, bins=35, kde=True,
                 edgecolor='white', linewidth=0.4)
    ax.axvline(media,   color=ROJO,    linestyle='--', lw=1.5, label=f'Media {media:.1f}')
    ax.axvline(mediana, color='#2e7d32', linestyle=':', lw=1.5, label=f'Mediana {mediana:.1f}')
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

### 4.2 Distribución de la Variable Evasión

> **Hallazgo clave:** El dataset presenta un **desbalance de clases** (73.5% vs 26.5%). Esto debe considerarse en modelos predictivos futuros (usar técnicas como SMOTE o pesos de clase).

conteo = df_clean['Evasion_Label'].value_counts()
pct    = df_clean['Evasion_Label'].value_counts(normalize=True).mul(100)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Distribución de Evasión de Clientes', fontsize=14, fontweight='bold')

colores_bar = [AZUL if l == 'No' else ROJO for l in conteo.index]
bars = axes[0].bar(conteo.index, conteo.values, color=colores_bar,
                   edgecolor='white', linewidth=0.8, width=0.5)
for bar, val, p in zip(bars, conteo.values, pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                 f'{val:,}\n({p:.1f}%)', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Cantidad de Clientes', fontweight='bold')
axes[0].set_ylabel('N° de Clientes')
axes[0].set_ylim(0, conteo.max() * 1.2)

axes[1].pie(conteo.values, labels=conteo.index, colors=[AZUL, ROJO],
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
            textprops=dict(fontsize=11))
axes[1].set_title('Proporción de Evasión', fontweight='bold')

plt.tight_layout()
plt.show()

### 4.3 Evasión por Variables Categóricas

Se calcula la **tasa de evasión** (% de clientes que cancelaron) dentro de cada categoría. Las barras en **rojo** indican tasas iguales o superiores al 30%, consideradas de alto riesgo.
COLS_CAT = [
    ('Genero',         'Género'),
    ('Tipo_Contrato',  'Tipo de Contrato'),
    ('Metodo_Pago',    'Método de Pago'),
    ('Tipo_Internet',  'Tipo de Internet'),
    ('Adulto_Mayor',   'Adulto Mayor'),
    ('Factura_Digital','Factura Digital'),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Tasa de Evasión por Variable Categórica', fontsize=15, fontweight='bold', y=1.01)

for ax, (col, titulo) in zip(axes.flat, COLS_CAT):
    tasa = (df_clean.groupby(col)['Evasion']
              .mean().mul(100).sort_values(ascending=False)
              .reset_index())
    tasa.columns = [col, 'Tasa']
    barras = ax.barh(tasa[col].astype(str), tasa['Tasa'],
                     color=[ROJO if v >= 30 else AZUL for v in tasa['Tasa']],
                     edgecolor='white', linewidth=0.6)
    for bar, val in zip(barras, tasa['Tasa']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax.set_title(titulo, fontweight='bold', pad=8)
    ax.set_xlabel('Tasa de Evasión (%)')
    ax.set_xlim(0, tasa['Tasa'].max() * 1.25)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

plt.tight_layout()
plt.show()

### 4.4 Evasión por Variables Numéricas

Los gráficos KDE permiten comparar la **forma completa de la distribución** entre clientes que evadieron y los que no, revelando diferencias en el comportamiento de gasto y permanencia.

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Distribución de Variables Numéricas según Evasión',
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flat, COLS_NUM):
    for label, color in [('No', AZUL), ('Si', ROJO)]:
        subset = df_clean[df_clean['Evasion_Label'] == label][col]
        sns.kdeplot(subset, ax=ax, color=color, fill=True,
                    alpha=0.35, linewidth=1.8, label=label)
        ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(title='Evasión', fontsize=9)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Boxplots: Variables Numéricas por Evasión', fontsize=14, fontweight='bold')

for ax, col in zip(axes, COLS_NUM):
    sns.boxplot(data=df_clean, x='Evasion_Label', y=col, ax=ax,
                order=['No','Si'], hue='Evasion_Label',
                palette={'No': AZUL, 'Si': ROJO}, legend=False,
                width=0.5, linewidth=1.2,
                flierprops=dict(marker='o', markersize=2, alpha=0.4))
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Evasión')
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

resumen = df_clean.groupby('Evasion_Label')[COLS_NUM].agg(['mean','median','std']).round(2)
display(resumen)


---
## 5. 💡 Conclusiones e Insights

### 5.1 Perfil del Cliente que Evade

A partir del análisis exploratorio se construye el siguiente perfil de riesgo:

| Dimensión | Cliente de alto riesgo | Tasa de evasión |
|---|---|---|
| **Contrato** | Mes a Mes | 42.7% |
| **Pago** | Cheque Electrónico | 45.3% |
| **Internet** | Fibra Óptica | 41.9% |
| **Demografía** | Adulto Mayor (≥65 años) | 41.7% |
| **Facturación** | Factura Digital | 33.6% |
| **Género** | Similar entre géneros | ~26-27% |

### 5.2 Insights Numéricos Clave

| Variable | Clientes que SE QUEDAN | Clientes que EVADEN | Diferencia |
|---|---|---|---|
| **Meses de contrato (media)** | 37.6 meses | 18.0 meses | **-52%** |
| **Cargo mensual (media)** | \$61.3 | \$74.4 | **+21%** |
| **Cargo total (media)** | \$2.553 | \$1.532 | **-40%** |

### 5.3 Interpretación

1. **La evasión es un fenómeno temprano**: los clientes que cancelan llevan en promedio solo 18 meses (vs 38 de quienes permanecen). La experiencia en los primeros meses es crítica.

2. **La Fibra Óptica tiene un problema de precio-valor**: siendo el servicio más caro, presenta la mayor tasa de evasión. Los clientes perciben que no justifica el costo.

3. **El Cheque Electrónico como señal de desenganche**: este método no es automático, lo que implica una decisión activa de pago cada mes y menor compromiso con el servicio.

4. **Los contratos cortos son la puerta de salida**: los contratos mes a mes facilitan la cancelación sin penalidades, concentrando el 42.7% de la evasión.

5. **El dataset está desbalanceado (73.5% vs 26.5%)**: para modelos predictivos futuros se deberán aplicar técnicas de balanceo como SMOTE o ajuste de pesos.

---
## 6. 🚀 Recomendaciones Estratégicas

### 🎯 Acciones de Retención por Segmento

#### 1. Migrar contratos mes a mes a contratos anuales
> El mayor diferencial de evasión está en el tipo de contrato (42.7% → 11.3% → 2.8%). Ofrecer **descuentos o beneficios exclusivos** (meses gratis, upgrades de servicio) a cambio de comprometerse a contratos anuales o bianuales puede reducir drásticamente la evasión.

#### 2. Programa de bienvenida en los primeros 6 meses
> La evasión se concentra en clientes nuevos (media: 18 meses). Implementar un **programa de onboarding proactivo**: llamadas de seguimiento, tutoriales, soporte prioritario y ofertas de fidelización en los primeros 6 meses.

#### 3. Revisar la propuesta de valor de Fibra Óptica
> Con 41.9% de evasión, los clientes de Fibra Óptica pagan más pero no perciben suficiente valor. Se recomienda auditar la **calidad del servicio**, tiempos de resolución de incidencias y considerar paquetes diferenciados que justifiquen el costo premium.

#### 4. Incentivar la migración a pagos automáticos
> El Cheque Electrónico tiene 45.3% de evasión, mientras que Transferencia Bancaria y Tarjeta de Crédito (automáticos) rondan el 16%. Ofrecer **descuentos por débito automático** reduce la fricción de pago y aumenta la retención.

#### 5. Atención diferenciada para adultos mayores
> Con 41.7% de evasión, este segmento necesita **soporte técnico simplificado**, canales de atención telefónica prioritaria y planes adaptados a su uso real del servicio.

#### 6. Modelo predictivo de riesgo de evasión
> Como próximo paso, construir un modelo de clasificación (Random Forest, XGBoost) que identifique clientes en riesgo **antes de que cancelen**, permitiendo intervenciones preventivas personalizadas. Los factores identificados en este análisis son los predictores más relevantes.

---

### 📌 Resumen Ejecutivo

> TelecomX pierde **1 de cada 4 clientes**. Los tres palancas de mayor impacto para reducir esta tasa son: (1) **migrar contratos a modalidades anuales**, (2) **mejorar la experiencia en los primeros meses de contrato**, y (3) **revisar la relación precio-calidad de la Fibra Óptica**. Con estas acciones focalizadas es posible reducir la tasa de evasión de 26.5% a menos del 15% en un horizonte de 12 meses.

