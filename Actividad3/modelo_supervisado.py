"""
=============================================================
 Aprendizaje Supervisado - Metro de Medellín
 Actividad 3 - Métodos de Aprendizaje Supervisado
 Materia: Inteligencia Artificial
 Autor: Keannu Ivorys Devia Bohorquez
=============================================================

Descripción:
  Modelos de aprendizaje supervisado para predecir si un
  viaje en el Metro de Medellín requiere transbordo.
  Modelos: Árbol de Decisión + Random Forest

Ejecución:
  python modelo_supervisado.py

Dependencias:
  pip install scikit-learn pandas matplotlib seaborn
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# Carpeta de salida para gráficos
os.makedirs("resultados", exist_ok=True)

SEP = "=" * 60

# =============================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================
print(f"\n{SEP}")
print("  1. CARGA Y EXPLORACIÓN DEL DATASET")
print(SEP)

df = pd.read_csv("metro_viajes.csv")
print(f"\n  Registros totales : {len(df)}")
print(f"  Columnas          : {len(df.columns)}")
print(f"\n  Primeras filas:")
print(df.head(3).to_string())
print(f"\n  Estadísticas descriptivas (numéricas):")
print(df.describe().round(2).to_string())
print(f"\n  Distribución de la variable objetivo:")
print(df["requiere_transbordo"].value_counts().to_string())
print(f"\n  Valores nulos por columna:")
print(df.isnull().sum().to_string())

# =============================================================
# 2. PREPROCESAMIENTO
# =============================================================
print(f"\n{SEP}")
print("  2. PREPROCESAMIENTO")
print(SEP)

# Codificar variables categóricas
le_franja = LabelEncoder()
le_lo     = LabelEncoder()
le_ld     = LabelEncoder()

df["franja_cod"]       = le_franja.fit_transform(df["franja_horaria"])
df["linea_origen_cod"] = le_lo.fit_transform(df["linea_origen"])
df["linea_dest_cod"]   = le_ld.fit_transform(df["linea_destino"])

# Features seleccionadas
FEATURES = [
    "distancia_km",
    "misma_linea",
    "origen_es_transbordo",
    "destino_es_transbordo",
    "franja_cod",
    "dia_semana",
    "es_festivo",
    "ocupacion_vagon",
    "tiempo_espera_min",
    "num_paradas_estimadas",
    "requiere_transbordo",
    "linea_origen_cod",
    "linea_dest_cod",
]

TARGET = "congestion_alta"

X = df[FEATURES]
y = df[TARGET]

# División entrenamiento / prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Features usadas   : {len(FEATURES)}")
print(f"  Datos entrenamiento: {len(X_train)} registros ({len(X_train)/len(df)*100:.0f}%)")
print(f"  Datos prueba       : {len(X_test)} registros ({len(X_test)/len(df)*100:.0f}%)")
print(f"\n  Codificación de franja horaria:")
for i, clase in enumerate(le_franja.classes_):
    print(f"    {i} → {clase}")

# =============================================================
# 3. MODELO 1 - ÁRBOL DE DECISIÓN
# =============================================================
print(f"\n{SEP}")
print("  3. MODELO 1 - ÁRBOL DE DECISIÓN")
print(SEP)

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

acc_dt  = accuracy_score(y_test, y_pred_dt)
auc_dt  = roc_auc_score(y_test, dt.predict_proba(X_test)[:,1])
cv_dt   = cross_val_score(dt, X, y, cv=5, scoring="accuracy").mean()

print(f"\n  Parámetros del modelo:")
print(f"    max_depth        : 5")
print(f"    min_samples_split: 10")
print(f"    min_samples_leaf : 5")
print(f"\n  RESULTADOS ÁRBOL DE DECISIÓN:")
print(f"    Exactitud (test) : {acc_dt:.4f} ({acc_dt*100:.2f}%)")
print(f"    AUC-ROC          : {auc_dt:.4f}")
print(f"    Exactitud CV (5) : {cv_dt:.4f} ({cv_dt*100:.2f}%)")
print(f"\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred_dt,
      target_names=["Sin congestión","Con congestión"]))

print(f"\n  Reglas del árbol (primeros 3 niveles):")
print(export_text(dt, feature_names=FEATURES, max_depth=3))

# Importancia de features - Árbol
importancias_dt = pd.Series(dt.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Importancia de características (Árbol):")
for feat, val in importancias_dt.items():
    bar = "█" * int(val * 40)
    print(f"    {feat:<30} {val:.4f} {bar}")

# =============================================================
# 4. MODELO 2 - RANDOM FOREST
# =============================================================
print(f"\n{SEP}")
print("  4. MODELO 2 - RANDOM FOREST")
print(SEP)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf  = accuracy_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
cv_rf   = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()

print(f"\n  Parámetros del modelo:")
print(f"    n_estimators     : 100 árboles")
print(f"    max_depth        : 8")
print(f"    min_samples_split: 5")
print(f"    min_samples_leaf : 2")
print(f"\n  RESULTADOS RANDOM FOREST:")
print(f"    Exactitud (test) : {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"    AUC-ROC          : {auc_rf:.4f}")
print(f"    Exactitud CV (5) : {cv_rf:.4f} ({cv_rf*100:.2f}%)")
print(f"\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred_rf,
      target_names=["Sin congestión","Con congestión"]))

# Importancia de features - RF
importancias_rf = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Importancia de características (Random Forest):")
for feat, val in importancias_rf.items():
    bar = "█" * int(val * 40)
    print(f"    {feat:<30} {val:.4f} {bar}")

# =============================================================
# 5. COMPARACIÓN DE MODELOS
# =============================================================
print(f"\n{SEP}")
print("  5. COMPARACIÓN DE MODELOS")
print(SEP)

print(f"\n  {'Métrica':<25} {'Árbol Decisión':>16} {'Random Forest':>16}")
print(f"  {'-'*57}")
print(f"  {'Exactitud (test)':<25} {acc_dt:>16.4f} {acc_rf:>16.4f}")
print(f"  {'AUC-ROC':<25} {auc_dt:>16.4f} {auc_rf:>16.4f}")
print(f"  {'Exactitud CV-5':<25} {cv_dt:>16.4f} {cv_rf:>16.4f}")
print(f"  {'N° parámetros':<25} {'1 árbol, prof=5':>16} {'100 árboles, p=8':>16}")

mejor = "Random Forest" if acc_rf > acc_dt else "Árbol de Decisión"
print(f"\n  ✓ Mejor modelo: {mejor}")

# =============================================================
# 6. PREDICCIÓN DE EJEMPLO
# =============================================================
print(f"\n{SEP}")
print("  6. PREDICCIONES DE EJEMPLO")
print(SEP)

ejemplos = [
    # [dist_km, misma_linea, orig_transb, dest_transb, franja_cod,
    #  dia_sem, festivo, ocupacion, espera, paradas, lin_orig, lin_dest, req_transbordo]
    [3.2, 1, 0, 0, 1, 2, 0, 0.85, 3, 4, 0, 0, 0],   # Misma línea, hora pico
    [8.7, 0, 1, 1, 1, 1, 0, 0.90, 4, 9, 0, 2, 1],   # Diferente línea, transbordo
    [1.1, 1, 0, 0, 0, 6, 1, 0.12, 10, 2, 0, 0, 0],  # Misma línea, madrugada festivo
    [12.5, 0, 0, 0, 2, 3, 0, 0.65, 6, 15, 1, 4, 1], # Larga, diferente línea
]

descripciones = [
    "Viaje corto, misma línea, hora pico",
    "Viaje con transbordo, hora pico laboral",
    "Viaje corto, madrugada festivo",
    "Viaje largo, diferente línea, mañana valle",
]

cols = FEATURES
for i, (ej, desc) in enumerate(zip(ejemplos, descripciones), 1):
    xej = pd.DataFrame([ej], columns=cols)
    pred_dt = dt.predict(xej)[0]
    pred_rf = rf.predict(xej)[0]
    prob_dt = dt.predict_proba(xej)[0][1]
    prob_rf = rf.predict_proba(xej)[0][1]
    etq = lambda p: "CON congestión" if p == 1 else "SIN congestión"
    print(f"\n  Ejemplo {i}: {desc}")
    print(f"    Árbol Dec. → {etq(pred_dt)} (prob: {prob_dt:.2f})")
    print(f"    Random F.  → {etq(pred_rf)} (prob: {prob_rf:.2f})")

# =============================================================
# 7. GENERACIÓN DE GRÁFICOS
# =============================================================
print(f"\n{SEP}")
print("  7. GENERANDO GRÁFICOS → carpeta resultados/")
print(SEP)

plt.style.use("seaborn-v0_8-whitegrid")
BLUE = "#1F3864"
LBLUE = "#2E75B6"

# ── Gráfico 1: Matriz de confusión (ambos modelos) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, titulo in zip(
    axes,
    [y_pred_dt, y_pred_rf],
    ["Árbol de Decisión", "Random Forest"]
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sin TB","Con TB"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusión\n{titulo}", fontsize=13, fontweight="bold", color=BLUE)
plt.suptitle("Comparación de Matrices de Confusión", fontsize=14, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("resultados/01_matrices_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados/01_matrices_confusion.png")

# ── Gráfico 2: Importancia de características ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, importancias, titulo, color in zip(
    axes,
    [importancias_dt, importancias_rf],
    ["Árbol de Decisión", "Random Forest"],
    [BLUE, LBLUE]
):
    importancias.plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_title(f"Importancia de Características\n{titulo}", fontsize=12, fontweight="bold", color=BLUE)
    ax.set_xlabel("Importancia")
    ax.invert_yaxis()
    for bar, val in zip(ax.patches, importancias.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
plt.suptitle("Importancia de Variables por Modelo", fontsize=13, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("resultados/02_importancia_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados/02_importancia_features.png")

# ── Gráfico 3: Árbol de decisión (visualización) ─────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=FEATURES, class_names=["Sin cong","Con cong"],
          filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3)
ax.set_title("Árbol de Decisión (primeros 3 niveles)", fontsize=14, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("resultados/03_arbol_decision.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ resultados/03_arbol_decision.png")

# ── Gráfico 4: Comparación de métricas ──────────────────────────────────────
metricas = ["Exactitud\n(test)", "AUC-ROC", "Exactitud\nCV-5"]
vals_dt  = [acc_dt, auc_dt, cv_dt]
vals_rf  = [acc_rf, auc_rf, cv_rf]
x = np.arange(len(metricas)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - w/2, vals_dt, w, label="Árbol de Decisión", color=BLUE, alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, vals_rf, w, label="Random Forest", color=LBLUE, alpha=0.85, edgecolor="white")
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylim(0, 1.12); ax.set_ylabel("Valor", fontsize=11)
ax.set_title("Comparación de Métricas entre Modelos", fontsize=13, fontweight="bold", color=BLUE)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("resultados/04_comparacion_metricas.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados/04_comparacion_metricas.png")

# ── Gráfico 5: Distribución de variables clave ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["requiere_transbordo"].value_counts().plot(
    kind="bar", ax=axes[0], color=[BLUE, LBLUE], edgecolor="white", rot=0)
axes[0].set_title("Distribución Clase Objetivo", fontweight="bold", color=BLUE)
axes[0].set_xticklabels(["Sin transbordo","Con transbordo"], rotation=15)
axes[0].set_ylabel("Cantidad")

df.groupby("franja_horaria")["requiere_transbordo"].mean().plot(
    kind="bar", ax=axes[1], color=LBLUE, edgecolor="white", rot=30)
axes[1].set_title("Tasa de Transbordo por Franja", fontweight="bold", color=BLUE)
axes[1].set_ylabel("Proporción")
axes[1].set_xlabel("")

df.boxplot(column="distancia_km", by="requiere_transbordo", ax=axes[2],
           patch_artist=True,
           boxprops=dict(facecolor=LBLUE, color=BLUE),
           medianprops=dict(color="white", linewidth=2))
axes[2].set_title("Distancia por Clase", fontweight="bold", color=BLUE)
axes[2].set_xlabel("Requiere Transbordo")
axes[2].set_ylabel("Distancia (km)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("resultados/05_exploracion_datos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados/05_exploracion_datos.png")

print(f"\n{SEP}")
print("  PROCESO COMPLETADO EXITOSAMENTE")
print(f"  Gráficos guardados en: resultados/")
print(SEP)
