"""
=============================================================
 Aprendizaje No Supervisado - Metro de Medellín
 Actividad 4 - Métodos de Aprendizaje No Supervisado
 Materia: Inteligencia Artificial
 Autor: Keannu Ivorys Devia Bohorquez
=============================================================

Descripción:
  Modelos de aprendizaje no supervisado para descubrir
  patrones de comportamiento de viajes en el Metro de
  Medellín SIN usar etiquetas previas.

  Técnicas aplicadas:
    1. K-Means Clustering     → agrupa viajes por comportamiento
    2. DBSCAN                 → detecta clusters y anomalías
    3. PCA                    → reduce dimensionalidad y visualiza

Ejecución:
  python modelo_nosupervisado.py

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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

os.makedirs("resultados4", exist_ok=True)

SEP = "=" * 60
BLUE  = "#1F3864"
LBLUE = "#2E75B6"
COLORS = ["#1F3864","#2E75B6","#C00000","#70AD47","#ED7D31","#7030A0"]

# =============================================================
# 1. CARGA Y PREPARACIÓN DEL DATASET
# =============================================================
print(f"\n{SEP}")
print("  1. CARGA Y PREPARACIÓN DEL DATASET")
print(SEP)

df = pd.read_csv("metro_viajes.csv")
print(f"\n  Registros cargados : {len(df)}")
print(f"  Columnas           : {len(df.columns)}")

# Codificar variables categóricas
le_franja = LabelEncoder()
le_lo     = LabelEncoder()
le_ld     = LabelEncoder()
df["franja_cod"]       = le_franja.fit_transform(df["franja_horaria"])
df["linea_origen_cod"] = le_lo.fit_transform(df["linea_origen"])
df["linea_dest_cod"]   = le_ld.fit_transform(df["linea_destino"])

# Features para clustering (sin variable objetivo)
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
]

X = df[FEATURES].copy()

# Escalar features (necesario para K-Means y DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Features usadas    : {len(FEATURES)}")
print(f"  Features: {FEATURES}")
print(f"\n  Estadísticas del dataset:")
print(X.describe().round(2).to_string())

# =============================================================
# 2. MÉTODO DEL CODO - Encontrar k óptimo para K-Means
# =============================================================
print(f"\n{SEP}")
print("  2. MÉTODO DEL CODO - K óptimo para K-Means")
print(SEP)

inercias  = []
siluetas  = []
k_rango   = range(2, 11)

for k in k_rango:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_scaled, km.labels_))
    print(f"  k={k}  Inercia={km.inertia_:8.1f}  Silhouette={siluetas[-1]:.4f}")

k_optimo = k_rango[siluetas.index(max(siluetas))]
print(f"\n  ✓ k óptimo por Silhouette: {k_optimo}")

# =============================================================
# 3. K-MEANS CLUSTERING
# =============================================================
print(f"\n{SEP}")
print(f"  3. K-MEANS CLUSTERING (k={k_optimo})")
print(SEP)

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

sil_km = silhouette_score(X_scaled, df["cluster_kmeans"])
db_km  = davies_bouldin_score(X_scaled, df["cluster_kmeans"])

print(f"\n  MÉTRICAS K-MEANS:")
print(f"    Silhouette Score  : {sil_km:.4f}  (más cercano a 1 = mejor)")
print(f"    Davies-Bouldin    : {db_km:.4f}  (más cercano a 0 = mejor)")
print(f"    Inercia           : {kmeans.inertia_:.2f}")

print(f"\n  DISTRIBUCIÓN DE CLUSTERS:")
dist = df["cluster_kmeans"].value_counts().sort_index()
for c, n in dist.items():
    bar = "█" * int(n / 15)
    print(f"    Cluster {c}: {n:4d} registros  {bar}")

print(f"\n  PERFIL DE CADA CLUSTER:")
perfil = df.groupby("cluster_kmeans")[FEATURES].mean().round(3)
for c in range(k_optimo):
    print(f"\n  --- Cluster {c} ({dist[c]} viajes) ---")
    row = perfil.loc[c]
    franja_idx = int(round(row["franja_cod"]))
    franja_idx = min(franja_idx, len(le_franja.classes_)-1)
    franja_nom = le_franja.classes_[franja_idx]
    print(f"    Distancia promedio  : {row['distancia_km']:.2f} km")
    print(f"    Ocupación promedio  : {row['ocupacion_vagon']:.2f}")
    print(f"    Franja predominante : {franja_nom}")
    print(f"    Misma línea         : {'Sí' if row['misma_linea'] > 0.5 else 'No'}")
    print(f"    Requiere transbordo : {'Sí' if row['requiere_transbordo'] > 0.5 else 'No'}")
    print(f"    Tiempo espera prom. : {row['tiempo_espera_min']:.1f} min")

# Etiquetas interpretativas por cluster
etiquetas_cluster = {}
for c in range(k_optimo):
    row = perfil.loc[c]
    franja_idx = int(round(row["franja_cod"]))
    franja_idx = min(franja_idx, len(le_franja.classes_)-1)
    franja_nom = le_franja.classes_[franja_idx]
    oc = row["ocupacion_vagon"]
    ml = row["misma_linea"]
    if oc > 0.65 and "pico" in franja_nom:
        etq = "Viajes pico alta demanda"
    elif oc < 0.3:
        etq = "Viajes baja ocupación"
    elif ml > 0.6:
        etq = "Viajes directos (sin transbordo)"
    elif row["distancia_km"] > 9:
        etq = "Viajes largos multi-línea"
    else:
        etq = "Viajes intermedios"
    etiquetas_cluster[c] = etq
    print(f"\n  ✓ Cluster {c} → \"{etq}\"")

# =============================================================
# 4. DBSCAN
# =============================================================
print(f"\n{SEP}")
print("  4. DBSCAN - Detección de clusters y anomalías")
print(SEP)

# Estimar epsilon con k-NN
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distancias, _ = nbrs.kneighbors(X_scaled)
dist_sorted = np.sort(distancias[:, 4])

# Usar percentil 90 como epsilon
eps_val = float(np.percentile(dist_sorted, 90))
print(f"\n  Epsilon estimado (percentil 90 de 5-NN): {eps_val:.4f}")

dbscan = DBSCAN(eps=eps_val, min_samples=10)
df["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df["cluster_dbscan"])) - (1 if -1 in df["cluster_dbscan"].values else 0)
n_ruido       = (df["cluster_dbscan"] == -1).sum()

print(f"\n  RESULTADOS DBSCAN:")
print(f"    Clusters encontrados: {n_clusters_db}")
print(f"    Puntos de ruido     : {n_ruido} ({n_ruido/len(df)*100:.1f}%) ← posibles anomalías")

print(f"\n  DISTRIBUCIÓN DBSCAN:")
dist_db = df["cluster_dbscan"].value_counts().sort_index()
for c, n in dist_db.items():
    etq = "RUIDO/Anomalías" if c == -1 else f"Cluster {c}"
    bar = "█" * int(n / 15)
    print(f"    {etq:20}: {n:4d} registros  {bar}")

if n_clusters_db >= 2:
    mask = df["cluster_dbscan"] != -1
    sil_db = silhouette_score(X_scaled[mask], df.loc[mask, "cluster_dbscan"])
    print(f"\n  Silhouette Score (sin ruido): {sil_db:.4f}")

# Análisis de anomalías
anomalias = df[df["cluster_dbscan"] == -1]
print(f"\n  PERFIL DE ANOMALÍAS (ruido DBSCAN):")
if len(anomalias) > 0:
    print(f"    Distancia prom.  : {anomalias['distancia_km'].mean():.2f} km")
    print(f"    Ocupación prom.  : {anomalias['ocupacion_vagon'].mean():.2f}")
    print(f"    Espera prom.     : {anomalias['tiempo_espera_min'].mean():.1f} min")
    print(f"    Con transbordo   : {anomalias['requiere_transbordo'].mean()*100:.1f}%")

# =============================================================
# 5. PCA - REDUCCIÓN DE DIMENSIONALIDAD
# =============================================================
print(f"\n{SEP}")
print("  5. PCA - REDUCCIÓN DE DIMENSIONALIDAD")
print(SEP)

pca_full = PCA()
pca_full.fit(X_scaled)
varianza_acum = np.cumsum(pca_full.explained_variance_ratio_)

print(f"\n  Varianza explicada por componente:")
for i, (v, va) in enumerate(zip(pca_full.explained_variance_ratio_, varianza_acum)):
    bar = "█" * int(v * 60)
    print(f"    PC{i+1:2d}: {v:.4f} ({va:.4f} acum.)  {bar}")
    if va >= 0.95:
        print(f"    → 95% de varianza explicada con {i+1} componentes")
        break

# Reducir a 2 componentes para visualización
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

print(f"\n  Varianza explicada por PC1: {pca2.explained_variance_ratio_[0]:.4f}")
print(f"  Varianza explicada por PC2: {pca2.explained_variance_ratio_[1]:.4f}")
print(f"  Total explicado (2 comp.)  : {sum(pca2.explained_variance_ratio_):.4f}")

# Cargas de cada feature en PC1 y PC2
cargas = pd.DataFrame(
    pca2.components_.T,
    index=FEATURES,
    columns=["PC1","PC2"]
).round(4)
print(f"\n  Cargas de features en PC1 y PC2:")
print(cargas.sort_values("PC1", ascending=False).to_string())

# =============================================================
# 6. COMPARACIÓN DE MÉTODOS
# =============================================================
print(f"\n{SEP}")
print("  6. COMPARACIÓN DE MÉTODOS")
print(SEP)

print(f"\n  {'Método':<20} {'Clusters':>10} {'Silhouette':>12} {'Anomalías':>12}")
print(f"  {'-'*54}")
print(f"  {'K-Means':<20} {k_optimo:>10} {sil_km:>12.4f} {'N/A':>12}")
if n_clusters_db >= 2:
    print(f"  {'DBSCAN':<20} {n_clusters_db:>10} {sil_db:>12.4f} {n_ruido:>12}")
else:
    print(f"  {'DBSCAN':<20} {n_clusters_db:>10} {'N/A':>12} {n_ruido:>12}")

print(f"\n  Interpretación:")
print(f"  • K-Means agrupa todos los viajes en {k_optimo} grupos homogéneos")
print(f"  • DBSCAN encontró {n_clusters_db} clusters densos y {n_ruido} viajes atípicos")
print(f"  • PCA reduce las {len(FEATURES)} variables a 2 componentes visualizables")

# =============================================================
# 7. GENERACIÓN DE GRÁFICOS
# =============================================================
print(f"\n{SEP}")
print("  7. GENERANDO GRÁFICOS → carpeta resultados4/")
print(SEP)

plt.style.use("seaborn-v0_8-whitegrid")

# ── Gráfico 1: Método del codo + Silhouette ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_rango), inercias, "o-", color=BLUE, linewidth=2, markersize=7)
axes[0].axvline(k_optimo, color="#C00000", linestyle="--", label=f"k óptimo = {k_optimo}")
axes[0].set_xlabel("Número de clusters (k)", fontsize=11)
axes[0].set_ylabel("Inercia", fontsize=11)
axes[0].set_title("Método del Codo", fontsize=13, fontweight="bold", color=BLUE)
axes[0].legend()

axes[1].plot(list(k_rango), siluetas, "s-", color=LBLUE, linewidth=2, markersize=7)
axes[1].axvline(k_optimo, color="#C00000", linestyle="--", label=f"k óptimo = {k_optimo}")
axes[1].set_xlabel("Número de clusters (k)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].set_title("Silhouette por k", fontsize=13, fontweight="bold", color=BLUE)
axes[1].legend()

plt.suptitle("Selección del k óptimo para K-Means", fontsize=14, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("resultados4/01_metodo_codo.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/01_metodo_codo.png")

# ── Gráfico 2: K-Means en espacio PCA ────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(k_optimo):
    mask = df["cluster_kmeans"] == c
    ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
               c=COLORS[c % len(COLORS)], alpha=0.6, s=30,
               label=f"Cluster {c}: {etiquetas_cluster[c]}")
# Centroides
centroides_pca = pca2.transform(kmeans.cluster_centers_)
ax.scatter(centroides_pca[:,0], centroides_pca[:,1],
           c="black", marker="X", s=200, zorder=5, label="Centroides")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} varianza)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} varianza)", fontsize=11)
ax.set_title(f"K-Means (k={k_optimo}) – Espacio PCA", fontsize=13, fontweight="bold", color=BLUE)
ax.legend(fontsize=9, loc="best")
plt.tight_layout()
plt.savefig("resultados4/02_kmeans_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/02_kmeans_pca.png")

# ── Gráfico 3: DBSCAN en espacio PCA ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
labels_db = sorted(df["cluster_dbscan"].unique())
for lbl in labels_db:
    mask = df["cluster_dbscan"] == lbl
    color = "#AAAAAA" if lbl == -1 else COLORS[lbl % len(COLORS)]
    nombre = "Ruido / Anomalías" if lbl == -1 else f"Cluster {lbl}"
    alpha = 0.3 if lbl == -1 else 0.6
    ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
               c=color, alpha=alpha, s=25, label=f"{nombre} ({mask.sum()})")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} varianza)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} varianza)", fontsize=11)
ax.set_title("DBSCAN – Clusters y Anomalías", fontsize=13, fontweight="bold", color=BLUE)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("resultados4/03_dbscan_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/03_dbscan_pca.png")

# ── Gráfico 4: Perfil de clusters (heatmap) ──────────────────
perfil_norm = perfil[["distancia_km","ocupacion_vagon","tiempo_espera_min",
                       "misma_linea","requiere_transbordo","franja_cod","dia_semana"]]
perfil_norm = (perfil_norm - perfil_norm.min()) / (perfil_norm.max() - perfil_norm.min() + 1e-9)
perfil_norm.index = [f"Cluster {i}\n{etiquetas_cluster[i]}" for i in range(k_optimo)]

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(perfil_norm, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label":"Valor normalizado"})
ax.set_title("Perfil de Clusters K-Means (valores normalizados)",
             fontsize=13, fontweight="bold", color=BLUE)
ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout()
plt.savefig("resultados4/04_perfil_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/04_perfil_clusters.png")

# ── Gráfico 5: PCA varianza explicada ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
var_ratio = pca_full.explained_variance_ratio_[:8]
var_acum  = np.cumsum(var_ratio)
x = range(1, len(var_ratio)+1)
axes[0].bar(x, var_ratio*100, color=LBLUE, edgecolor="white", alpha=0.85)
axes[0].plot(x, var_acum*100, "o-", color=BLUE, linewidth=2)
axes[0].axhline(95, color="#C00000", linestyle="--", label="95% umbral")
axes[0].set_xlabel("Componente Principal", fontsize=11)
axes[0].set_ylabel("Varianza explicada (%)", fontsize=11)
axes[0].set_title("Varianza por Componente (PCA)", fontsize=12, fontweight="bold", color=BLUE)
axes[0].legend()

# Biplot simplificado: cargas de features
for i, feat in enumerate(FEATURES):
    c = pca2.components_
    axes[1].arrow(0, 0, c[0,i]*3, c[1,i]*3,
                  head_width=0.08, head_length=0.05, fc=BLUE, ec=BLUE, alpha=0.6)
    axes[1].text(c[0,i]*3.2, c[1,i]*3.2, feat, fontsize=7.5, ha="center",
                 color=BLUE, fontweight="bold")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].axvline(0, color="gray", linewidth=0.5)
axes[1].set_xlabel("PC1", fontsize=11); axes[1].set_ylabel("PC2", fontsize=11)
axes[1].set_title("Biplot – Cargas de Variables", fontsize=12, fontweight="bold", color=BLUE)

plt.suptitle("Análisis PCA", fontsize=14, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("resultados4/05_pca_analisis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/05_pca_analisis.png")

# ── Gráfico 6: Distribución de clusters por franja ───────────
fig, ax = plt.subplots(figsize=(11, 5))
ct = pd.crosstab(df["franja_horaria"], df["cluster_kmeans"])
ct.columns = [f"Cluster {c}" for c in ct.columns]
ct.plot(kind="bar", ax=ax, color=COLORS[:k_optimo], edgecolor="white", alpha=0.85, rot=30)
ax.set_title("Distribución de Clusters por Franja Horaria",
             fontsize=13, fontweight="bold", color=BLUE)
ax.set_xlabel("Franja horaria", fontsize=11)
ax.set_ylabel("Número de viajes", fontsize=11)
ax.legend(title="Cluster", fontsize=9)
plt.tight_layout()
plt.savefig("resultados4/06_clusters_franja.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ resultados4/06_clusters_franja.png")

print(f"\n{SEP}")
print("  PROCESO COMPLETADO EXITOSAMENTE")
print(f"  Gráficos guardados en: resultados4/")
print(SEP)
