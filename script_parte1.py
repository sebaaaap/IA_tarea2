
# 1.1 - CARGA Y PREPROCESAMIENTO
# ======================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import pickle, os, sys

CSV_NAME = "dataset_t2.csv"


if not os.path.exists(CSV_NAME):
    print("No se encontró el dataset_t2.csv")
    print("Colócalo en la misma carpeta y vuelve a ejecutar el script.")
    sys.exit()

df = pd.read_csv(CSV_NAME)
print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
print(df.head())


# 1.2 - VARIABLES X e Y
# ======================================================

Y_col = "Rating_Category"
X_cols = [c for c in df.columns if c != Y_col and df[c].dtype != 'object']

X = df[X_cols]
Y = df[Y_col]

print("Variables usadas para clustering:")
print(X_cols)
print(f"Etiqueta Y: {Y_col} (clases: {df[Y_col].unique()})")


# 1.3 - DIVISIÓN DE DATOS
# ======================================================

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} filas, Test: {len(X_test)} filas")


# 1.4 - NORMALIZACIÓN
# ======================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("clustering_outputs", exist_ok=True)
pickle.dump(scaler, open("clustering_outputs/scaler.pkl", "wb"))


# 1.5 - ENTRENAMIENTO Y EVALUACIÓN
# ======================================================

models_info = []

# Configuraciones kmeans
for k in [3, 4, 5, 6]:
    for init_method in ["random", "k-means++"]:
        model = KMeans(n_clusters=k, init=init_method, random_state=42)
        labels = model.fit_predict(X_train_scaled)
        score = silhouette_score(X_train_scaled, labels)
        models_info.append({
            "model": model,
            "name": f"KMeans_k={k}_{init_method}",
            "silhouette": score
        })
        print(f"KMeans(k={k}, init={init_method}) - Silhouette = {score:.3f}")

# Configuraciones meanshift
for quantile in [0.2, 0.3, 0.4, 0.5]:
    bandwidth = estimate_bandwidth(X_train_scaled, quantile=quantile)
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(X_train_scaled)
    score = silhouette_score(X_train_scaled, labels)
    models_info.append({
        "model": model,
        "name": f"MeanShift_q={quantile:.1f}",
        "silhouette": score
    })
    print(f"MeanShift(q={quantile:.1f}) - Silhouette = {score:.3f}")
    
# Resumen general
results_df = pd.DataFrame([
    {"Modelo": m["name"], "Silhouette": round(m["silhouette"], 4)}
    for m in models_info
]).sort_values("Silhouette", ascending=False)

print("RANKING MODELOS (Top 12)")
print(results_df)

# Guarda resultados, ver en el editor de codigo
results_df.to_csv("clustering_outputs/model_silhouette_scores.csv", index=False)


# 1.6 - APLICAR LOS 3 MEJORES MODELOS AL TEST
# ======================================================

top3 = results_df.head(3)
for i, name in enumerate(top3["Modelo"], 1):
    model = [m for m in models_info if m["name"] == name][0]["model"]
    pickle.dump(model, open(f"clustering_outputs/model_top{i}.pkl", "wb"))

    # predicción en test, si vemos la densidad predomina la clase 'baja'
    preds = model.predict(X_test_scaled)
    df_result = pd.DataFrame({
        "Cluster": preds,
        "Y_real": Y_test.values
    })

    # mapeo cluster → clase dominante
    mapping = df_result.groupby("Cluster")["Y_real"].agg(lambda x: x.mode()[0])
    df_result["Y_pred"] = df_result["Cluster"].map(mapping)

    # precisión simple
    accuracy = (df_result["Y_real"] == df_result["Y_pred"]).mean()
    print(f"Modelo {i}: {name} - Precisión mapeada = {accuracy:.3f}")

    # guardar resultados
    df_result.to_csv(f"clustering_outputs/test_labels_model_{i}.csv", index=False)

print("Proceso completado. Revisa la carpeta clustering_outputs/")