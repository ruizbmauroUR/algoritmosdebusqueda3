import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def load_data() -> pd.DataFrame:
    uploaded = st.sidebar.file_uploader("Sube un CSV (se usarán 2 columnas numéricas)", type=["csv"]) 
    if uploaded is None:
        st.info("Sube un archivo CSV para continuar.")
        st.stop()

    df_raw = pd.read_csv(uploaded)

    # Tomar dos columnas numéricas; si los nombres son 'ingresos'/'puntuacion' o similares, también funcionan
    # y se renombrarán internamente para reutilizar la lógica del script original
    # Normalizar encabezados para eliminar espacios accidentales
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        use_cols = numeric_cols[:2]
    else:
        st.error("El CSV debe contener al menos dos columnas numéricas.")
        st.stop()

    df_raw = df_raw[use_cols].copy()
    df_raw.columns = ["Saldo", "Transacciones"]
    scaler = MinMaxScaler().fit(df_raw.values)
    df_scaled = pd.DataFrame(
        scaler.transform(df_raw.values), columns=["Saldo", "Transacciones"]
    )
    return df_scaled


def plot_clusters(df: pd.DataFrame, k: int, random_state: int, n_init: int):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(df.values)
    df_plot = df.copy()
    df_plot["cluster"] = kmeans.labels_

    colors = ["red", "blue", "orange", "black", "purple", "pink", "brown"]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=100, constrained_layout=True)

    for cluster in range(kmeans.n_clusters):
        ax.scatter(
            df_plot[df_plot["cluster"] == cluster]["Saldo"],
            df_plot[df_plot["cluster"] == cluster]["Transacciones"],
            marker="o",
            s=120,
            color=colors[cluster % len(colors)],
            alpha=0.6,
        )
        ax.scatter(
            kmeans.cluster_centers_[cluster][0],
            kmeans.cluster_centers_[cluster][1],
            marker="P",
            s=220,
            color=colors[cluster % len(colors)],
        )

    ax.set_title("Clientes", fontsize=22)
    ax.set_xlabel("Saldo en cuenta de ahorros", fontsize=14)
    ax.set_ylabel("Veces que usó tarjeta de crédito", fontsize=14)

    # Textos fuera del área de datos
    ax.text(1.02, 0.25, f"k = {kmeans.n_clusters}", fontsize=14, transform=ax.transAxes, ha="left", va="center")
    ax.text(1.02, 0.12, f"Inercia = {kmeans.inertia_:.2f}", fontsize=14, transform=ax.transAxes, ha="left", va="center")

    # Rango y ticks para ver posibles cortes
    ax.margins(x=0.02, y=0.02)
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(np.arange(-0.1, 1.2, 0.1))

    st.pyplot(fig)


def plot_elbow(df: pd.DataFrame, random_state: int, n_init: int):
    ks = list(range(2, 10))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(df.values)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=100, constrained_layout=True)
    ax.scatter(ks, inertias, marker="o", s=120, color="purple")
    ax.set_xlabel("Número de clusters", fontsize=14)
    ax.set_ylabel("Inercia", fontsize=14)
    ax.set_title("Método del Codo para determinar k óptimo", fontsize=18)
    ax.margins(x=0.05, y=0.1)
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Clientes - K-Means", layout="wide")
    st.title("Clientes - Agrupamiento K-Means")

    with st.sidebar:
        st.header("Parámetros")
        k = st.number_input("Número de clusters (k)", min_value=2, max_value=7, value=3, step=1)
        # random_state y n_init quedan fijos internamente

    df = load_data()
    random_state = 42
    n_init = 10

    st.subheader("Distribución de clientes")
    plot_clusters(df, k=k, random_state=random_state, n_init=n_init)

    st.subheader("Método del codo")
    plot_elbow(df, random_state=random_state, n_init=n_init)


if __name__ == "__main__":
    main()


