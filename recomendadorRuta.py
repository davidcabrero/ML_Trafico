import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree
from folium import plugins

# Variables en session_state para no recalcular 
if "ruta" not in st.session_state:
    st.session_state.ruta = None
if "tiempo_estimado" not in st.session_state:
    st.session_state.tiempo_estimado = None
if "mostrar_ruta" not in st.session_state:
    st.session_state.mostrar_ruta = False  # Indica si ya se ha pulsado el botón

def load_data():
    df = pd.read_csv("trafico_asturies.csv")
    df = df[(df['currentTravelTime'] > 0) & (df['currentTravelTime'] < df['currentTravelTime'].quantile(0.99))]
    df.reset_index(drop=True, inplace=True)
    return df

def build_graph_from_df(dataframe, max_distance=0.005):
    G = nx.Graph()
    for i, row in dataframe.iterrows():
        G.add_node(row['street'], latitude=row['latitude'], longitude=row['longitude'], currentTravelTime=row['currentTravelTime'])
    coords = dataframe[['latitude', 'longitude']].values
    tree = KDTree(coords, leaf_size=2)
    indices = tree.query_radius(coords, r=max_distance)
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                time_i = dataframe.loc[i, 'currentTravelTime']
                time_j = dataframe.loc[j, 'currentTravelTime']
                weight = (time_i + time_j) / 2.0
                G.add_edge(dataframe.loc[i, 'street'], dataframe.loc[j, 'street'], weight=weight)
    return G

def train_model(df):
    variables = ["currentSpeed", "freeFlowSpeed", "currentTravelTime", "freeFlowTravelTime"]
    X = df[variables].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    df["cluster"] = kmeans.labels_
    X_sup = df[["currentSpeed", "freeFlowSpeed", "confidence", "cluster"]]
    y_sup = df["currentTravelTime"]
    X_train, X_test, y_train, y_test = train_test_split(X_sup, y_sup, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    return reg

def recomendar_ruta(origen, destino, G, df, reg):
    if origen not in G.nodes or destino not in G.nodes:
        return None, "No se encontró una ruta."
    try:
        path = nx.dijkstra_path(G, origen, destino, weight='weight')
        prediccion_total = 0
        for street in path:
            street_data = df[df['street'] == street][["currentSpeed", "freeFlowSpeed", "confidence", "cluster"]]
            if not street_data.empty:
                prediccion = reg.predict(street_data)
                prediccion_total += prediccion[0]
        pred_hours, pred_remainder = divmod(prediccion_total, 3600)
        pred_minutes, pred_seconds = divmod(pred_remainder, 60)
        return path, f"{int(pred_hours)}h {int(pred_minutes)}m {int(pred_seconds)}s"
    except nx.NetworkXNoPath:
        return None, "No hay ruta disponible."

def create_map(df, path, origen, destino):
    mapa = folium.Map(location=[df[df['street'] == origen]['latitude'].values[0], df[df['street'] == origen]['longitude'].values[0]], zoom_start=13)
    folium.Marker([df[df['street'] == origen]['latitude'].values[0], df[df['street'] == origen]['longitude'].values[0]], popup=origen, icon=folium.Icon(color='green')).add_to(mapa)
    folium.Marker([df[df['street'] == destino]['latitude'].values[0], df[df['street'] == destino]['longitude'].values[0]], popup=destino, icon=folium.Icon(color='red')).add_to(mapa)
    for i in range(len(path) - 1):
        folium.PolyLine([
            [df[df['street'] == path[i]]['latitude'].values[0], df[df['street'] == path[i]]['longitude'].values[0]],
            [df[df['street'] == path[i + 1]]['latitude'].values[0], df[df['street'] == path[i + 1]]['longitude'].values[0]]
        ], color='blue').add_to(mapa)
    mapa.add_child(plugins.Fullscreen())
    mapa.add_child(plugins.MeasureControl())
    mapa.add_child(folium.LayerControl())
    return mapa

st.title("Recomendador de Rutas en Asturias")
df = load_data()
G = build_graph_from_df(df)
model = train_model(df)

disponibles = df['street'].unique()
origen = st.selectbox("Seleccione la calle de origen", disponibles)
destino = st.selectbox("Seleccione la calle de destino", disponibles)

# Botón para recomendar ruta
if st.button("Recomendar Ruta"):
    # Se ejecuta sólo cuando se hace clic
    st.session_state.mostrar_ruta = True
    
    # Solo recalcular si no existe ruta o cambian origen/destino
    if (st.session_state.ruta is None or
        "origen_prev" not in st.session_state or
        "destino_prev" not in st.session_state or
        st.session_state.origen_prev != origen or
        st.session_state.destino_prev != destino):

        st.session_state.ruta, st.session_state.tiempo_estimado = recomendar_ruta(origen, destino, G, df, model)
        st.session_state.origen_prev = origen
        st.session_state.destino_prev = destino

# Mostrar la ruta sólo si se ha hecho clic (mostrar_ruta = True)
if st.session_state.mostrar_ruta:
    if st.session_state.ruta:
        st.success(f"Ruta encontrada: {st.session_state.ruta}")
        st.info(f"Tiempo estimado: {st.session_state.tiempo_estimado}")
        mapa = create_map(df, st.session_state.ruta, origen, destino)
        st_folium(mapa, width=700, height=500)
    else:
        st.error("No se encontró ruta")