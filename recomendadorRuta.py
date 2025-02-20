import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree
import folium
from folium import plugins

def build_graph_from_df(dataframe, max_distance=0.005):
    # Se crea un grafo vacío utilizando NetworkX
    G = nx.Graph()
    
    # Itera sobre cada fila del DataFrame
    for i, row in dataframe.iterrows():
        # Añade un nodo al grafo por cada calle, con atributos de latitud, longitud y tiempo de viaje actual
        G.add_node(
            row['street'],
            latitude=row['latitude'],
            longitude=row['longitude'],
            currentTravelTime=row['currentTravelTime']
        )
    
    # Extrae las coordenadas de latitud y longitud del DataFrame y las convierte en un array de valores
    coords = dataframe[['latitude', 'longitude']].values
    
    # Crea un KDTree (estructura de datos para búsqueda eficiente de vecinos) con las coordenadas
    tree = KDTree(coords, leaf_size=2)
    
    # Encuentra los índices de las calles que están dentro de la distancia máxima especificada
    indices = tree.query_radius(coords, r=max_distance)
    
    # Itera sobre cada calle y sus vecinos cercanos
    for i, neighbors in enumerate(indices):
        # Itera sobre cada vecino de la calle actual
        for j in neighbors:
            # Evita añadir una arista de un nodo a sí mismo
            if i != j:
                # Obtiene los tiempos de viaje actuales de la calle y su vecino
                time_i = dataframe.loc[i, 'currentTravelTime']
                time_j = dataframe.loc[j, 'currentTravelTime']
                # Calcula el peso de la arista como el promedio de los tiempos de viaje
                weight = (time_i + time_j) / 2.0
                # Añade una arista entre la calle y su vecino con el peso calculado
                G.add_edge(dataframe.loc[i, 'street'], dataframe.loc[j, 'street'], weight=weight)
    
    # Devuelve el grafo creado
    return G

def recomendar_ruta(origen, destino, G, df, reg):
    # Verifica si la calle de origen está en el grafo
    if origen not in G.nodes:
        return f"La calle de origen '{origen}' no se encuentra en el grafo."
    
    # Verifica si la calle de destino está en el grafo
    if destino not in G.nodes:
        return f"La calle de destino '{destino}' no se encuentra en el grafo."
    
    try:
        # Encuentra la ruta más rápida utilizando el algoritmo de Dijkstra
        path = nx.dijkstra_path(G, origen, destino, weight='weight')
        
        # Predice el tiempo de viaje utilizando el modelo de regresión
        prediccion_total = 0
        for street in path:
            street_data = df[df['street'] == street][["currentSpeed", "freeFlowSpeed", "confidence", "cluster"]]
            if not street_data.empty:
                prediccion = reg.predict(street_data)
                prediccion_total += prediccion[0]
        
        # Convierte la predicción total de viaje a horas, minutos y segundos
        pred_hours, pred_remainder = divmod(prediccion_total, 3600)
        pred_minutes, pred_seconds = divmod(pred_remainder, 60)
        
        # Devuelve la ruta, el tiempo total de viaje y la predicción del modelo
        return path, f"{int(pred_hours)}h {int(pred_minutes)}m {int(pred_seconds)}s"
    
    except nx.NetworkXNoPath:
        # Maneja la excepción si no hay una ruta disponible entre origen y destino
        return f"No hay ruta disponible entre '{origen}' y '{destino}'."

if __name__ == "__main__":

    df = pd.read_csv("trafico_asturies.csv")

    # Filtrar valores atípicos
    df = df[(df['currentTravelTime'] > 0) & (df['currentTravelTime'] < df['currentTravelTime'].quantile(0.99))]

    df.reset_index(drop=True, inplace=True)

    G = build_graph_from_df(df, max_distance=0.005)

    origen_ejemplo = "Calle Linares Rivas"
    destino_ejemplo = "Calle Ponga"
    
    # Variables para los clusters
    variables = ["currentSpeed", "freeFlowSpeed", "currentTravelTime", "freeFlowTravelTime"]
    X = df[variables].copy()

    # Escalar datos y aplicar clusters
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    df["cluster"] = kmeans.labels_

    X_sup = df[["currentSpeed", "freeFlowSpeed", "confidence", "cluster"]]
    y_sup = df["currentTravelTime"]
    X_train, X_test, y_train, y_test = train_test_split(X_sup, y_sup, test_size=0.2, random_state=42)

    # RandomForestRegressor para predecir tiempo de viaje
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R^2 del modelo de regresión: {score:.4f}")

    resultado = recomendar_ruta(origen_ejemplo, destino_ejemplo, G, df, reg)
    print("Ruta recomendada:", resultado[0])
    print("Tiempo estimado:", resultado[1])

    # Crear un mapa centrado en en el origen y destino
    mapa = folium.Map(
        location=[df.loc[df['street'] == origen_ejemplo, 'latitude'].values[0], df.loc[df['street'] == origen_ejemplo, 'longitude'].values[0]],
        zoom_start=13
    )

    # Añadir un marcador para el origen
    folium.Marker(
        location=[df.loc[df['street'] == origen_ejemplo, 'latitude'].values[0], df.loc[df['street'] == origen_ejemplo, 'longitude'].values[0]],
        popup=origen_ejemplo,
        icon=folium.Icon(color='green')
    ).add_to(mapa)

    # Añadir un marcador para el destino
    folium.Marker(
        location=[df.loc[df['street'] == destino_ejemplo, 'latitude'].values[0], df.loc[df['street'] == destino_ejemplo, 'longitude'].values[0]],
        popup=destino_ejemplo,
        icon=folium.Icon(color='red')
    ).add_to(mapa)

    # colorea la ruta recomendada
    for i in range(len(resultado[0]) - 1):
        street1 = resultado[0][i]
        street2 = resultado[0][i + 1]
        folium.PolyLine(
            locations=[
                [df.loc[df['street'] == street1, 'latitude'].values[0], df.loc[df['street'] == street1, 'longitude'].values[0]],
                [df.loc[df['street'] == street2, 'latitude'].values[0], df.loc[df['street'] == street2, 'longitude'].values[0]]
            ],
            color='blue'
        ).add_to(mapa)

    # Imprime en la web un cuadro con el tiempo estimado
    folium.Marker(
        location=[df.loc[df['street'] == origen_ejemplo, 'latitude'].values[0], df.loc[df['street'] == origen_ejemplo, 'longitude'].values[0]],
        popup=f"Tiempo estimado: {resultado[1]}",
        icon=folium.Icon(color='blue')
    ).add_to(mapa)

    # Añadir una capa de control de plugins
    mapa.add_child(plugins.Fullscreen())
    mapa.add_child(plugins.MeasureControl())

    # Añadir una capa de control de escala
    mapa.add_child(folium.LayerControl())

    # Guardar el mapa como un archivo HTML
    mapa.save("mapa_ruta.html")
    print("Mapa guardado como 'mapa_ruta.html'.")
