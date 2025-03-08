import sys
import pandas as pd
import numpy as np
import networkx as nx
import folium
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QPushButton, QLineEdit, QCompleter, QHBoxLayout)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from sklearn.neighbors import KDTree
from geopy.distance import geodesic
from io import BytesIO
import base64
from sklearn.cluster import KMeans

class RutasApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.df = self.load_data()
        self.G = self.build_graph_from_df(self.df)
        self.origen = None
        self.destino = None
    
    def initUI(self):
        self.setWindowTitle("Recomendador de Rutas en Asturias")
        self.setGeometry(100, 100, 900, 700)
        
        layout = QVBoxLayout()
        
        self.label_origen = QLabel("Seleccione la calle de origen")
        self.input_origen = QLineEdit()
        self.completer_origen = QCompleter()
        self.input_origen.setCompleter(self.completer_origen)
        
        self.label_destino = QLabel("Seleccione la calle de destino")
        self.input_destino = QLineEdit()
        self.completer_destino = QCompleter()
        self.input_destino.setCompleter(self.completer_destino)
        
        self.boton_ruta = QPushButton("Recomendar Ruta")
        self.boton_ruta.clicked.connect(self.calcular_ruta)
        
        self.mapa_view = QWebEngineView()
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.label_origen)
        controls_layout.addWidget(self.input_origen)
        controls_layout.addWidget(self.label_destino)
        controls_layout.addWidget(self.input_destino)
        controls_layout.addWidget(self.boton_ruta)
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.mapa_view)
        
        self.setLayout(layout)

        self.label_ruta = QLabel("")
        layout.addWidget(self.label_ruta)
    
    def load_data(self):
        df = pd.read_csv("trafico_asturies.csv")
        df = df[(df['currentTravelTime'] > 0) & (df['currentTravelTime'] < df['currentTravelTime'].quantile(0.99))]
        df.reset_index(drop=True, inplace=True)

        # Ejemplo de clustering simple con KMeans usando la velocidad actual
        X = df[['currentSpeed']].copy()
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        calles = df['street'].unique().tolist()
        self.completer_origen.setModel(self.create_model(calles))
        self.completer_destino.setModel(self.create_model(calles))
        return df
    
    def create_model(self, items):
        from PyQt6.QtCore import QStringListModel
        model = QStringListModel()
        model.setStringList(items)
        return model
    
    def build_graph_from_df(self, dataframe, max_distance=0.005):
        G = nx.Graph()
        for i, row in dataframe.iterrows():
            G.add_node(row['street'],
                       latitude=row['latitude'],
                       longitude=row['longitude'],
                       currentSpeed=row['currentSpeed'],
                       cluster=row['cluster'])

        coords = dataframe[['latitude', 'longitude']].values
        tree = KDTree(coords, leaf_size=2)
        indices = tree.query_radius(coords, r=max_distance)

        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    coord_i = (dataframe.loc[i, 'latitude'], dataframe.loc[i, 'longitude'])
                    coord_j = (dataframe.loc[j, 'latitude'], dataframe.loc[j, 'longitude'])
                    distance = geodesic(coord_i, coord_j).km
                    speed_i = dataframe.loc[i, 'currentSpeed']
                    speed_j = dataframe.loc[j, 'currentSpeed']
                    avg_speed = (speed_i + speed_j) / 2.0
                    travel_time = (distance / avg_speed) * 3600 if avg_speed > 0 else float('inf')
                    
                    # Ajuste de ejemplo: reduce el tiempo si pertenecen al mismo cluster
                    if dataframe.loc[i, 'cluster'] == dataframe.loc[j, 'cluster']:
                        travel_time *= 0.9  # Reducir el 10% si están en el mismo cluster

                    G.add_edge(dataframe.loc[i, 'street'],
                               dataframe.loc[j, 'street'],
                               weight=travel_time,
                               distance=distance)
        return G
    
    def calcular_ruta(self):
        origen = self.input_origen.text()
        destino = self.input_destino.text()
        if origen not in self.G.nodes or destino not in self.G.nodes:
            return
        
        try:
            path = nx.dijkstra_path(self.G, origen, destino, weight='weight')
            tiempo_total = sum(self.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            horas, resto = divmod(tiempo_total, 3600)
            minutos, segundos = divmod(resto, 60)
            duracion = f"{int(horas)}h {int(minutos)}m {int(segundos)}s"
            self.label_ruta.setText(f"Calles de la ruta:\n{', '.join(path)}\nDuración estimada: {duracion}")
            self.mostrar_mapa(path, origen, destino, duracion)
        except nx.NetworkXNoPath:
            pass
    
    def mostrar_mapa(self, path, origen, destino, tiempo):
        mapa = folium.Map(location=[self.df[self.df['street'] == origen]['latitude'].values[0],
                                    self.df[self.df['street'] == origen]['longitude'].values[0]], zoom_start=13)
        folium.Marker([self.df[self.df['street'] == origen]['latitude'].values[0],
                       self.df[self.df['street'] == origen]['longitude'].values[0]],
                      popup=f"Origen: {origen}", icon=folium.Icon(color='green')).add_to(mapa)
        folium.Marker([self.df[self.df['street'] == destino]['latitude'].values[0],
                       self.df[self.df['street'] == destino]['longitude'].values[0]],
                      popup=f"Destino: {destino}", icon=folium.Icon(color='red')).add_to(mapa)
        
        for i in range(len(path) - 1):
            folium.PolyLine([
                [self.df[self.df['street'] == path[i]]['latitude'].values[0], self.df[self.df['street'] == path[i]]['longitude'].values[0]],
                [self.df[self.df['street'] == path[i+1]]['latitude'].values[0], self.df[self.df['street'] == path[i+1]]['longitude'].values[0]]
            ], color='blue').add_to(mapa)
        
        data = BytesIO()
        mapa.save(data, close_file=False)
        html = data.getvalue().decode()
        self.mapa_view.setHtml(html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RutasApp()
    window.show()
    sys.exit(app.exec())
