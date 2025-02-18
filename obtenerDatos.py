import csv
import math
import requests
import time

def get_gijon_streets():
    """
    Obtiene todas las calles de Gijón desde OpenStreetMap utilizando Overpass API.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    (
      way["highway"](43.30,-6.00,43.65,-5.40);
    );
    /*added by auto repair*/
    (._;>;);
    /*end of auto repair*/
    out body;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})
    print("Llamando a Overpass API...")
    print("URL:", overpass_url)
    print("Consulta:", overpass_query) 
    if response.status_code == 200:
        data = response.json()
        streets = []
        for element in data["elements"]:
            if "tags" in element and "name" in element["tags"]:
                streets.append(element["tags"]["name"])
        return list(set(streets))  # Eliminar duplicados
    else:
        print("Error obteniendo calles de OSM")
        return []

def get_traffic_data(api_key, street_name):
    """
    Obtiene datos de tráfico para una calle específica usando TomTom API.
    """
    print(f"Buscando tráfico para: {street_name}")  # Agregar este print
    url = ("https://api.tomtom.com/search/2/geocode/"
           f"{street_name}, Gijón.json?key={api_key}")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "results" in data and data["results"]:
                lat = data["results"][0]["position"]["lat"]
                lon = data["results"][0]["position"]["lon"]
                
                traffic_url = ("https://api.tomtom.com/traffic/services/4/"
                               f"flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key}")
                traffic_resp = requests.get(traffic_url, timeout=10)
                if traffic_resp.status_code == 200:
                    traffic_data = traffic_resp.json()
                    flow_data = traffic_data.get("flowSegmentData", {})
                    if flow_data:
                        return {
                            "street": street_name,
                            "latitude": lat,
                            "longitude": lon,
                            "currentSpeed": flow_data.get("currentSpeed"),
                            "freeFlowSpeed": flow_data.get("freeFlowSpeed"),
                            "currentTravelTime": flow_data.get("currentTravelTime"),
                            "freeFlowTravelTime": flow_data.get("freeFlowTravelTime"),
                            "confidence": flow_data.get("confidence"),
                            "roadClosure": flow_data.get("roadClosure")
                        }
                    print(f"Respuesta de TomTom para {street_name}: {resp.json()}")
    except requests.RequestException as e:
        print(f"Error al obtener tráfico para {street_name}: {e}")
    return None

def write_csv(file_name, data):
    """
    Escribe la lista de diccionarios en un archivo CSV.
    """
    if not data:
        print("No hay datos para escribir en el CSV.")
        return
    fieldnames = list(data[0].keys())
    with open(file_name, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    """
    Obtiene datos de tráfico para todas las calles de Gijón y los guarda en un CSV.
    """
    api_key = "zOiHVbXkmB9oDibWaCK4dx3SjuBs5odI"
    streets = get_gijon_streets()

    print(f"Calles obtenidas: {len(streets)}")
    print(streets[:10])  # Muestra las primeras 10 calles
    
    data = []
    for street in streets:
        traffic_info = get_traffic_data(api_key, street)
        if traffic_info:
            data.append(traffic_info)
        time.sleep(1)  # Evitar sobrecarga de la API
    
    write_csv("trafico_asturies.csv", data)
    print("Proceso finalizado. Datos escritos en trafico_asturies.csv")

if __name__ == "__main__":
    main()
