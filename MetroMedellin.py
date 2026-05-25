"""
=============================================================
 Sistema Inteligente de Rutas - Metro de Medellín
 Actividad 2 - Búsqueda y Sistemas Basados en Reglas
 Materia: Inteligencia Artificial
 Autor: Keannu Ivorys Devia Bohorquez
=============================================================

Descripción:
  Sistema basado en conocimiento que encuentra la mejor ruta
  entre dos estaciones del sistema de transporte masivo de
  Medellín (Metro, Tranvía y Cables) usando:
    - Base de conocimiento en reglas lógicas
    - Algoritmo de búsqueda A* con heurística de distancia
    - Representación del grafo como red de transporte

Ejecución:
  python MetroMedellin.py

Dependencias:
  Solo librerías estándar de Python (heapq, math, sys)
=============================================================
"""

import heapq
import math
import sys

# =============================================================
# BASE DE CONOCIMIENTO - ESTACIONES
# Representada como diccionario: nombre -> (latitud, longitud)
# Fuente: coordenadas aproximadas reales del sistema Metro
# =============================================================
ESTACIONES = {
    # --- LÍNEA A (Norte-Sur) ---
    "Niquía":           (6.3382, -75.6147),
    "Bello":            (6.3291, -75.6040),
    "Madera":           (6.3152, -75.5942),
    "Acevedo":          (6.2993, -75.5805),
    "Tricentenario":    (6.2871, -75.5748),
    "Caribe":           (6.2766, -75.5723),
    "Universidad":      (6.2672, -75.5675),
    "Hospital":         (6.2595, -75.5662),
    "Prado":            (6.2527, -75.5654),
    "Parque Berrío":    (6.2467, -75.5665),
    "San Antonio":      (6.2437, -75.5699),
    "Alpujarra":        (6.2406, -75.5739),
    "Exposiciones":     (6.2357, -75.5773),
    "Industriales":     (6.2283, -75.5792),
    "Poblado":          (6.2094, -75.5740),
    "Aguacatala":       (6.1988, -75.5733),
    "Ayurá":            (6.1873, -75.5726),
    "Envigado":         (6.1742, -75.5865),
    "Itagüí":           (6.1625, -75.5992),
    "La Estrella":      (6.1534, -75.6095),

    # --- LÍNEA B (Este-Oeste) ---
    "San Javier":       (6.2466, -75.6097),
    "Floresta":         (6.2462, -75.6023),
    "Santa Lucía":      (6.2455, -75.5955),
    "Trinidad":         (6.2448, -75.5887),
    "Suramericana":     (6.2443, -75.5819),
    # San Antonio ya está en Línea A (estación de transbordo)

    # --- TRANVÍA DE AYACUCHO ---
    "Miraflores":       (6.2468, -75.5605),
    "Alejandro Echavarría": (6.2476, -75.5546),
    "Bicentenario":     (6.2479, -75.5497),
    "Buenos Aires":     (6.2484, -75.5447),
    "Loyola":           (6.2490, -75.5390),
    "Asomadera":        (6.2498, -75.5336),
    "Oriente":          (6.2504, -75.5278),

    # --- CABLE LÍNEA J (San Javier) ---
    "Juan XXIII":       (6.2520, -75.6198),
    "Vallejuelos":      (6.2568, -75.6289),
    "La Aurora":        (6.2610, -75.6372),

    # --- CABLE LÍNEA K (Acevedo) ---
    "Andalucía":        (6.3025, -75.5712),
    "Villa Sierra":     (6.3062, -75.5625),
    "La Independencia": (6.3095, -75.5542),
    "Santo Domingo":    (6.3130, -75.5458),

    # --- CABLE LÍNEA L (Santo Domingo - Parque Arví) ---
    "Arví":             (6.2956, -75.4989),

    # --- CABLE LÍNEA M (Miraflores) ---
    "Trece de Noviembre": (6.2391, -75.5556),
    "La Sierra":        (6.2315, -75.5490),
}

# =============================================================
# BASE DE CONOCIMIENTO - CONEXIONES (REGLAS)
# Formato: (estacion_origen, estacion_destino, linea, tiempo_min)
# Las conexiones son bidireccionales.
# =============================================================
CONEXIONES = [
    # --- LÍNEA A ---
    ("Niquía",        "Bello",           "A", 3),
    ("Bello",         "Madera",          "A", 3),
    ("Madera",        "Acevedo",         "A", 4),
    ("Acevedo",       "Tricentenario",   "A", 3),
    ("Tricentenario", "Caribe",          "A", 3),
    ("Caribe",        "Universidad",     "A", 3),
    ("Universidad",   "Hospital",        "A", 2),
    ("Hospital",      "Prado",           "A", 2),
    ("Prado",         "Parque Berrío",   "A", 2),
    ("Parque Berrío", "San Antonio",     "A", 2),
    ("San Antonio",   "Alpujarra",       "A", 2),
    ("Alpujarra",     "Exposiciones",    "A", 2),
    ("Exposiciones",  "Industriales",    "A", 3),
    ("Industriales",  "Poblado",         "A", 4),
    ("Poblado",       "Aguacatala",      "A", 3),
    ("Aguacatala",    "Ayurá",           "A", 3),
    ("Ayurá",         "Envigado",        "A", 4),
    ("Envigado",      "Itagüí",          "A", 4),
    ("Itagüí",        "La Estrella",     "A", 3),

    # --- LÍNEA B ---
    ("San Javier",    "Floresta",        "B", 3),
    ("Floresta",      "Santa Lucía",     "B", 3),
    ("Santa Lucía",   "Trinidad",        "B", 3),
    ("Trinidad",      "Suramericana",    "B", 3),
    ("Suramericana",  "San Antonio",     "B", 3),

    # --- TRANVÍA DE AYACUCHO (desde San Antonio) ---
    ("San Antonio",            "Miraflores",              "T", 4),
    ("Miraflores",             "Alejandro Echavarría",    "T", 3),
    ("Alejandro Echavarría",   "Bicentenario",            "T", 3),
    ("Bicentenario",           "Buenos Aires",            "T", 3),
    ("Buenos Aires",           "Loyola",                  "T", 3),
    ("Loyola",                 "Asomadera",               "T", 3),
    ("Asomadera",              "Oriente",                 "T", 3),

    # --- CABLE LÍNEA J (desde San Javier) ---
    ("San Javier",    "Juan XXIII",      "J", 5),
    ("Juan XXIII",    "Vallejuelos",     "J", 5),
    ("Vallejuelos",   "La Aurora",       "J", 5),

    # --- CABLE LÍNEA K (desde Acevedo) ---
    ("Acevedo",          "Andalucía",        "K", 5),
    ("Andalucía",        "Villa Sierra",     "K", 4),
    ("Villa Sierra",     "La Independencia", "K", 4),
    ("La Independencia", "Santo Domingo",    "K", 4),

    # --- CABLE LÍNEA L (desde Santo Domingo a Arví) ---
    ("Santo Domingo", "Arví",            "L", 15),

    # --- CABLE LÍNEA M (desde Miraflores) ---
    ("Miraflores",          "Trece de Noviembre", "M", 6),
    ("Trece de Noviembre",  "La Sierra",          "M", 5),
]

# =============================================================
# REGLAS LÓGICAS DEL SISTEMA
# Define restricciones y condiciones del sistema de transporte
# =============================================================
REGLAS = [
    {
        "id": "R1",
        "descripcion": "Transbordo en San Antonio entre Línea A y Línea B",
        "condicion": lambda origen, linea_salida, linea_llegada: (
            origen == "San Antonio" and
            set([linea_salida, linea_llegada]) == set(["A", "B"])
        ),
        "accion": "Permitir cambio entre Línea A y Línea B en San Antonio"
    },
    {
        "id": "R2",
        "descripcion": "Transbordo en Acevedo entre Línea A y Cable K",
        "condicion": lambda origen, linea_salida, linea_llegada: (
            origen == "Acevedo" and
            set([linea_salida, linea_llegada]) == set(["A", "K"])
        ),
        "accion": "Permitir cambio entre Línea A y Cable K en Acevedo"
    },
    {
        "id": "R3",
        "descripcion": "Transbordo en San Javier entre Línea B y Cable J",
        "condicion": lambda origen, linea_salida, linea_llegada: (
            origen == "San Javier" and
            set([linea_salida, linea_llegada]) == set(["B", "J"])
        ),
        "accion": "Permitir cambio entre Línea B y Cable J en San Javier"
    },
    {
        "id": "R4",
        "descripcion": "Transbordo en Miraflores entre Tranvía y Cable M",
        "condicion": lambda origen, linea_salida, linea_llegada: (
            origen == "Miraflores" and
            set([linea_salida, linea_llegada]) == set(["T", "M"])
        ),
        "accion": "Permitir cambio entre Tranvía y Cable M en Miraflores"
    },
    {
        "id": "R5",
        "descripcion": "Transbordo en Santo Domingo entre Cable K y Cable L",
        "condicion": lambda origen, linea_salida, linea_llegada: (
            origen == "Santo Domingo" and
            set([linea_salida, linea_llegada]) == set(["K", "L"])
        ),
        "accion": "Permitir cambio entre Cable K y Cable L en Santo Domingo"
    },
    {
        "id": "R6",
        "descripcion": "Tiempo adicional por transbordo de línea",
        "condicion": lambda origen, linea_salida, linea_llegada: False,
        "accion": "Agregar 3 minutos de espera al cambiar de línea"
    },
]

# Costo adicional en minutos por transbordo de línea
COSTO_TRANSBORDO = 3

# =============================================================
# CONSTRUCCIÓN DEL GRAFO
# =============================================================
def construir_grafo():
    """Construye el grafo de adyacencia bidireccional."""
    grafo = {estacion: [] for estacion in ESTACIONES}
    for origen, destino, linea, tiempo in CONEXIONES:
        grafo[origen].append((destino, linea, tiempo))
        grafo[destino].append((origen, linea, tiempo))
    return grafo

# =============================================================
# HEURÍSTICA - DISTANCIA HAVERSINE
# Calcula la distancia en km entre dos puntos geográficos.
# Se usa como heurística admisible para A*.
# =============================================================
def distancia_haversine(est1, est2):
    """
    Calcula distancia real entre dos estaciones usando
    la fórmula de Haversine (distancia sobre esfera terrestre).
    Retorna distancia en kilómetros.
    """
    lat1, lon1 = ESTACIONES[est1]
    lat2, lon2 = ESTACIONES[est2]
    R = 6371  # Radio de la Tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def heuristica(estacion_actual, destino):
    """
    Heurística admisible para A*:
    Convierte distancia geográfica a minutos estimados
    asumiendo velocidad promedio de 40 km/h en Metro.
    """
    dist_km = distancia_haversine(estacion_actual, destino)
    return (dist_km / 40) * 60  # minutos

# =============================================================
# ALGORITMO A* - BÚSQUEDA DE RUTA ÓPTIMA
# =============================================================
def buscar_ruta_astar(origen, destino, grafo):
    """
    Implementación del algoritmo A* para encontrar la ruta
    de menor tiempo entre origen y destino.

    Estados: (costo_acumulado, estacion_actual, linea_actual, camino)
    Heurística: distancia haversine convertida a minutos

    Retorna:
        (camino, tiempo_total, transbordos) o None si no hay ruta
    """
    if origen not in ESTACIONES:
        return None, None, None
    if destino not in ESTACIONES:
        return None, None, None
    if origen == destino:
        return [origen], 0, 0

    # Cola de prioridad: (f = g + h, g, estacion, linea_actual, camino, transbordos)
    cola = []
    h_inicial = heuristica(origen, destino)
    heapq.heappush(cola, (h_inicial, 0, origen, None, [origen], 0))

    # Registro de mejor costo conocido por (estacion, linea)
    visitados = {}

    while cola:
        f, g, estacion, linea_actual, camino, transbordos = heapq.heappop(cola)

        # Llegamos al destino
        if estacion == destino:
            return camino, g, transbordos

        # Clave de estado: estacion + línea en que llegamos
        estado = (estacion, linea_actual)
        if estado in visitados and visitados[estado] <= g:
            continue
        visitados[estado] = g

        # Explorar vecinos
        for vecino, linea_vecino, tiempo in grafo[estacion]:
            if vecino in camino:
                continue  # Evitar ciclos

            # Calcular costo con posible transbordo
            costo_extra = 0
            nuevo_transbordos = transbordos
            if linea_actual is not None and linea_vecino != linea_actual:
                costo_extra = COSTO_TRANSBORDO  # Regla R6
                nuevo_transbordos += 1

            nuevo_g = g + tiempo + costo_extra
            nuevo_h = heuristica(vecino, destino)
            nuevo_f = nuevo_g + nuevo_h

            nuevo_camino = camino + [vecino]
            heapq.heappush(cola, (
                nuevo_f, nuevo_g, vecino,
                linea_vecino, nuevo_camino, nuevo_transbordos
            ))

    return None, None, None  # No se encontró ruta

# =============================================================
# IDENTIFICAR LÍNEA ENTRE DOS ESTACIONES
# =============================================================
def obtener_linea(est1, est2, grafo):
    """Retorna la línea que conecta dos estaciones adyacentes."""
    for vecino, linea, _ in grafo[est1]:
        if vecino == est2:
            return linea
    return "?"

NOMBRES_LINEA = {
    "A": "Línea A (Metro Norte-Sur)",
    "B": "Línea B (Metro Este-Oeste)",
    "T": "Tranvía de Ayacucho",
    "J": "Cable Línea J",
    "K": "Cable Línea K",
    "L": "Cable Línea L (Arví)",
    "M": "Cable Línea M",
}

# =============================================================
# MOSTRAR RUTA CON DETALLE
# =============================================================
def mostrar_ruta(origen, destino, camino, tiempo, transbordos, grafo):
    """Imprime la ruta encontrada de forma detallada y legible."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RUTA ENCONTRADA - METRO DE MEDELLÍN")
    print(sep)
    print(f"  Origen  : {origen}")
    print(f"  Destino : {destino}")
    print(f"  Tiempo  : {tiempo:.1f} minutos")
    print(f"  Paradas : {len(camino) - 1}")
    print(f"  Transbordos: {transbordos}")
    print(sep)
    print("\n  DETALLE DEL RECORRIDO:")
    print()

    linea_anterior = None
    for i, estacion in enumerate(camino):
        if i < len(camino) - 1:
            linea = obtener_linea(estacion, camino[i + 1], grafo)
            nombre_linea = NOMBRES_LINEA.get(linea, linea)

            if linea != linea_anterior and linea_anterior is not None:
                print(f"      ↕  [TRANSBORDO → {nombre_linea}] (+{COSTO_TRANSBORDO} min)")
            elif linea != linea_anterior:
                print(f"      ● {nombre_linea}")

            print(f"      {i + 1:2}. {estacion}")
            linea_anterior = linea
        else:
            print(f"      {i + 1:2}. {estacion} ← DESTINO")

    print(f"\n{sep}\n")

# =============================================================
# APLICAR Y MOSTRAR REGLAS DISPARADAS
# =============================================================
def mostrar_reglas_disparadas(camino, grafo):
    """Muestra qué reglas lógicas se aplicaron durante el recorrido."""
    print("  REGLAS LÓGICAS APLICADAS:")
    print()
    reglas_disparadas = set()
    hubo_transbordo = False
    linea_anterior = None

    for i in range(len(camino) - 1):
        est = camino[i]
        linea = obtener_linea(est, camino[i + 1], grafo)

        # Solo disparar reglas cuando REALMENTE cambia la línea
        if linea_anterior is not None and linea != linea_anterior:
            hubo_transbordo = True
            # Buscar qué regla R1-R5 aplica a este transbordo real
            for regla in REGLAS:
                if regla["id"] == "R6":
                    continue
                if regla["condicion"](est, linea_anterior, linea):
                    if regla["id"] not in reglas_disparadas:
                        reglas_disparadas.add(regla["id"])
                        print(f"  [{regla['id']}] {regla['descripcion']}")
                        print(f"       → Acción: {regla['accion']}")
                        print()
            # R6 siempre aplica cuando hay transbordo real
            if "R6" not in reglas_disparadas:
                reglas_disparadas.add("R6")
                r6 = next(r for r in REGLAS if r["id"] == "R6")
                print(f"  [R6] {r6['descripcion']}")
                print(f"       → Acción: {r6['accion']}")
                print()

        linea_anterior = linea

    if not hubo_transbordo:
        print("  (Sin transbordos - viaje en una sola línea)")
        print()

# =============================================================
# MENÚ INTERACTIVO
# =============================================================
def mostrar_estaciones():
    """Muestra todas las estaciones agrupadas por línea."""
    grupos = {
        "Línea A (Norte-Sur)": [
            "Niquía","Bello","Madera","Acevedo","Tricentenario","Caribe",
            "Universidad","Hospital","Prado","Parque Berrío","San Antonio",
            "Alpujarra","Exposiciones","Industriales","Poblado","Aguacatala",
            "Ayurá","Envigado","Itagüí","La Estrella"
        ],
        "Línea B (Este-Oeste)": [
            "San Javier","Floresta","Santa Lucía","Trinidad","Suramericana","San Antonio"
        ],
        "Tranvía de Ayacucho": [
            "San Antonio","Miraflores","Alejandro Echavarría","Bicentenario",
            "Buenos Aires","Loyola","Asomadera","Oriente"
        ],
        "Cable Línea J": ["San Javier","Juan XXIII","Vallejuelos","La Aurora"],
        "Cable Línea K": ["Acevedo","Andalucía","Villa Sierra","La Independencia","Santo Domingo"],
        "Cable Línea L": ["Santo Domingo","Arví"],
        "Cable Línea M": ["Miraflores","Trece de Noviembre","La Sierra"],
    }
    print("\n  ESTACIONES DISPONIBLES:")
    print()
    for linea, estaciones in grupos.items():
        print(f"  [{linea}]")
        for est in estaciones:
            print(f"    - {est}")
        print()

def main():
    grafo = construir_grafo()

    print("=" * 60)
    print("  SISTEMA INTELIGENTE DE RUTAS - METRO DE MEDELLÍN")
    print("  Algoritmo: A* con heurística Haversine")
    print("  Autor: Keannu Ivorys Devia Bohorquez")
    print("=" * 60)

    while True:
        print("\n  OPCIONES:")
        print("  1. Buscar ruta")
        print("  2. Ver estaciones disponibles")
        print("  3. Ejecutar casos de prueba")
        print("  4. Salir")
        print()
        opcion = input("  Seleccione una opción: ").strip()

        if opcion == "1":
            mostrar_estaciones()
            origen = input("  Ingrese estación de ORIGEN  : ").strip()
            destino = input("  Ingrese estación de DESTINO : ").strip()

            # Validar entradas
            if origen not in ESTACIONES:
                print(f"\n  ✗ La estación '{origen}' no existe en el sistema.")
                continue
            if destino not in ESTACIONES:
                print(f"\n  ✗ La estación '{destino}' no existe en el sistema.")
                continue

            camino, tiempo, transbordos = buscar_ruta_astar(origen, destino, grafo)

            if camino is None:
                print(f"\n  ✗ No se encontró ruta entre {origen} y {destino}.")
            else:
                mostrar_ruta(origen, destino, camino, tiempo, transbordos, grafo)
                mostrar_reglas_disparadas(camino, grafo)

        elif opcion == "2":
            mostrar_estaciones()

        elif opcion == "3":
            casos = [
                ("Niquía",       "La Estrella",   "Recorrido completo Línea A"),
                ("San Javier",   "Miraflores",    "Línea B → San Antonio → Tranvía"),
                ("Bello",        "Santo Domingo", "Metro + Cable K"),
                ("La Aurora",    "Arví",          "Cable J → Línea B → A → Cable K → L"),
                ("Poblado",      "Buenos Aires",  "Línea A → Tranvía"),
                ("La Estrella",  "Oriente",       "Ruta larga multi-línea"),
            ]
            for origen, destino, descripcion in casos:
                print(f"\n  {'─'*58}")
                print(f"  CASO: {descripcion}")
                camino, tiempo, transbordos = buscar_ruta_astar(origen, destino, grafo)
                if camino:
                    mostrar_ruta(origen, destino, camino, tiempo, transbordos, grafo)
                    mostrar_reglas_disparadas(camino, grafo)
                else:
                    print(f"  ✗ Sin ruta entre {origen} y {destino}")

        elif opcion == "4":
            print("\n  ¡Hasta luego!\n")
            break
        else:
            print("\n  Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()