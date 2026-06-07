"""
=============================================================
 Generador de Dataset - Metro de Medellín
 Actividad 3 - Métodos de Aprendizaje Supervisado
 Autor: Keannu Ivorys Devia Bohorquez
=============================================================
Genera un dataset simulado de viajes en el Metro de Medellín
con características relevantes para predecir si una ruta
requiere transbordo o no (clasificación supervisada).
=============================================================
"""

import csv
import random
import math

random.seed(42)

# ── Estaciones por línea ───────────────────────────────────────────────────────
LINEAS = {
    "A": ["Niquía","Bello","Madera","Acevedo","Tricentenario","Caribe",
          "Universidad","Hospital","Prado","Parque Berrío","San Antonio",
          "Alpujarra","Exposiciones","Industriales","Poblado","Aguacatala",
          "Ayurá","Envigado","Itagüí","La Estrella"],
    "B": ["San Javier","Floresta","Santa Lucía","Trinidad","Suramericana","San Antonio"],
    "T": ["San Antonio","Miraflores","Alejandro Echavarría","Bicentenario",
          "Buenos Aires","Loyola","Asomadera","Oriente"],
    "J": ["San Javier","Juan XXIII","Vallejuelos","La Aurora"],
    "K": ["Acevedo","Andalucía","Villa Sierra","La Independencia","Santo Domingo"],
    "L": ["Santo Domingo","Arví"],
    "M": ["Miraflores","Trece de Noviembre","La Sierra"],
}

# Coordenadas reales aproximadas
COORDS = {
    "Niquía":(6.3382,-75.6147),"Bello":(6.3291,-75.6040),"Madera":(6.3152,-75.5942),
    "Acevedo":(6.2993,-75.5805),"Tricentenario":(6.2871,-75.5748),"Caribe":(6.2766,-75.5723),
    "Universidad":(6.2672,-75.5675),"Hospital":(6.2595,-75.5662),"Prado":(6.2527,-75.5654),
    "Parque Berrío":(6.2467,-75.5665),"San Antonio":(6.2437,-75.5699),
    "Alpujarra":(6.2406,-75.5739),"Exposiciones":(6.2357,-75.5773),
    "Industriales":(6.2283,-75.5792),"Poblado":(6.2094,-75.5740),
    "Aguacatala":(6.1988,-75.5733),"Ayurá":(6.1873,-75.5726),
    "Envigado":(6.1742,-75.5865),"Itagüí":(6.1625,-75.5992),"La Estrella":(6.1534,-75.6095),
    "San Javier":(6.2466,-75.6097),"Floresta":(6.2462,-75.6023),
    "Santa Lucía":(6.2455,-75.5955),"Trinidad":(6.2448,-75.5887),
    "Suramericana":(6.2443,-75.5819),
    "Miraflores":(6.2468,-75.5605),"Alejandro Echavarría":(6.2476,-75.5546),
    "Bicentenario":(6.2479,-75.5497),"Buenos Aires":(6.2484,-75.5447),
    "Loyola":(6.2490,-75.5390),"Asomadera":(6.2498,-75.5336),"Oriente":(6.2504,-75.5278),
    "Juan XXIII":(6.2520,-75.6198),"Vallejuelos":(6.2568,-75.6289),"La Aurora":(6.2610,-75.6372),
    "Andalucía":(6.3025,-75.5712),"Villa Sierra":(6.3062,-75.5625),
    "La Independencia":(6.3095,-75.5542),"Santo Domingo":(6.3130,-75.5458),
    "Arví":(6.2956,-75.4989),
    "Trece de Noviembre":(6.2391,-75.5556),"La Sierra":(6.2315,-75.5490),
}

# Nodos de transbordo
TRANSBORDOS = {"San Antonio","Acevedo","San Javier","Miraflores","Santo Domingo"}

def linea_de(estacion):
    for linea, ests in LINEAS.items():
        if estacion in ests:
            return linea
    return "A"

def haversine(e1, e2):
    lat1,lon1 = COORDS[e1]; lat2,lon2 = COORDS[e2]
    R = 6371
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def necesita_transbordo(origen, destino):
    lo = linea_de(origen); ld = linea_de(destino)
    if lo == ld:
        return 0
    # Mismo conjunto de conexión directa
    conectados = {
        frozenset(["A","B"]), frozenset(["A","K"]),
        frozenset(["B","J"]), frozenset(["T","M"]),
        frozenset(["K","L"]),
    }
    if frozenset([lo, ld]) in conectados:
        return 1
    return 1

# Franjas horarias
FRANJAS = ["madrugada","mañana_pico","mañana_valle","tarde_pico","tarde_valle","noche"]
FRANJA_OCUPACION = {
    "madrugada": (0.1, 0.2),
    "mañana_pico": (0.75, 1.0),
    "mañana_valle": (0.3, 0.55),
    "tarde_pico": (0.7, 0.95),
    "tarde_valle": (0.25, 0.5),
    "noche": (0.15, 0.35),
}
FRANJA_ESPERA = {
    "madrugada": (8,15), "mañana_pico": (2,5), "mañana_valle": (4,8),
    "tarde_pico": (2,5), "tarde_valle": (4,9), "noche": (6,12),
}

todas = list(COORDS.keys())

# ── Generar registros ─────────────────────────────────────────────────────────
registros = []
for _ in range(1200):
    origen  = random.choice(todas)
    destino = random.choice([e for e in todas if e != origen])
    franja  = random.choice(FRANJAS)
    dia_sem = random.randint(1, 7)   # 1=lunes … 7=domingo
    es_festivo = 1 if (dia_sem == 7 and random.random() < 0.3) else 0

    dist_km       = round(haversine(origen, destino), 3)
    lo            = linea_de(origen)
    ld            = linea_de(destino)
    misma_linea   = 1 if lo == ld else 0
    origen_transb = 1 if origen in TRANSBORDOS else 0
    destino_transb= 1 if destino in TRANSBORDOS else 0

    oc_min, oc_max = FRANJA_OCUPACION[franja]
    ocupacion     = round(random.uniform(oc_min, oc_max), 2)

    esp_min, esp_max = FRANJA_ESPERA[franja]
    tiempo_espera = random.randint(esp_min, esp_max)

    # Paradas estimadas (simplificado)
    num_paradas   = random.randint(1, 18)

    # Tarifa (en COP)
    tarifa = 3300 if es_festivo == 0 else 3300
    if franja in ["mañana_pico","tarde_pico"]:
        tarifa = 3300

    # Variable objetivo: requiere_transbordo
    requiere_transbordo = necesita_transbordo(origen, destino)

    registros.append({
        "origen": origen,
        "destino": destino,
        "linea_origen": lo,
        "linea_destino": ld,
        "franja_horaria": franja,
        "dia_semana": dia_sem,
        "es_festivo": es_festivo,
        "distancia_km": dist_km,
        "misma_linea": misma_linea,
        "origen_es_transbordo": origen_transb,
        "destino_es_transbordo": destino_transb,
        "ocupacion_vagon": ocupacion,
        "tiempo_espera_min": tiempo_espera,
        "num_paradas_estimadas": num_paradas,
        "tarifa_cop": tarifa,
        "requiere_transbordo": requiere_transbordo,
    })

# Guardar CSV
campos = list(registros[0].keys())
with open("/home/claude/metro_supervisado/metro_viajes.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(registros)

print(f"Dataset generado: {len(registros)} registros")
print(f"Columnas: {campos}")
transbordos_count = sum(r["requiere_transbordo"] for r in registros)
print(f"Viajes con transbordo: {transbordos_count} ({transbordos_count/len(registros)*100:.1f}%)")
print(f"Viajes sin transbordo: {len(registros)-transbordos_count} ({(len(registros)-transbordos_count)/len(registros)*100:.1f}%)")
