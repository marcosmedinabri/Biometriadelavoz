"""
data.py — Carga y preparación de la base de datos PIN_16kHz
============================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

Este módulo recorre la base de datos del profesor (reconocedorLocutor_TF/
PIN_16kHz), extrae características acústicas de cada audio y devuelve los
datos listos para entrenar las dos redes:

  - Red 1: identificación de LOCUTOR  (24 clases)
  - Red 2: verificación de PIN/FRASE  (40 clases)

Estructura de la base de datos:
    PIN_16kHz/
      <locutor>/                  ej: 093, 147, a77, b48 ...
        <locutor>_pin(XXXX)_cN.wav        PIN normal
        <locutor>_pin(XXXX)M_cN.wav       PIN de familia (M = variante)

Decisiones tomadas en la Fase 3 (exploración), todas justificadas:
  - duration = 2.0 s  -> cubre el percentil 95 (1.95 s) sin recortar y
                         sin rellenar con silencio innecesario.
  - Audios "(def)"    -> EXCLUIDOS del entrenamiento; se reservan para
                         las pruebas de robustez.
  - Etiqueta de PIN   -> contenido del paréntesis + dígito de variante.
                         "pin(3920)0" y "pin(3920)1" son PINs DISTINTOS.
  - Errata "3290"     -> se reasigna a "3920" (dígitos transpuestos).
"""

import re
from pathlib import Path

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import augment

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

CONFIG = {
    # Ruta a la base de datos (relativa a la raíz del proyecto)
    "data_dir":     "reconocedorLocutor_TF/PIN_16kHz",

    # Audio
    "sample_rate":  16000,   # Hz — la base de datos ya está a 16 kHz
    "duration":     2.0,     # s  — decidido en la Fase 3

    # Extracción de características (MFCC)
    "n_mfcc":       40,      # nº de coeficientes cepstrales
    "n_fft":        1024,
    "hop_length":   512,
    "use_delta":    True,    # añadir deltas y delta-deltas

    # Splits (estratificados)
    "test_size":    0.15,
    "val_size":     0.15,
    "random_state": 42,
}


# ─────────────────────────────────────────────────────────────────
# 1. RECORRIDO DE LA BASE DE DATOS  ->  ÍNDICE DE FICHEROS
# ─────────────────────────────────────────────────────────────────

# Regex que captura el PIN:
#   grupo 1 = contenido del paréntesis  ->  "3920"
#   grupo 2 = dígito de variante opcional tras el ")" -> "0", "1", ... o ""
_PIN_RE = re.compile(r"pin\(([^)]+)\)(\d*)")


def _pin_de_nombre(nombre_fichero):
    """
    Extrae la etiqueta de PIN del nombre de un .wav.

    "093_pin(0185)_c2.wav"   -> "0185"
    "093_pin(3920)7_c1.wav"  -> "39207"
    "785_pin(3290)0_c1.wav"  -> "39200"  (errata 3290 corregida)

    Devuelve None si el nombre no encaja con el patrón esperado.
    """
    m = _PIN_RE.search(nombre_fichero)
    if m is None:
        return None
    clave = m.group(1) + m.group(2)
    # Corrección de la errata detectada en la Fase 3
    clave = clave.replace("3290", "3920")
    return clave


def construir_indice(data_dir=None, incluir_def=False):
    """
    Recorre la base de datos y construye una lista de registros.

    Cada registro es un dict:
        {"path": <ruta wav>, "locutor": <str>, "pin": <str>}

    Las etiquetas se devuelven como STRINGS legibles (ej. "093", "0185").
    La conversión a enteros 0..N-1 se hace después, en codificar_etiquetas().

    Args:
        data_dir     : ruta a PIN_16kHz. Si None, usa CONFIG["data_dir"].
        incluir_def  : si False (por defecto) descarta los audios "(def)".
                       Poner True solo para las pruebas de robustez.

    Returns:
        registros : list[dict]
    """
    raiz = Path(data_dir or CONFIG["data_dir"])
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"No se encuentra la base de datos en '{raiz}'. "
            f"¿Está descargada y en la ruta correcta?"
        )

    locutores = sorted([d for d in raiz.iterdir() if d.is_dir()])
    if not locutores:
        raise ValueError(f"No hay subcarpetas de locutor en '{raiz}'.")

    registros = []
    n_def_descartados = 0
    n_sin_pin = 0

    for carpeta_loc in locutores:
        locutor = carpeta_loc.name
        for wav in sorted(carpeta_loc.glob("*.wav")):
            es_def = "(def)" in wav.name

            if es_def and not incluir_def:
                n_def_descartados += 1
                continue

            pin = _pin_de_nombre(wav.name)
            if pin is None:
                # Nombre inesperado: lo avisamos pero no rompemos
                n_sin_pin += 1
                continue

            registros.append({
                "path":    str(wav),
                "locutor": locutor,
                "pin":     pin,
            })

    print(f"Índice construido: {len(registros)} audios")
    print(f"  Locutores : {len(locutores)}")
    print(f"  PINs      : {len(set(r['pin'] for r in registros))}")
    if not incluir_def:
        print(f"  Descartados (def): {n_def_descartados}")
    if n_sin_pin:
        print(f"  AVISO: {n_sin_pin} ficheros con nombre inesperado, omitidos")

    return registros


# ─────────────────────────────────────────────────────────────────
# 2. CODIFICACIÓN DE ETIQUETAS  (str -> entero 0..N-1)
# ─────────────────────────────────────────────────────────────────

def codificar_etiquetas(registros, campo):
    """
    Construye un mapa etiqueta_str -> entero para un campo dado.

    Las redes neuronales necesitan etiquetas enteras, no strings.
    Se ordenan alfabéticamente para que el mapa sea reproducible
    (siempre da el mismo entero a la misma etiqueta).

    Args:
        registros : salida de construir_indice()
        campo     : "locutor" o "pin"

    Returns:
        mapa : dict  {etiqueta_str: entero}
    """
    etiquetas = sorted(set(r[campo] for r in registros))
    return {etq: i for i, etq in enumerate(etiquetas)}


# ─────────────────────────────────────────────────────────────────
# 3. EXTRACCIÓN DE CARACTERÍSTICAS  (audio -> MFCC)
# ─────────────────────────────────────────────────────────────────

def cargar_audio(path, sr, duration):
    """
    Carga un .wav, lo lleva a longitud fija y lo normaliza.

    Longitud fija (pad/truncate): las redes necesitan tensores del mismo
    tamaño para poder agruparlos en lotes. Los audios duran ~1.5 s de
    media; con duration=2.0 s casi ninguno se recorta.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)

    objetivo = int(sr * duration)
    if len(audio) < objetivo:
        # Audio corto: se rellena con ceros (silencio) al final
        audio = np.pad(audio, (0, objetivo - len(audio)))
    else:
        # Audio largo: recorte centrado
        ini = (len(audio) - objetivo) // 2
        audio = audio[ini: ini + objetivo]

    # Normalización de amplitud a [-1, 1]
    pico = np.max(np.abs(audio))
    if pico > 0:
        audio = audio / pico

    return audio


def extraer_mfcc(audio, cfg):
    """
    Extrae los MFCC de un audio y aplica normalización CMVN.

    - MFCC: representan la forma del espectro (timbre, formantes), que
      es lo que distingue tanto a locutores como a fonemas.
    - Deltas / delta-deltas: capturan cómo evolucionan los MFCC en el
      tiempo. Útiles porque una frase es una secuencia temporal.
    - CMVN (resta de media, división por desviación por coeficiente):
      estabiliza el entrenamiento y reduce el efecto del canal/micro.

    Devuelve una matriz (n_caracteristicas, T).
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=cfg["sample_rate"],
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
    )

    if cfg["use_delta"]:
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, delta, delta2])   # (n_mfcc*3, T)

    # CMVN por coeficiente (a lo largo del eje temporal)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / \
           (mfcc.std(axis=1, keepdims=True) + 1e-8)

    return mfcc


# ─────────────────────────────────────────────────────────────────
# 4. CONSTRUCCIÓN DEL DATASET COMPLETO
# ─────────────────────────────────────────────────────────────────

def construir_dataset(registros, cfg=None, verbose=True):
    """
    Convierte el índice de ficheros en arrays de NumPy listos para Keras.

    Procesa CADA audio una sola vez y obtiene sus dos etiquetas, de modo
    que los mismos datos sirven para entrenar la Red 1 y la Red 2.

    Returns:
        X         : np.ndarray (N, n_caract, T, 1)  — los MFCC con canal
        y_locutor : np.ndarray (N,)  — etiqueta de locutor (entero)
        y_pin     : np.ndarray (N,)  — etiqueta de PIN (entero)
        mapas     : dict con "locutor" y "pin" -> {str: entero}
    """
    cfg = cfg or CONFIG

    mapa_loc = codificar_etiquetas(registros, "locutor")
    mapa_pin = codificar_etiquetas(registros, "pin")

    X, y_locutor, y_pin = [], [], []

    for i, reg in enumerate(registros):
        if verbose and i % 250 == 0:
            print(f"  Procesando audio {i}/{len(registros)}...")

        audio = cargar_audio(reg["path"], cfg["sample_rate"], cfg["duration"])
        mfcc  = extraer_mfcc(audio, cfg)

        X.append(mfcc)
        y_locutor.append(mapa_loc[reg["locutor"]])
        y_pin.append(mapa_pin[reg["pin"]])

    X = np.array(X, dtype=np.float32)
    X = X[..., np.newaxis]   # canal: (N, n_caract, T) -> (N, n_caract, T, 1)

    y_locutor = np.array(y_locutor, dtype=np.int64)
    y_pin     = np.array(y_pin,     dtype=np.int64)

    if verbose:
        print(f"\nDataset construido:")
        print(f"  X         : {X.shape}")
        print(f"  y_locutor : {y_locutor.shape}  ({len(mapa_loc)} clases)")
        print(f"  y_pin     : {y_pin.shape}  ({len(mapa_pin)} clases)")

    return X, y_locutor, y_pin, {"locutor": mapa_loc, "pin": mapa_pin}


# ─────────────────────────────────────────────────────────────────
# 5. SPLITS TRAIN / VAL / TEST  (estratificados)
# ─────────────────────────────────────────────────────────────────

def dividir_datos(X, y, cfg=None):
    """
    Divide en train / val / test de forma ESTRATIFICADA.

    Estratificado = cada clase aparece en los tres subconjuntos en la
    misma proporción. Es importante aquí porque las clases de PIN no
    están equilibradas (los PINs normales tienen ~90 audios y las
    variantes de familia ~45): sin estratificar, una clase pequeña
    podría quedar fuera de validación o de test por azar.

    'y' debe ser el vector de etiquetas sobre el que estratificar:
    usa y_locutor para la Red 1 e y_pin para la Red 2.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    cfg = cfg or CONFIG

    # Primer corte: separamos test
    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        stratify=y,
        random_state=cfg["random_state"],
    )

    # Segundo corte: del resto, separamos validación
    # (ajustamos la proporción porque se calcula sobre un conjunto menor)
    val_ratio = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto,
        test_size=val_ratio,
        stratify=y_resto,
        random_state=cfg["random_state"],
    )

    print(f"Splits -> train: {len(X_train)}  |  "
          f"val: {len(X_val)}  |  test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def dividir_registros(registros, cfg=None):
    """
    Divide la LISTA DE REGISTROS en train / val / test (estratificado).

    A diferencia de dividir_datos(), que parte los MFCC ya calculados,
    esta función parte los registros (rutas de fichero + etiquetas).
    Es lo que permite aplicar el aumento de datos: necesitamos saber
    QUÉ ficheros de audio están en train para poder aumentarlos antes
    de extraer sus MFCC.

    Estratifica por locutor.

    Args:
        registros : salida de construir_indice()
        cfg       : config; si None usa CONFIG

    Returns:
        reg_train, reg_val, reg_test : tres listas de registros
    """
    cfg = cfg or CONFIG

    # Vector de etiquetas de locutor para estratificar
    y_loc = [r["locutor"] for r in registros]

    # Primer corte: separamos test
    reg_resto, reg_test = train_test_split(
        registros,
        test_size=cfg["test_size"],
        stratify=y_loc,
        random_state=cfg["random_state"],
    )

    # Segundo corte: del resto, separamos validación
    val_ratio = cfg["val_size"] / (1 - cfg["test_size"])
    y_loc_resto = [r["locutor"] for r in reg_resto]
    reg_train, reg_val = train_test_split(
        reg_resto,
        test_size=val_ratio,
        stratify=y_loc_resto,
        random_state=cfg["random_state"],
    )

    print(f"Registros divididos -> train: {len(reg_train)}  |  "
          f"val: {len(reg_val)}  |  test: {len(reg_test)}")

    return reg_train, reg_val, reg_test


def construir_dataset_desde_registros(registros, mapas, cfg=None,
                                      aumentar=False, n_aug=2,
                                      verbose=True):
    """
    Construye los arrays (X, y_locutor, y_pin) a partir de una lista
    de registros, usando MAPAS de etiquetas ya existentes.

    Se diferencia de construir_dataset() en dos cosas:
      - Recibe los mapas de etiquetas desde fuera (en vez de crearlos),
        para que train/val/test usen LA MISMA codificación.
      - Puede aplicar aumento de datos (solo debe usarse en train).

    Si aumentar=True, por cada audio original se generan 'n_aug' copias
    aumentadas, que se añaden al conjunto con las MISMAS etiquetas que
    el original (una variación del locutor 093 sigue siendo el 093).

    Args:
        registros : lista de registros (de dividir_registros)
        mapas     : dict {"locutor":..., "pin":...} de construir_dataset
        cfg       : config; si None usa CONFIG
        aumentar  : si True, genera copias aumentadas
        n_aug     : nº de copias aumentadas por audio original
        verbose   : imprime progreso

    Returns:
        X, y_locutor, y_pin  (np.ndarray)
    """
    cfg = cfg or CONFIG
    mapa_loc, mapa_pin = mapas["locutor"], mapas["pin"]
    rng = np.random.default_rng(cfg["random_state"])

    X, y_locutor, y_pin = [], [], []

    for i, reg in enumerate(registros):
        if verbose and i % 250 == 0:
            print(f"  Procesando audio {i}/{len(registros)}...")

        et_loc = mapa_loc[reg["locutor"]]
        et_pin = mapa_pin[reg["pin"]]

        # --- Audio original ---
        audio = cargar_audio(reg["path"], cfg["sample_rate"], cfg["duration"])
        X.append(extraer_mfcc(audio, cfg))
        y_locutor.append(et_loc)
        y_pin.append(et_pin)

        # --- Copias aumentadas (solo si se pide) ---
        if aumentar:
            # cargamos el audio crudo (sin recortar) para aumentarlo
            crudo, _ = librosa.load(reg["path"], sr=cfg["sample_rate"],
                                    mono=True)
            for _ in range(n_aug):
                aug = augment.aumentar_audio(crudo, cfg["sample_rate"], rng=rng)
                # re-ajustamos a longitud fija (el aumento pudo cambiarla)
                objetivo = int(cfg["sample_rate"] * cfg["duration"])
                if len(aug) < objetivo:
                    aug = np.pad(aug, (0, objetivo - len(aug)))
                else:
                    ini = (len(aug) - objetivo) // 2
                    aug = aug[ini:ini + objetivo]
                # normalizar amplitud, igual que en cargar_audio
                pico = np.max(np.abs(aug))
                if pico > 0:
                    aug = aug / pico

                X.append(extraer_mfcc(aug, cfg))
                y_locutor.append(et_loc)
                y_pin.append(et_pin)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y_locutor = np.array(y_locutor, dtype=np.int64)
    y_pin     = np.array(y_pin,     dtype=np.int64)

    if verbose:
        extra = f" (con aumento x{n_aug})" if aumentar else ""
        print(f"  Construido: {X.shape}{extra}")

    return X, y_locutor, y_pin


# ─────────────────────────────────────────────────────────────────
# Prueba rápida del módulo (se ejecuta solo si lanzas: python data.py)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Prueba del módulo data.py ===\n")

    registros = construir_indice()

    # Construimos el dataset completo
    X, y_loc, y_pin, mapas = construir_dataset(registros)

    # Ejemplo de split para la Red 1 (locutor)
    print("\n-- Split para la Red 1 (locutor) --")
    dividir_datos(X, y_loc)

    print("\nOK: el módulo funciona correctamente.")
