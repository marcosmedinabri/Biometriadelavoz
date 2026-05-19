"""
augment.py — Aumento de datos (data augmentation) para audio
=============================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

El sobreajuste de la Red 1 viene, en buena parte, de tener pocos audios
(2658): la red memoriza grabaciones concretas en vez de aprender a
reconocer voces en general. El aumento de datos genera VARIACIONES
artificiales y realistas de cada audio, de modo que la red ve muchos
más ejemplos y se ve forzada a generalizar.

Regla de oro: cada variación debe sonar como "el mismo locutor diciendo
el mismo PIN en otra grabación". Son cambios SUAVES.

Técnicas incluidas (todas seguras para identificación de locutor):
  - agregar_ruido      : suma ruido de fondo leve.
  - desplazar_tiempo   : mueve el audio adelante/atrás en el tiempo.
  - cambiar_velocidad  : acelera/ralentiza sin cambiar el tono.

NO se incluye cambio de tono (pitch shift): el tono es un rasgo de la
IDENTIDAD del locutor; desplazarlo difuminaría justo lo que la Red 1
debe aprender. (Para la Red 2 sí sería inofensivo.)

IMPORTANTE: el aumento se aplica SOLO a los datos de entrenamiento,
nunca a validación ni test (esos deben medir rendimiento sobre datos
reales para que la medida sea honesta).
"""

import numpy as np
import librosa


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL AUMENTO
# ─────────────────────────────────────────────────────────────────

CONFIG_AUG = {
    # Ruido: cantidad relativa a la amplitud de la señal (0.5%–1%)
    "ruido_factor":     0.008,

    # Desplazamiento temporal: fracción máxima de la señal a desplazar
    "shift_max":        0.15,    # hasta un 15% de la duración

    # Cambio de velocidad: rango del factor de velocidad
    "velocidad_min":    0.92,    # hasta un 8% más lento
    "velocidad_max":    1.08,    # hasta un 8% más rápido

    # Nº de copias aumentadas a generar por cada audio original
    "n_aumentos":       2,
}


# ─────────────────────────────────────────────────────────────────
# TÉCNICAS INDIVIDUALES
# ─────────────────────────────────────────────────────────────────

def agregar_ruido(audio, factor=None, rng=None):
    """
    Suma ruido gaussiano leve al audio.

    Simula el ruido de fondo presente en cualquier grabación real
    (ambiente, micrófono). Obliga a la red a ignorar el ruido y
    fijarse solo en la voz. No altera quién habla ni qué dice.

    Args:
        audio  : señal de audio (np.ndarray 1-D)
        factor : intensidad del ruido relativa a la amplitud de la
                 señal. Si None, usa CONFIG_AUG.
        rng    : generador aleatorio de NumPy (para reproducibilidad)

    Returns:
        audio con ruido (misma longitud)
    """
    factor = factor if factor is not None else CONFIG_AUG["ruido_factor"]
    rng = rng or np.random.default_rng()

    amplitud = np.max(np.abs(audio)) + 1e-8
    ruido = rng.normal(0, amplitud * factor, size=audio.shape)
    return (audio + ruido).astype(np.float32)


def desplazar_tiempo(audio, shift_max=None, rng=None):
    """
    Desplaza el audio adelante o atrás en el tiempo.

    Simula que el locutor empieza a hablar un poco antes o después.
    Enseña a la red que la posición exacta de la voz no importa.
    El hueco que deja el desplazamiento se rellena con silencio.

    Args:
        audio     : señal de audio (np.ndarray 1-D)
        shift_max : fracción máxima de la longitud a desplazar.
                    Si None, usa CONFIG_AUG.
        rng       : generador aleatorio de NumPy

    Returns:
        audio desplazado (misma longitud)
    """
    shift_max = shift_max if shift_max is not None else CONFIG_AUG["shift_max"]
    rng = rng or np.random.default_rng()

    # Desplazamiento en muestras: positivo = hacia delante
    max_muestras = int(len(audio) * shift_max)
    if max_muestras == 0:
        return audio.copy()
    shift = rng.integers(-max_muestras, max_muestras + 1)

    # np.roll mueve circularmente; luego ponemos a cero la parte
    # que ha "dado la vuelta" para que sea silencio, no audio repetido.
    desplazado = np.roll(audio, shift)
    if shift > 0:
        desplazado[:shift] = 0.0
    elif shift < 0:
        desplazado[shift:] = 0.0
    return desplazado.astype(np.float32)


def cambiar_velocidad(audio, sr, v_min=None, v_max=None, rng=None):
    """
    Acelera o ralentiza el audio sin cambiar el tono.

    Simula que el locutor habla a un ritmo algo distinto. El tono se
    mantiene, así que la voz sigue sonando a la misma persona.

    Nota: cambiar la velocidad cambia la LONGITUD del audio. Quien
    llame a esta función debe re-ajustar la longitud después (con
    pad/truncate), ya que las redes necesitan longitud fija.

    Args:
        audio        : señal de audio (np.ndarray 1-D)
        sr           : frecuencia de muestreo
        v_min, v_max : rango del factor de velocidad. Si None, usa
                       CONFIG_AUG. (>1 = más rápido, <1 = más lento)
        rng          : generador aleatorio de NumPy

    Returns:
        audio con velocidad alterada (longitud DISTINTA al original)
    """
    v_min = v_min if v_min is not None else CONFIG_AUG["velocidad_min"]
    v_max = v_max if v_max is not None else CONFIG_AUG["velocidad_max"]
    rng = rng or np.random.default_rng()

    factor = rng.uniform(v_min, v_max)
    # librosa.effects.time_stretch cambia la duración conservando el tono
    return librosa.effects.time_stretch(audio, rate=factor).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# APLICAR UN AUMENTO ALEATORIO
# ─────────────────────────────────────────────────────────────────

def aumentar_audio(audio, sr, cfg=None, rng=None):
    """
    Aplica al audio UNA combinación aleatoria de técnicas de aumento.

    Para cada copia aumentada se eligen al azar una o varias de las
    técnicas seguras. Así las variaciones son diversas y la red ve
    ejemplos muy distintos entre sí.

    Devuelve el audio aumentado SIN ajustar longitud: el ajuste a
    longitud fija lo hace después data.cargar_audio / el pipeline.

    Args:
        audio : señal de audio (np.ndarray 1-D)
        sr    : frecuencia de muestreo
        cfg   : configuración de aumento; si None usa CONFIG_AUG
        rng   : generador aleatorio de NumPy

    Returns:
        audio aumentado (la longitud puede variar si se aplicó
        cambio de velocidad)
    """
    cfg = cfg or CONFIG_AUG
    rng = rng or np.random.default_rng()

    resultado = audio.copy()

    # Cada técnica se aplica con un 50% de probabilidad. Garantizamos
    # que al menos UNA se aplique, para que la copia no sea idéntica.
    aplicadas = 0

    if rng.random() < 0.5:
        resultado = agregar_ruido(resultado, cfg["ruido_factor"], rng)
        aplicadas += 1

    if rng.random() < 0.5:
        resultado = desplazar_tiempo(resultado, cfg["shift_max"], rng)
        aplicadas += 1

    if rng.random() < 0.5:
        resultado = cambiar_velocidad(
            resultado, sr,
            cfg["velocidad_min"], cfg["velocidad_max"], rng)
        aplicadas += 1

    # Si por azar no se aplicó ninguna, forzamos el ruido (la más suave)
    if aplicadas == 0:
        resultado = agregar_ruido(resultado, cfg["ruido_factor"], rng)

    return resultado


# ─────────────────────────────────────────────────────────────────
# Prueba rápida del módulo (python augment.py)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Prueba del módulo augment.py ===\n")

    # Señal de prueba: 2 s de una onda senoidal a 16 kHz
    sr = 16000
    t = np.linspace(0, 2, 2 * sr)
    senal = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    rng = np.random.default_rng(42)

    print(f"Audio original          : {len(senal)} muestras")
    print(f"Con ruido               : {len(agregar_ruido(senal, rng=rng))} muestras")
    print(f"Desplazado en el tiempo  : {len(desplazar_tiempo(senal, rng=rng))} muestras")
    vel = cambiar_velocidad(senal, sr, rng=rng)
    print(f"Velocidad alterada       : {len(vel)} muestras (longitud distinta, "
          f"se re-ajusta después)")
    aug = aumentar_audio(senal, sr, rng=rng)
    print(f"Aumento aleatorio combinado: {len(aug)} muestras")

    print("\nOK: el módulo funciona correctamente.")
