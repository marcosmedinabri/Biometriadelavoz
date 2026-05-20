"""
augment.py — Aumento de datos (data augmentation) para audio
=============================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

El sobreajuste de la Red 1 viene, en buena parte, de tener pocos audios
por locutor: la red memoriza grabaciones concretas en vez de aprender
a reconocer voces en general. El aumento de datos genera VARIACIONES
artificiales y realistas de cada audio, de modo que la red ve muchos
más ejemplos y se ve forzada a generalizar.

Regla de oro: cada variación debe sonar como "el mismo locutor diciendo
la misma frase en otra grabación". Son cambios SUAVES.

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


# CONFIGURACIÓN DEL AUMENTO

CONFIG_AUG = {
    "ruido_factor":     0.008,
    "shift_max":        0.15,
    "velocidad_min":    0.92,
    "velocidad_max":    1.08,
    "n_aumentos":       2,
}


def agregar_ruido(audio, factor=None, rng=None):
    """Suma ruido gaussiano leve al audio. Simula ruido de fondo."""
    factor = factor if factor is not None else CONFIG_AUG["ruido_factor"]
    rng = rng or np.random.default_rng()
    amplitud = np.max(np.abs(audio)) + 1e-8
    ruido = rng.normal(0, amplitud * factor, size=audio.shape)
    return (audio + ruido).astype(np.float32)


def desplazar_tiempo(audio, shift_max=None, rng=None):
    """Desplaza el audio adelante o atrás en el tiempo."""
    shift_max = shift_max if shift_max is not None else CONFIG_AUG["shift_max"]
    rng = rng or np.random.default_rng()
    max_muestras = int(len(audio) * shift_max)
    if max_muestras == 0:
        return audio.copy()
    shift = rng.integers(-max_muestras, max_muestras + 1)
    desplazado = np.roll(audio, shift)
    if shift > 0:
        desplazado[:shift] = 0.0
    elif shift < 0:
        desplazado[shift:] = 0.0
    return desplazado.astype(np.float32)


def cambiar_velocidad(audio, sr, v_min=None, v_max=None, rng=None):
    """Acelera o ralentiza el audio sin cambiar el tono."""
    v_min = v_min if v_min is not None else CONFIG_AUG["velocidad_min"]
    v_max = v_max if v_max is not None else CONFIG_AUG["velocidad_max"]
    rng = rng or np.random.default_rng()
    factor = rng.uniform(v_min, v_max)
    return librosa.effects.time_stretch(audio, rate=factor).astype(np.float32)


def aumentar_audio(audio, sr, cfg=None, rng=None):
    """
    Aplica al audio UNA combinación aleatoria de técnicas de aumento.

    Devuelve el audio aumentado SIN ajustar longitud: el ajuste a
    longitud fija lo hace después data.cargar_audio / el pipeline.
    """
    cfg = cfg or CONFIG_AUG
    rng = rng or np.random.default_rng()

    resultado = audio.copy()
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

    if aplicadas == 0:
        resultado = agregar_ruido(resultado, cfg["ruido_factor"], rng)

    return resultado