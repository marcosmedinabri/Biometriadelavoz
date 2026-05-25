"""
verificacion.py — Protocolo de verificación de locutor texto-dependiente
=========================================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

Este módulo implementa la FASE DE VERIFICACIÓN, que es el corazón del
trabajo. Mientras las redes 1 y 2 hacen IDENTIFICACIÓN (problema cerrado:
elegir entre N clases conocidas), la verificación responde a una pregunta
binaria: dado un audio y una afirmación "soy el locutor L diciendo la
frase F", ¿es cierta o no?

El módulo cubre:

  1. Generación de trials (genuinos e impostores).
  2. Cálculo de scores con dos enfoques:
       - Fusión de probabilidades (Red 1 y Red 2 softmax).
       - Similitud coseno sobre embeddings (capa "embedding" de cada red).
  3. Métricas: EER (Equal Error Rate) y curva DET.

Nota sobre el split de datos: usamos el conjunto de TEST de la Red 1
(estratificado por locutor) como pool de evaluación. Cada locutor tiene
~22 audios en ese conjunto, suficiente para muestrear genuinos e
impostores con variedad.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model


# ─────────────────────────────────────────────────────────────────
# 1. GENERACIÓN DE TRIALS
# ─────────────────────────────────────────────────────────────────

def generar_trials(y_locutor, y_pin, n_genuinos=2000, n_impostores=2000,
                   random_state=42):
    """
    Genera una lista de trials para evaluar el sistema de verificación,
    cubriendo los cuatro casos del enunciado.

    Cada trial es una tupla (i_audio, locutor_reclamado, frase_reclamada,
    es_genuino, tipo):

      - i_audio            : índice del audio en el conjunto de test.
      - locutor_reclamado  : etiqueta de locutor que se afirma.
      - frase_reclamada    : etiqueta de frase que se afirma.
      - es_genuino         : True si la afirmación es cierta.
      - tipo               : etiqueta del caso ("a", "b", "c", "d") para
                             análisis posterior.

    Tipos de trial:
      - "a" GENUINO     : el locutor reclamado y la frase reclamada
                          coinciden con la realidad del audio. ACEPTAR.
      - "b" FRASE       : el locutor coincide, la frase NO. RECHAZAR.
                          (Mide la Red 2: ¿detecta la frase incorrecta?)
      - "c" LOCUTOR     : el locutor NO coincide, la frase SÍ. RECHAZAR.
                          (Mide la Red 1: ¿detecta el impostor?)
      - "d" AMBOS       : ni locutor ni frase coinciden. RECHAZAR.
                          (Caso más fácil: ambas redes deberían rechazar.)

    Los tres tipos de impostor se reparten uniformemente dentro del
    presupuesto total n_impostores.

    Args:
        y_locutor    : etiquetas reales de locutor (test set)
        y_pin        : etiquetas reales de frase (test set)
        n_genuinos   : número de trials genuinos (tipo "a") a generar
        n_impostores : número total de trials impostores (b + c + d)
        random_state : semilla para reproducibilidad

    Returns:
        trials : lista de tuplas (i_audio, loc, frase, es_genuino, tipo)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_locutor)
    locutores_unicos = np.unique(y_locutor)
    frases_unicas    = np.unique(y_pin)
    trials = []

    # ── Tipo "a" — GENUINOS (locutor correcto, frase correcta) ──
    idx_a = rng.choice(np.arange(n), size=n_genuinos, replace=True)
    for i in idx_a:
        trials.append((int(i), int(y_locutor[i]), int(y_pin[i]),
                       True, "a"))

    # ── Reparto del presupuesto de impostores entre b, c, d ──
    n_b = n_impostores // 3
    n_c = n_impostores // 3
    n_d = n_impostores - n_b - n_c   # el resto va al tipo d

    # ── Tipo "b" — Mismo locutor, frase incorrecta ──
    idx_b = rng.choice(np.arange(n), size=n_b, replace=True)
    for i in idx_b:
        frase_real = int(y_pin[i])
        frase_falsa = int(rng.choice(frases_unicas[frases_unicas != frase_real]))
        trials.append((int(i), int(y_locutor[i]), frase_falsa,
                       False, "b"))

    # ── Tipo "c" — Otro locutor, frase correcta ──
    idx_c = rng.choice(np.arange(n), size=n_c, replace=True)
    for i in idx_c:
        loc_real = int(y_locutor[i])
        loc_falso = int(rng.choice(locutores_unicos[locutores_unicos != loc_real]))
        trials.append((int(i), loc_falso, int(y_pin[i]),
                       False, "c"))

    # ── Tipo "d" — Otro locutor, frase incorrecta ──
    idx_d = rng.choice(np.arange(n), size=n_d, replace=True)
    for i in idx_d:
        loc_real    = int(y_locutor[i])
        frase_real  = int(y_pin[i])
        loc_falso   = int(rng.choice(locutores_unicos[locutores_unicos != loc_real]))
        frase_falsa = int(rng.choice(frases_unicas[frases_unicas != frase_real]))
        trials.append((int(i), loc_falso, frase_falsa,
                       False, "d"))

    rng.shuffle(trials)

    # ── Resumen ──
    from collections import Counter
    cuentas = Counter(t[4] for t in trials)
    print(f"Trials generados: {len(trials)} total")
    print(f"  (a) Genuino                       : {cuentas['a']}")
    print(f"  (b) Mismo locutor, frase falsa    : {cuentas['b']}")
    print(f"  (c) Otro locutor, frase correcta  : {cuentas['c']}")
    print(f"  (d) Otro locutor, frase falsa     : {cuentas['d']}")
    return trials


# ─────────────────────────────────────────────────────────────────
# 2. SCORES — ENFOQUE 1: PROBABILIDADES SOFTMAX
# ─────────────────────────────────────────────────────────────────

def scores_probabilidades(red1, red2, X, trials):
    """
    Calcula el score de cada trial fusionando las probabilidades softmax.

    Para cada trial (i, L, F, _):
        score = P(L | audio_i, Red1) × P(F | audio_i, Red2)

    La fusión por PRODUCTO equivale a un AND lógico suave: el sistema
    está "contento" solo si AMBAS redes apoyan la afirmación. Una
    alternativa común es la suma (OR lógico suave); el producto es más
    estricto y suele dar mejor EER.

    Args:
        red1, red2 : modelos entrenados
        X          : audios de test (N, 120, 63, 1)
        trials     : lista de tuplas (i, loc, frase, es_genuino, tipo)

    Returns:
        scores : np.ndarray (n_trials,) con el score de cada trial
        labels : np.ndarray (n_trials,) con 1=genuino, 0=impostor
    """
    # Calculamos las probabilidades de TODOS los audios una sola vez,
    # luego indexamos por trial. Más rápido que pasar la red por trial.
    print("Calculando probabilidades de la Red 1...")
    probs1 = red1.predict(X, verbose=0)   # (N, 50)
    print("Calculando probabilidades de la Red 2...")
    probs2 = red2.predict(X, verbose=0)   # (N, 5)

    scores, labels = [], []
    for i, loc, frase, es_genuino, tipo in trials:
        p_loc = probs1[i, loc]
        p_fr  = probs2[i, frase]
        scores.append(p_loc * p_fr)
        labels.append(1 if es_genuino else 0)

    return np.array(scores), np.array(labels)


# ─────────────────────────────────────────────────────────────────
# 3. SCORES — ENFOQUE 2: EMBEDDINGS + SIMILITUD COSENO
# ─────────────────────────────────────────────────────────────────

def extraer_embeddings(red, X):
    """
    Extrae los vectores de embedding (capa 'embedding') para cada audio.

    La capa 'embedding' es la penúltima capa Dense (ver model.py): un
    vector de 128 dimensiones que representa una huella compacta del
    audio en el espacio aprendido por la red.

    Returns:
        emb : np.ndarray (N, 128)
    """
    extractor = Model(inputs=red.input,
                      outputs=red.get_layer("embedding").output)
    return extractor.predict(X, verbose=0)


def centroides_por_clase(embeddings, etiquetas, n_clases):
    """
    Calcula el embedding promedio de cada clase.

    El centroide es la "huella prototípica" de la clase. En verificación,
    para decidir si un audio nuevo pertenece a la clase L, se mide cuán
    parecido es su embedding al centroide de L (similitud coseno).

    Returns:
        centroides : np.ndarray (n_clases, dim_embedding)
    """
    dim = embeddings.shape[1]
    centroides = np.zeros((n_clases, dim), dtype=np.float32)
    for c in range(n_clases):
        mask = (etiquetas == c)
        if mask.sum() > 0:
            centroides[c] = embeddings[mask].mean(axis=0)
    return centroides


def coseno(a, b):
    """Similitud coseno entre dos vectores (o entre uno y un batch)."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=-1)


def scores_embeddings(red1, red2, X, y_locutor_train, y_pin_train,
                      X_train, trials, n_loc=50, n_pin=5):
    """
    Calcula el score de cada trial con embeddings + coseno.

    Pasos:
      1. Extrae embeddings de TRAIN con ambas redes.
      2. Calcula el centroide de cada locutor (Red 1) y cada frase
         (Red 2) sobre esos embeddings de train. Los centroides
         representan la huella prototípica de cada clase.
      3. Extrae embeddings de TEST (los audios a verificar).
      4. Para cada trial, score = coseno(emb_test, centroide_loc) ×
                                    coseno(emb_test, centroide_frase).

    Los centroides se calculan sobre TRAIN (no test) para que la
    información que define al locutor/frase no provenga del propio
    conjunto de evaluación: así medimos generalización, no memoria.

    Args:
        red1, red2          : modelos entrenados
        X                   : audios de test
        y_locutor_train,
        y_pin_train         : etiquetas de los audios de train
        X_train             : audios de train (para calcular centroides)
        trials              : lista de trials
        n_loc, n_pin        : nº de clases de locutor y frase

    Returns:
        scores, labels  (igual formato que scores_probabilidades)
    """
    print("Extrayendo embeddings de train (Red 1)...")
    emb1_train = extraer_embeddings(red1, X_train)
    print("Extrayendo embeddings de train (Red 2)...")
    emb2_train = extraer_embeddings(red2, X_train)

    cent_loc = centroides_por_clase(emb1_train, y_locutor_train, n_loc)
    cent_pin = centroides_por_clase(emb2_train, y_pin_train, n_pin)

    print("Extrayendo embeddings de test (Red 1)...")
    emb1_test = extraer_embeddings(red1, X)
    print("Extrayendo embeddings de test (Red 2)...")
    emb2_test = extraer_embeddings(red2, X)

    scores, labels = [], []
    for i, loc, frase, es_genuino, tipo in trials:
        s_loc = coseno(emb1_test[i], cent_loc[loc])
        s_fr  = coseno(emb2_test[i], cent_pin[frase])
        # Coseno está en [-1, 1]; lo desplazamos a [0, 2] antes de
        # multiplicar para que el producto tenga sentido (sin que un
        # coseno negativo invierta el signo final).
        scores.append((s_loc + 1) * (s_fr + 1))
        labels.append(1 if es_genuino else 0)

    return np.array(scores), np.array(labels)


# ─────────────────────────────────────────────────────────────────
# 4. MÉTRICAS — EER Y CURVA DET
# ─────────────────────────────────────────────────────────────────

def calcular_eer_y_det(scores, labels):
    """
    Calcula el EER (Equal Error Rate) y los datos de la curva DET.

    Procedimiento:
      1. Ordenamos los scores y barremos cada uno como umbral.
      2. Para cada umbral calculamos:
           FAR = nº impostores con score >= umbral / total impostores
           FRR = nº genuinos    con score <  umbral / total genuinos
      3. El EER es el punto donde FAR == FRR (en la práctica, donde su
         diferencia cambia de signo: interpolamos).

    Args:
        scores : np.ndarray (n_trials,)
        labels : np.ndarray (n_trials,) — 1=genuino, 0=impostor

    Returns:
        eer       : float, equal error rate en [0, 1]
        umbral_eer: float, valor del score donde se alcanza el EER
        far       : np.ndarray, false acceptance rate por umbral
        frr       : np.ndarray, false rejection rate por umbral
        umbrales  : np.ndarray, los umbrales evaluados
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_gen = (labels == 1).sum()
    n_imp = (labels == 0).sum()

    # Barremos todos los scores únicos como umbrales (ordenados)
    umbrales = np.sort(np.unique(scores))
    far = np.zeros(len(umbrales))
    frr = np.zeros(len(umbrales))
    for k, u in enumerate(umbrales):
        far[k] = ((scores >= u) & (labels == 0)).sum() / max(n_imp, 1)
        frr[k] = ((scores <  u) & (labels == 1)).sum() / max(n_gen, 1)

    # EER: punto donde |FAR - FRR| es mínimo. Interpolamos para precisión.
    diff = far - frr
    k_eer = np.argmin(np.abs(diff))
    eer = (far[k_eer] + frr[k_eer]) / 2
    umbral_eer = umbrales[k_eer]

    return eer, umbral_eer, far, frr, umbrales


def graficar_det(resultados_dict, titulo="Curva DET", guardar=None):
    """
    Dibuja la(s) curva(s) DET. resultados_dict es:
        {"nombre_sistema": (far, frr, eer), ...}

    Una curva DET muestra FAR (eje X) frente a FRR (eje Y) en escala
    logarítmica. Cuanto más cerca del origen pasa la curva, mejor el
    sistema. El EER es el punto donde la curva cruza la diagonal x=y.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    for nombre, (far, frr, eer) in resultados_dict.items():
        ax.plot(far * 100, frr * 100,
                label=f"{nombre}  (EER = {eer*100:.2f}%)")

    # Diagonal de referencia FAR = FRR
    ax.plot([0.01, 100], [0.01, 100], "k:", alpha=0.4, label="FAR = FRR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.1, 100)
    ax.set_xlabel("Falsa Aceptación (FAR) [%]")
    ax.set_ylabel("Falso Rechazo (FRR) [%]")
    ax.set_title(titulo)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
        print(f"DET guardada en {guardar}")
    plt.show()