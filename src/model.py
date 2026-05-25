"""
model.py — Arquitecturas de red para verificación de locutor
=============================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

Este módulo define DOS arquitecturas, ambas genéricas (parametrizadas
por el número de clases), de modo que sirven tanto para la Red 1
(identificación de locutor, 50 clases) como para la Red 2
(verificación de frase, 5 clases):

  1) construir_cnn()       — CNN 2-D pura.
  2) construir_cnn_lstm()  — CNN + BiLSTM + atención.

¿Cuál usar para qué? (justificación)
  - Red 1 (LOCUTOR): la identidad de un locutor está sobre todo en el
    timbre y los formantes, propiedades casi instantáneas del espectro.
    Una CNN las captura bien y, al ser más simple, sobreajusta menos
    (importante con ~110 audios por locutor). -> empezar con la CNN.
  - Red 2 (FRASE): una frase es una secuencia ORDENADA de fonemas
    en el tiempo. La parte LSTM + atención modela esa evolución
    temporal. -> la CNN-LSTM encaja especialmente aquí.

El modelo expone una capa con name='embedding' (Dense de 128
unidades) para poder extraerla posteriormente en la fase de
verificación mediante get_layer('embedding'), útil tanto para el
enfoque por probabilidad softmax como para similitud coseno.

Adaptado de los scripts de referencia del profesor
(reconocedorLocutor.py y reconocedorLocutor2.py).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO DE LOS MODELOS
# ─────────────────────────────────────────────────────────────────

CONFIG_MODELO = {
    "dropout":     0.4,
    "l2_reg":      1e-4,

    # Específico de la CNN-LSTM
    "lstm_units":  [128, 64],   # unidades por capa LSTM
    "bidireccional": True,      # usar BiLSTM
}


# ─────────────────────────────────────────────────────────────────
# 1. CNN 2-D PURA
# ─────────────────────────────────────────────────────────────────

def construir_cnn(input_shape, n_clases, cfg=None, nombre="CNN"):
    """
    CNN 2-D para clasificación sobre espectrogramas MFCC.

    Tres bloques convolucionales (Conv + BatchNorm + ReLU + MaxPool)
    que aprenden filtros espectrales locales, seguidos de
    GlobalAveragePooling y un clasificador denso.

    El GlobalAveragePooling, en lugar de aplanar, hace al modelo
    robusto a pequeñas variaciones de la dimensión temporal y reduce
    mucho el nº de parámetros (menos sobreajuste).

    Args:
        input_shape : forma de entrada (n_caract, T, 1)
        n_clases    : nº de clases de salida (50 locutores / 5 frases)
        cfg         : dict de hiperparámetros; si None usa CONFIG_MODELO
        nombre      : nombre del modelo

    Returns:
        modelo de Keras SIN compilar (la compilación se hace en train.py)
    """
    cfg  = cfg or CONFIG_MODELO
    reg  = regularizers.l2(cfg["l2_reg"])
    drop = cfg["dropout"]

    inp = layers.Input(shape=input_shape, name="entrada_mfcc")

    # ── Bloque 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(drop / 2)(x)

    # ── Bloque 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(drop / 2)(x)

    # ── Bloque 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # ── Embedding: vector de características del audio.
    #    Se le pone nombre para poder extraerlo en la Fase 7.
    emb = layers.Dense(128, activation="relu", kernel_regularizer=reg,
                       name="embedding")(x)
    x   = layers.Dropout(drop)(emb)

    # ── Clasificador
    salida = layers.Dense(n_clases, activation="softmax", name="salida")(x)

    return models.Model(inp, salida, name=nombre)


# ─────────────────────────────────────────────────────────────────
# 2. CNN-LSTM CON ATENCIÓN
# ─────────────────────────────────────────────────────────────────

class AtencionBahdanau(layers.Layer):
    """
    Mecanismo de atención aditiva (Bahdanau) sobre la secuencia LSTM.

    Dado el tensor de estados ocultos h de forma (batch, T, unidades),
    calcula un vector de contexto como suma ponderada de h. Los pesos
    se aprenden end-to-end: la red decide qué instantes de la frase son
    más discriminativos.
    """

    def __init__(self, unidades, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(unidades)   # proyección de estados ocultos
        self.V = layers.Dense(1)          # score escalar

    def call(self, h):
        # h: (batch, T, unidades_lstm)
        score   = self.V(tf.nn.tanh(self.W(h)))      # (batch, T, 1)
        pesos   = tf.nn.softmax(score, axis=1)        # (batch, T, 1)
        contexto = tf.reduce_sum(pesos * h, axis=1)   # (batch, unidades_lstm)
        return contexto

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"unidades": self.W.units})
        return cfg


def construir_cnn_lstm(input_shape, n_clases, cfg=None, nombre="CNN_LSTM"):
    """
    Modelo híbrido CNN-LSTM con atención.

    Arquitectura:
        [CNN local] extrae mapas de características 2-D
          -> reshape a secuencia temporal
          -> [BiLSTM] modela dependencias temporales
          -> [Atención] pondera los instantes más informativos
          -> Dense -> Softmax

    Razonamiento:
      - La CNN aprende filtros espectrales locales (timbre, formantes).
      - La LSTM modela la evolución temporal de esos filtros (el orden
        de los fonemas, el ritmo del habla).
      - La atención decide qué instantes pesan más en la decisión.

    Nota: el pooling de la CNN comprime SOLO el eje de frecuencia
    (pool=(2,1)) para conservar intacto el eje temporal, que es lo que
    luego procesa la LSTM.

    Args/Returns: igual que construir_cnn().
    """
    cfg   = cfg or CONFIG_MODELO
    reg   = regularizers.l2(cfg["l2_reg"])
    drop  = cfg["dropout"]
    units = cfg["lstm_units"]
    bidir = cfg["bidireccional"]

    inp = layers.Input(shape=input_shape, name="entrada_mfcc")

    # ── Extractor CNN local (pool solo en frecuencia)
    def bloque_conv(x, filtros):
        x = layers.Conv2D(filtros, (3, 3), padding="same",
                          kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 1))(x)   # comprime SOLO frecuencia
        return layers.Dropout(drop / 2)(x)

    x = bloque_conv(inp, 32)
    x = bloque_conv(x,   64)
    x = bloque_conv(x,  128)

    # ── Reshape: (batch, freq_red, T, filtros) -> (batch, T, caract)
    freq_red = x.shape[1]    # frecuencia residual tras los pools
    seq_len  = x.shape[2]    # dimensión temporal (conservada)
    n_filt   = x.shape[3]    # nº de filtros

    x = layers.Permute((2, 1, 3))(x)                      # (b, T, f, c)
    x = layers.Reshape((seq_len, freq_red * n_filt))(x)   # (b, T, caract)

    # ── Capas LSTM (bidireccionales) sobre la secuencia temporal
    for i, u in enumerate(units):
        capa_lstm = layers.LSTM(
            u,
            return_sequences=True,
            dropout=drop / 2,
            kernel_regularizer=reg,
            name=f"lstm_{i+1}",
        )
        x = (layers.Bidirectional(capa_lstm, name=f"bilstm_{i+1}")(x)
             if bidir else capa_lstm(x))
        x = layers.LayerNormalization()(x)

    # ── Atención: colapsa la secuencia en un único vector de contexto
    unidades_att = units[-1] * (2 if bidir else 1)
    x = AtencionBahdanau(unidades_att, name="atencion")(x)

    # ── Embedding + clasificador
    emb    = layers.Dense(128, activation="relu", kernel_regularizer=reg,
                          name="embedding")(x)
    x      = layers.Dropout(drop)(emb)
    salida = layers.Dense(n_clases, activation="softmax", name="salida")(x)

    return models.Model(inp, salida, name=nombre)


# ─────────────────────────────────────────────────────────────────
# 3. SELECTOR
# ─────────────────────────────────────────────────────────────────

def construir_modelo(arquitectura, input_shape, n_clases, cfg=None):
    """
    Devuelve un modelo según la arquitectura pedida.

    Args:
        arquitectura : "cnn" o "cnn_lstm"
        input_shape  : (n_caract, T, 1)
        n_clases     : nº de clases de salida
        cfg          : hiperparámetros opcionales

    Returns:
        modelo de Keras sin compilar
    """
    arquitectura = arquitectura.lower()
    if arquitectura == "cnn":
        return construir_cnn(input_shape, n_clases, cfg)
    elif arquitectura == "cnn_lstm":
        return construir_cnn_lstm(input_shape, n_clases, cfg)
    else:
        raise ValueError(
            f"Arquitectura desconocida: '{arquitectura}'. "
            f"Opciones válidas: 'cnn', 'cnn_lstm'."
        )


# ─────────────────────────────────────────────────────────────────
# Prueba rápida del módulo (python model.py)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Prueba del módulo model.py ===\n")

    # Forma de entrada de ejemplo: 120 características (40 MFCC x3 con
    # deltas), ~63 frames temporales para 2 s de audio, 1 canal.
    forma_ejemplo = (120, 63, 1)

    for arq in ["cnn", "cnn_lstm"]:
        print(f"\n--- Arquitectura: {arq} (50 clases) ---")
        modelo = construir_modelo(arq, forma_ejemplo, n_clases=50)
        modelo.summary()

    print("\nOK: el módulo funciona correctamente.")
