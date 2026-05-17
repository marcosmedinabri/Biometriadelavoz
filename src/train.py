"""
train.py — Entrenamiento y evaluación de las redes
===================================================

Verificación de locutor de texto dependiente — Biometría de la Voz.

Funciones GENÉRICAS de entrenamiento: sirven igual para la Red 1
(locutor) y la Red 2 (PIN). Lo único que cambia entre una y otra es
el número de clases y el vector de etiquetas; el proceso es idéntico.

Incluye:
  - entrenar()   : compila y entrena un modelo con early stopping.
  - evaluar()    : métricas sobre el conjunto de test.
  - graficar()   : curvas de entrenamiento y matriz de confusión.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────

CONFIG_TRAIN = {
    "batch_size":    32,
    "epochs":        60,
    "learning_rate": 1e-3,
    "patience":      10,        # épocas sin mejora antes de parar
    "results_dir":   "resultados",
}


# ─────────────────────────────────────────────────────────────────
# 1. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────

def entrenar(modelo, X_train, y_train, X_val, y_val, cfg=None,
             ruta_modelo=None):
    """
    Compila y entrena un modelo.

    - Pérdida: sparse_categorical_crossentropy (etiquetas son enteros,
      no one-hot).
    - Optimizador: Adam.
    - EarlyStopping sobre val_accuracy: para cuando la red deja de
      mejorar en validación, y restaura los mejores pesos. Evita
      desperdiciar épocas y mitiga el sobreajuste.
    - ModelCheckpoint: guarda en disco la mejor versión del modelo.

    Args:
        modelo      : modelo de Keras SIN compilar (de model.py)
        X_train, y_train, X_val, y_val : datos de entrenamiento/validación
        cfg         : hiperparámetros; si None usa CONFIG_TRAIN
        ruta_modelo : dónde guardar el mejor modelo (.keras). Si None,
                      no se guarda en disco.

    Returns:
        history : objeto History de Keras (curvas de entrenamiento)
    """
    cfg = cfg or CONFIG_TRAIN

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    lista_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if ruta_modelo:
        os.makedirs(os.path.dirname(ruta_modelo) or ".", exist_ok=True)
        lista_callbacks.append(
            callbacks.ModelCheckpoint(
                ruta_modelo,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            )
        )

    print(f"\nEntrenando '{modelo.name}'...")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")
    print(f"  Batch: {cfg['batch_size']}  |  Épocas máx: {cfg['epochs']}\n")

    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=lista_callbacks,
        verbose=1,
    )

    return history


# ─────────────────────────────────────────────────────────────────
# 2. EVALUACIÓN
# ─────────────────────────────────────────────────────────────────

def evaluar(modelo, X_test, y_test, nombres_clases=None):
    """
    Evalúa el modelo sobre el conjunto de test.

    Args:
        modelo         : modelo entrenado
        X_test, y_test : datos de test
        nombres_clases : lista de nombres legibles para el informe
                         (opcional, p.ej. los nombres de locutor)

    Returns:
        dict con accuracy, pérdida, y_pred y la matriz de confusión
    """
    perdida, acc = modelo.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)

    print(f"\n=== Evaluación en TEST ===")
    print(f"  Pérdida  : {perdida:.4f}")
    print(f"  Accuracy : {acc * 100:.2f}%\n")

    # classification_report necesita nombres como strings
    if nombres_clases is not None:
        nombres_clases = [str(n) for n in nombres_clases]

    informe = classification_report(
        y_test, y_pred,
        target_names=nombres_clases,
        zero_division=0,
    )
    print(informe)

    return {
        "accuracy": acc,
        "perdida":  perdida,
        "y_test":   y_test,
        "y_pred":   y_pred,
        "matriz_confusion": confusion_matrix(y_test, y_pred),
        "informe":  informe,
    }


# ─────────────────────────────────────────────────────────────────
# 3. GRÁFICAS
# ─────────────────────────────────────────────────────────────────

def graficar_entrenamiento(history, titulo="Entrenamiento", guardar=None):
    """
    Dibuja las curvas de accuracy y pérdida (train vs validación).

    Comparar la curva de train con la de validación es la forma
    estándar de detectar sobreajuste: si train sube pero validación
    se estanca o baja, la red está memorizando.
    """
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))

    ejes[0].plot(history.history["accuracy"], label="Train")
    ejes[0].plot(history.history["val_accuracy"], label="Validación",
                 linestyle="--")
    ejes[0].set_title("Accuracy")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Accuracy")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(history.history["loss"], label="Train")
    ejes[1].plot(history.history["val_loss"], label="Validación",
                 linestyle="--")
    ejes[1].set_title("Pérdida")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Loss")
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)

    fig.suptitle(titulo, fontsize=13)
    plt.tight_layout()

    if guardar:
        os.makedirs(os.path.dirname(guardar) or ".", exist_ok=True)
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
        print(f"Curvas guardadas en {guardar}")

    plt.show()


def graficar_matriz_confusion(resultados, nombres_clases=None,
                              titulo="Matriz de confusión", guardar=None):
    """
    Dibuja la matriz de confusión.

    La diagonal son los aciertos; lo de fuera, los errores. Permite ver
    QUÉ clases se confunden entre sí (p.ej. dos locutores con voz
    parecida, o dos PINs acústicamente similares).
    """
    cm = resultados["matriz_confusion"]
    n  = cm.shape[0]

    if nombres_clases is not None:
        nombres_clases = [str(x) for x in nombres_clases]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
    sns.heatmap(
        cm, annot=(n <= 25), fmt="d", cmap="Blues",
        xticklabels=nombres_clases, yticklabels=nombres_clases,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    plt.tight_layout()

    if guardar:
        os.makedirs(os.path.dirname(guardar) or ".", exist_ok=True)
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
        print(f"Matriz guardada en {guardar}")

    plt.show()


# Nota: con 40 clases (Red 2) las anotaciones numéricas dentro de la
# matriz se ven apretadas; por eso 'annot' se desactiva si hay más de
# 25 clases. La diagonal sigue leyéndose bien por color.
