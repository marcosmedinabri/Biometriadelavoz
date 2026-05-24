# Verificación de locutor de texto dependiente

Proyecto de la asignatura **Biometría de la Voz**. Sistema de verificación
biométrica que, dado un audio y una afirmación del tipo *"soy el locutor L
diciendo la frase F"*, decide si la acepta o la rechaza.

El sistema combina dos redes neuronales:
- **Red 1**: identifica al locutor (50 clases, accuracy 72,21 %).
- **Red 2**: identifica la frase pronunciada (5 clases S1–S5, accuracy 92,17 %).

La verificación se resuelve fusionando las salidas de ambas redes mediante
dos enfoques alternativos (producto de probabilidades softmax y similitud
coseno sobre embeddings), alcanzando un **EER del 5,35 %** en el mejor caso.

## Autores

- Alberto Buceta Coroba
- Marcos Medina Brihuega
- Carlos Sainz Sueiro

## Estructura del proyecto

```
.
├── orquestador.ipynb         Notebook principal (todo el pipeline)
├── src/
│   ├── data.py               Carga de audio, MFCC, splits train/val/test
│   ├── augment.py            Aumento de datos para la Red 1
│   ├── model.py              Arquitecturas CNN y CNN-LSTM (parametrizables)
│   ├── train.py              Entrenamiento y evaluación genéricos
│   └── verificacion.py       Trials, scores, EER, curva DET
├── resultados/               Modelos .keras y figuras generadas
├── requirements.txt          Dependencias Python
├── Dockerfile                Imagen del entorno
└── docker-compose.yml        Orquestación del contenedor
```

La base de datos `TextDependentSpeakerIdentification/` no se incluye en el
repositorio (16 GB). Debe colocarse en la raíz del proyecto antes de
ejecutar el notebook.

## Requisitos

- Docker y Docker Compose instalados.
- NVIDIA Container Toolkit (para acceso a GPU desde el contenedor).
- GPU NVIDIA con soporte CUDA. Probado en RTX 3060 Laptop (6 GB).

## Ejecución

1. Colocar la base de datos `TextDependentSpeakerIdentification/` en la raíz.
2. Levantar el contenedor:
```bash
   docker compose up
```
3. Abrir Jupyter Lab en `http://localhost:8888` y abrir `orquestador.ipynb`.
4. *Kernel → Restart Kernel and Run All Cells*.

La primera ejecución completa tarda ~40 minutos (entrenamiento de las dos
redes). Las siguientes son de 5 minutos: el notebook detecta los modelos
entrenados en `resultados/` y los carga directamente.

## Resultados principales

| Métrica | Valor |
|--------|-------|
| Red 1 — accuracy de test (50 locutores) | 72,21 % |
| Red 2 — accuracy de test (5 frases)     | 92,17 % |
| Verificación, fusión de probabilidades — EER | 5,55 % |
| Verificación, embeddings + coseno — EER      | 5,35 % |

EER desglosado por tipo de impostor (mejor enfoque por caso):

| Caso | Descripción | EER |
|------|--------------|-----|
| (b) | Mismo locutor + frase falsa  *(mide Red 2)*   | 3,30 % |
| (c) | Otro locutor + frase correcta *(mide Red 1)* | 6,03 % |
| (d) | Otro locutor + frase falsa *(caso fácil)*    | 0,87 % |

## Documentación

- **Memoria**: `memoria.docx` (entregable principal del proyecto).
- **Notebook orquestador**: `orquestador.ipynb`. Incluye los comentarios
  paso a paso del pipeline y todas las figuras generadas.

*Este trabajo ha hecho uso puntual de asistentes de IA durante las
fases de desarrollo y documentación.*

## Licencia y uso

Trabajo académico realizado para la asignatura de Biometría de la Voz
(Máster Universitario en Ingeniería de Telecomunicación, ETSIT-UPM). El
código se publica con fines de consulta. Los datos de voz son propiedad
de sus autores originales y no se redistribuyen aquí.
