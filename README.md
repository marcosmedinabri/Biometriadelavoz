```
O---o
 O-o    ________        __    __                        
  O    /  _____/_____ _/  |__/  |______    ____ _____   
 o-O  /   \  ___\__  \\   __\   __\__  \ _/ ___\\__  \  
o---O \    \_\  \/ __ \|  |  |  |  / __ \\  \___ / __ \_
O---o  \______  (____  /__|  |__| (____  /\___  >____  /
 O-o          \/     \/                \/     \/     \/
  O
 o-O       Text-Dependent Speaker Verification System
o---O
O---o
 O-o
  O
 o-O
o---O
```

> *”…there is no gene for the human spirit.”*

Sistema de verificación biométrica de locutor dependiente de texto, desarrollado como proyecto de la asignatura **Biometría de la Voz**. Dado un audio y una afirmación del tipo *“soy el locutor L diciendo la frase F”*, el sistema decide si acepta o rechaza la identidad declarada.

-----

## ¿Cómo funciona?

El sistema combina dos redes neuronales especializadas cuyas salidas se fusionan para tomar la decisión de verificación:

|Red      |Tarea                    |Clases          |Accuracy (test)|
|---------|-------------------------|----------------|---------------|
|**Red 1**|Identificación de locutor|50 locutores    |**69,83 %**    |
|**Red 2**|Identificación de frase  |5 frases (S1–S5)|**92,17 %**    |

La fusión se realiza mediante dos enfoques alternativos:

- **Producto de probabilidades softmax** — fusión a nivel de puntuación
- **Similitud coseno sobre embeddings** — fusión a nivel de representación

### Rendimiento de verificación (EER)

|Configuración                     |EER       |
|----------------------------------|----------|
|Fusión de probabilidades          |5,55 %    |
|Embeddings + coseno *(mejor caso)*|**5,35 %**|

**EER desglosado por tipo de impostor** *(mejor enfoque por caso)*:

|Caso|Descripción                                      |EER   |
|----|-------------------------------------------------|------|
|(b) |Mismo locutor, frase incorrecta — *mide Red 2*   |3,30 %|
|(c) |Locutor distinto, frase correcta — *mide Red 1*  |6,03 %|
|(d) |Locutor distinto, frase incorrecta — *caso fácil*|0,87 %|

-----

## Estructura del repositorio

```
gattaca/
├── orquestador.ipynb              Notebook principal — pipeline completo
├── src/
│   ├── data.py                    Carga de audio, extracción de MFCC, splits train/val/test
│   ├── augment.py                 Aumento de datos (Red 1)
│   ├── model.py                   Arquitecturas CNN y CNN-LSTM (parametrizables)
│   ├── train.py                   Entrenamiento y evaluación genéricos
│   └── verificacion.py            Generación de trials, scores, cálculo de EER y curva DET
├── resultados/                    Modelos entrenados (.keras) y figuras generadas
├── requirements.txt               Dependencias Python
├── Dockerfile                     Imagen del entorno de ejecución
└── docker-compose.yml             Orquestación del contenedor
```

> ⚠️ La base de datos `TextDependentSpeakerIdentification/` (~16 GB) **no se incluye** en el repositorio. Debe colocarse en la raíz del proyecto antes de ejecutar el notebook.

-----

## Requisitos

- **Docker** y **Docker Compose**
- **NVIDIA Container Toolkit** — para acceso a GPU desde el contenedor
- **GPU NVIDIA con soporte CUDA** — probado en RTX 3060 Laptop (6 GB VRAM)

-----

## Puesta en marcha

**1.** Coloca la base de datos en la raíz del proyecto:

```
gattaca/
└── TextDependentSpeakerIdentification/
```

**2.** Levanta el contenedor:

```bash
docker compose up
```

**3.** Abre Jupyter Lab en `http://localhost:8888` y abre `orquestador.ipynb`.

**4.** Ejecuta el pipeline completo:

```
Kernel → Restart Kernel and Run All Cells
```

> 💡 **Primera ejecución:** ~40 minutos (entrena ambas redes desde cero).  
> **Ejecuciones siguientes:** ~5 minutos — el notebook detecta los modelos en `resultados/` y los carga directamente.

-----

## Documentación

- **Memoria del proyecto:** `memoria.docx` — entregable principal con descripción completa del sistema, experimentos y análisis de resultados.
- **Notebook orquestador:** `orquestador.ipynb` — pipeline comentado paso a paso con todas las figuras generadas.

-----

## Autores

|Nombre                |
|----------------------|
|Alberto Buceta Coroba |
|Marcos Medina Brihuega|
|Carlos Sainz Sueiro   |

-----

## Licencia y uso

Trabajo académico realizado para la asignatura **Biometría de la Voz** — Grado en Ingeniería Informática, ETSII-UPM.

El código se publica con fines de consulta y referencia académica. Los datos de voz son propiedad de sus autores originales y no se redistribuyen en este repositorio.

*Este trabajo ha hecho uso puntual de asistentes de IA durante las fases de desarrollo y documentación.*