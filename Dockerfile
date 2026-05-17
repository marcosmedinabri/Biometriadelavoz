# Imagen oficial de TensorFlow con soporte GPU. El tag fijo (no "latest")
# es lo que garantiza reproducibilidad. Verifica el tag más reciente en
# hub.docker.com/r/tensorflow/tensorflow/tags
FROM tensorflow/tensorflow:2.17.0-gpu

# Dependencias de sistema para procesar audio:
# - ffmpeg y libsndfile1: backends que librosa necesita para leer/decodificar WAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--ServerApp.token="]
