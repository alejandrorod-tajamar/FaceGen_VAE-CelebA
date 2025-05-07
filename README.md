# Generación de Imágenes de Rostros con VAE y el Dataset CelebA

Este proyecto implementa un modelo de Autoencoder Variacional (VAE) para generar imágenes de rostros utilizando el dataset CelebA. A través de este proyecto, aprenderás a entrenar un modelo generativo, explorar el espacio latente y evaluar la calidad de las imágenes generadas.

## Requisitos Previos

1. **Python 3.8 o superior**
2. **GPU compatible con CUDA** (opcional, pero recomendado para acelerar el entrenamiento)
3. **Bibliotecas necesarias**:
   - TensorFlow
   - NumPy
   - Matplotlib
   - scikit-learn (opcional, para visualización del espacio latente)

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/alejandrorod-tajamar/FaceGen_VAE-CelebA.git
   ```

2. Cambia de directorio a la carpeta principal del proyecto:
   ```bash
   cd FaceGen_VAE-CelebA
   ```

3. Ejecuta el _script_ de inicio:
   ```bash
   ./01-setup/01-setup_script.sh
   ```

4. Instala los requisitos de inicio del proyecto:
   ```bash
   pip install -r 01-setup/02-setup_requirements.txt
   ```

5. Ejecuta el notebook [GPU Check](01-setup/03-gpu_check.ipynb) para comprobar si la GPU está disponible y está siendo utilizada por **TensorFlow**.

## Ejecución del Proyecto

1. Consulta la [guía](02-notebook/01-guide.md) en la que se ha basado este proyecto.

2. Instala los requisitos globales del proyecto:
   ```bash
   pip install -r 02-notebook/02-global_requirements.txt
   ```

3. Ejecuta el notebook [FaceImaGeneration](02-notebook/03-FaceImaGeneration.ipynb) y sigue todos los pasos. Observa los resultados de las celdas.

## Resultados

Los resultados incluyen:
- Imágenes generadas durante el entrenamiento.
- Comparaciones entre imágenes originales y reconstruidas.
- Visualizaciones del espacio latente.

## Estructura del Proyecto

```
CasoEstudio-GenerarRostros/
├── 01-setup/                # Scripts de configuración inicial
├── 02-notebook/             # Notebook principal
├── model/                   # Resultados generados por el modelo
│   ├── generated_faces/     # Imágenes generadas
│   ├── logs/                # Registros
|       └── vae_training/    # Entrenamiento del VAE
|           ├── train/       # Entrenamiento
|           └── validation/  # Validación
│   └── vae_checkpoints/     # Pesosfinales del modelo entrenado
└── README.md                # Este archivo
```

## Notas Adicionales

- Si encuentras problemas durante el entrenamiento, consulta la sección de "Pistas para Problemas Comunes" en el archivo [02-notebook/01-guide.md](02-notebook/01-guide.md).
- Para más información sobre el modelo VAE, consulta la [documentación oficial de Keras](https://keras.io/examples/generative/vae/).

## Créditos

- [Enunciado del proyecto](https://github.com/tamasma/master-ia-tajamar/blob/main/ia-generativa-desarrollo/deep-learning/autoencoders/practica-autoencoders-2.md)
- [Dataset en Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)