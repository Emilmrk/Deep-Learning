import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil
from tqdm import tqdm

IMG_SIZE = (48, 48)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
BATCH_SIZE = 32

model = tf.keras.models.load_model("fer2013_vgg_finetuned_model.h5")

carpeta_crudo = '../fotos'
carpeta_salida = '.'
carpeta_etiquetados = os.path.join(carpeta_salida, 'etiquetados')

if os.path.exists(carpeta_etiquetados):
    shutil.rmtree(carpeta_etiquetados)
os.makedirs(carpeta_etiquetados, exist_ok=True)

resultados = []
imagenes_crudo = [f for f in os.listdir(carpeta_crudo) if f.endswith('.jpg') or f.endswith('.png')]

for img_name in tqdm(imagenes_crudo, desc="Etiquetando imágenes VGG"):
    img_path = os.path.join(carpeta_crudo, img_name)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    class_index = np.argmax(pred)
    resultados.append((img_name, CLASS_NAMES[class_index], pred.tolist()))

    clase_dir = os.path.join(carpeta_etiquetados, CLASS_NAMES[class_index])
    os.makedirs(clase_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(clase_dir, img_name))

df_resultados = pd.DataFrame(resultados, columns=["Imagen", "Prediccion", "Probabilidades"])
df_resultados.to_csv(os.path.join(carpeta_salida, "clasificaciones_dataset_crudo_vgg.csv"), index=False)

conteo_clases = df_resultados["Prediccion"].value_counts().sort_index()
df_conteo = pd.DataFrame({"Emocion": conteo_clases.index, "Cantidad": conteo_clases.values})
df_conteo.to_csv(os.path.join(carpeta_salida, "conteo_emociones_vgg.csv"), index=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Emocion", y="Cantidad", data=df_conteo)
plt.title("Distribución de Emociones Detectadas VGG")
plt.ylabel("Cantidad")
plt.xlabel("Emocion")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "grafico_conteo_emociones_vgg.png"))
plt.close()
