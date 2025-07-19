import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
from PIL import Image

IMG_SIZE = (48, 48)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

if os.path.exists('etiquetadas'):
    shutil.rmtree('etiquetadas')
os.makedirs('etiquetadas', exist_ok=True)

for clase in CLASSES:
    os.makedirs(f'etiquetadas/{clase}', exist_ok=True)

model = load_model('fer2013_mobnetv2_finetuned_extra.h5')

fotos_dir = '../fotos'
resultados = []

for filename in tqdm(os.listdir(fotos_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(fotos_dir, filename)
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img).astype('float32')
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        clase_idx = np.argmax(pred)
        clase_nombre = CLASSES[clase_idx]

        shutil.copy(img_path, f'etiquetadas/{clase_nombre}/{filename}')

        resultados.append({'imagen': filename, 'clase': clase_nombre})

df = pd.DataFrame(resultados)
df.to_csv('resultados_etiquetado_mobnetv2.csv', index=False)

conteo = df['clase'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
conteo.plot(kind='bar', color='lightgreen')
plt.title('Conteo de Emociones Etiquetadas (MobileNetV2)')
plt.xlabel('Emoción')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafico_conteo_etiquetado_mobnetv2.png')
plt.close()
