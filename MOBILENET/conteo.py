import os
import pandas as pd

# Carpeta donde están las imágenes etiquetadas por clase
carpeta_etiquetadas = 'etiquetadas'  # asegúrate de estar en la carpeta del modelo

conteo = {}

# Recorremos cada subcarpeta (una por clase)
for clase in os.listdir(carpeta_etiquetadas):
    ruta_clase = os.path.join(carpeta_etiquetadas, clase)
    if os.path.isdir(ruta_clase):
        cantidad = len([
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        conteo[clase] = cantidad

# Guardamos como CSV
df = pd.DataFrame(list(conteo.items()), columns=['clase', 'cantidad'])
df.to_csv('conteo_etiquetas.csv', index=False)

print("✅ Conteo generado correctamente.")
print(df)
