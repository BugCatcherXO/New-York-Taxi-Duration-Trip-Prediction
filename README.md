# NYC Taxi Trip Duration — Kaggle

### Resumen
Este repositorio contiene una solución simple y reproducible para la competición “NYC Taxi Trip Duration” de Kaggle.
Objetivo: predecir la duración de viajes de taxi en segundos a partir de información temporal y geoespacial.
Métrica oficial: RMSLE.
Competición: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview


### Datos
Kaggle provee dos archivos principales: train.csv y test.csv. El conjunto de train incluye el objetivo trip_duration y la fecha de recogida. Test no tiene trip_duration.
Columnas usadas: id, vendor_id, pickup_datetime, passenger_count, store_and_fwd_flag, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, dropoff_datetime (solo en train).

### Enfoque
1) Carga sin lógica compleja y conversión de fechas.
2) Ingeniería de variables mínima y de alto impacto:
   - Tiempo: pickup_hour, pickup_dow (0=Lunes), pickup_month, is_weekend.
   - Geo: distancias haversine_km y manhattan_km aproximadas.
   - Categóricas: store_and_fwd_flag a binaria y one‑hot de vendor_id.
   - Se eliminan id, dropoff_datetime y pickup_datetime tras extraer las features.
3) Preparación del conjunto:
   - División aleatoria 98%/2% con semilla 42.
   - Imputación con la mediana calculada en train.
   - Recorte por percentiles 1 y 99 para estabilizar outliers.
   - Estandarización z‑score usando medias y desviaciones de train.
4) Modelado (PyTorch MLP en espacio log1p del objetivo):
   - Arquitectura: [input] → 128 → ReLU → Dropout 0.1 → 64 → ReLU → Dropout 0.1 → 1.
   - Optimización: Adam, lr 1e‑3, weight_decay 1e‑4, batch_size 4096, 8 épocas.
   - Pérdida: MSE sobre y = log1p(trip_duration). La métrica de validación se reporta como RMSLE = sqrt(MSE).
5) Predicción y envío:
   - Se aplica expm1 a las predicciones, con clip mínimo 1.0.
   - Se genera submission.csv con las columnas id y trip_duration.

### Resultados
Validación interna (2% hold‑out):
Epoch 01  val_RMSLE ≈ 0.7079
Epoch 02  val_RMSLE ≈ 0.5455
Epoch 03  val_RMSLE ≈ 0.5101
Epoch 04  val_RMSLE ≈ 0.4983
Epoch 05  val_RMSLE ≈ 0.4900
Epoch 06  val_RMSLE ≈ 0.4846
Epoch 07  val_RMSLE ≈ 0.4818
Epoch 08  val_RMSLE ≈ 0.4771  ← final

### Puntuación Kaggle
- Public Score: 0.48353
- Private Score: 0.48111

### Requisitos
- Python 3.10+
- pandas, numpy, matplotlib, torch

### Cómo reproducir
1) Descargar los datos de la competición y colocarlos en datasets/train, datasets/test y datasets/sample_submission.
2) Abrir 01_proyecto.ipynb y ejecutar todas las celdas en orden.
3) Al finalizar se crea submission.csv con 625134 filas.
4) Para enviar a Kaggle, comprimir en submission.zip si es necesario y subirlo al panel de Submissions de la competición.

###  Notas técnicas
- El objetivo se modela en log1p para alinear con la RMSLE y reducir el efecto de colas largas.
- La normalización se ajusta solo con estadísticas del subconjunto de train.
- El pipeline evita condicionales innecesarios y usa nombres de columna fijos para claridad y simplicidad.



### Agradecimientos
Datos y definición del problema por la comunidad de Kaggle.
