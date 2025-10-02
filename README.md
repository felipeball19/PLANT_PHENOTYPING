 Proyecto: Sistema de Fenotipado de Plantas con Hardware Embebido y Visión por Computadora

Este repositorio contiene el código, datos y resultados asociados al trabajo de grado “Prototipo de sistema para el fenotipado de plantas mediante hardware embebido y visión por computadora”.  
El sistema permite capturar imágenes y datos ambientales de cultivos de tomate y pimentón utilizando una Raspberry Pi y procesarlos mediante técnicas de visión por computadora (OpenCV, PlantCV y SAM) para extraer métricas fenotípicas como área foliar, perímetro, solidez, altura y NDVI.



 Funcionalidades principales
- Adquisición automática de datos:  
  - Imágenes RGB (superior), NIR (NoIR) y laterales.  
  - Lecturas de temperatura, humedad relativa y humedad de suelo.
- rocesamiento de imágenes:  
  - Preprocesamiento (balance de blancos, CLAHE, filtros bilaterales).  
  - Segmentación híbrida (ExG + HSV + SAM).  
  - Cálculo de métricas morfológicas y NDVI.
- Interfaz web interactiva (Gradio):  
  - Visualización de métricas temporales (área, altura, NDVI).  
  - Análisis de imágenes individuales y segmentación asistida.  
- Resultados reproducibles:  
  - Scripts para generar métricas y gráficas de crecimiento.



  Estructura del repositorio

| Carpeta/Archivo | Descripción |
|-----------------|-------------|
| REPORTE_PROPIO.ipynb | Notebook principal de análisis de datos y generación de resultados. |
| REPORTE_PROPIO (3).ipynb | Versión preliminar / respaldo del reporte principal. |
| procesamiento_arriba.py | Script para procesar imágenes RGB superiores. |
| procesamiento_noir.py | Script para procesar imágenes NIR y calcular NDVI. |
| procesamiento_SAM.py | Script para segmentación asistida con Segment Anything Model (SAM). |
| metricas_morfologicas.csv | Resultados de métricas morfológicas (área, perímetro, solidez). |
| metricas_reflectancia.csv | Datos de reflectancia y NDVI calculados. |
| metricas_tallos.csv | Datos de altura y número de tallos obtenidos de imágenes laterales. |
| sensores_ambiente.csv | Lecturas de temperatura y humedad relativa del aire. |
| sensores_humedad.csv | Lecturas de humedad del suelo durante el experimento. |
| prototipo_proyecto.7z | Archivo comprimido con diseños de PCB, estructura y documentación del prototipo. |
| requirements.txt | Lista de dependencias necesarias para ejecutar los scripts y notebooks. |

---

 ⚙️ Requisitos

- Python ≥ 3.10
- Librerías clave:
  bash
  pip install -r requirements.txt
  
  (Incluye: opencv-python, plantcv, gradio, numpy, pandas, matplotlib, scikit-image, etc.)

- Hardware recomendado:
  - Raspberry Pi (Modelo 4B o superior)
  - Sensores DHT22 y YL-69
  - Cámara Raspberry Pi Camera Module 3 + NoIR
  - Iluminación LED controlada



 ▶️ Uso básico

1. Procesar imágenes RGB superiores:
   bash
   python procesamiento_arriba.py
   
2. Procesar imágenes NIR y calcular NDVI:
   bash
   python procesamiento_noir.py
   
3. Segmentación asistida (SAM):
   bash
   python procesamiento_SAM.py
   
4. Analizar datos y generar gráficos finales:
   - Abrir y ejecutar `REPORTE_PROPIO.ipynb` en Jupyter Notebook.



 Datos de ejemplo
- Los CSV incluidos (metricas_morfologicas.csv, metricas_reflectancia.csv, metricas_tallos.csv) contienen resultados obtenidos durante un mes de monitoreo de cultivos de tomate y pimentón bajo invernadero.

 Anexos

En esta carpeta se incluyen las tablas y gráficas completas que complementan el trabajo escrito:

-  [Tabla de comparación métricas morfológicas entre el dataset externo y el propio (https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/Comparacion_metricas_dataset_externo_propio.xlsx)

-  [Datos de sensores ambientales](https://github.com/felipeball19/PLANT_PHENOTYPING/tree/main/anexos/sensores_ambiente.csv)
-  [Datos de sensores de humedad de suelo sensor 1](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/03_promedio_diario_sensor1%20(1).pdf)
-  [Datos de sensores de humedad de suelo sensor 2](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/04_promedio_diario_sensor2%20(1).pdf)
-  [Gráfica de evolución de área foliar](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/evolucion%20de%20area%20en%20cm.png)
-  [Gráfica de evolución de perimetro](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/evolucion%20perimetro%20cm.png)
-  [Gráfica de evolución de solidez](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/visualizaci%C3%B3n%20solidez.png)
-  [Gráfica de distribución área,perimetro,solidez](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/distribucion%20area%2Cperimetro%2Csolidez.png)
-  [visualización de NDVI medio y std](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/distribucion%20ndvi%20medio_std.png)
-  [Tabla de comparacion metricas entre el dataset externo y propio](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/Comparacion_metricas_dataset_externo_propio.xlsx)
-  [Gráfica de comparacion de área foliar entre el dataset externo y propio](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/comparacion_area_plantcv%20%20dataset%20externo%20vs%20propio.pdf)
-  [Gráfica de comparacion de perimetro entre el dataset externo y propio](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/comparacion_perimetro_opencv%20%20dataset%20externo%20vs%20propio.pdf)
-  [Gráfica de comparacion de área solidez entre el dataset externo y propio](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/comparacion_solidez_opencv%20dataset%20externo%20vs%20propio.pdf)
-  [Tabla de interfaz web entrada y salidas](https://github.com/felipeball19/PLANT_PHENOTYPING/blob/main/Anexos/tabla_interfaz_web.xlsx)


