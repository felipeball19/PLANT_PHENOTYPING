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

