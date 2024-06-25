import numpy as np
from nltk import tokenize
import heapq

def resumen_texto(nombre_archivo, longitud_resumen):#Función que resume un archivo de texto plano.

  with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
    texto = archivo.read()

  # Limpiar y tokenizar el texto
  texto = texto.lower()  # Convertir a minúsculas
  oraciones = tokenize.sent_tokenize(texto)  # Separar en oraciones

  # Calcular la importancia de cada oración
  matriz_tfidf = crear_matriz_tfidf(oraciones)
  puntuaciones_oracion = calcular_puntuacion_oracion(matriz_tfidf)

  # Seleccionar las oraciones más importantes
  oraciones_top = heapq.nlargest(longitud_resumen, range(len(puntuaciones_oracion)), key=puntuaciones_oracion.__getitem__)

  # Generar el resumen
  resumen = ' '.join([oraciones[i] for i in oraciones_top])
  return resumen

def crear_matriz_tfidf(oraciones):#Función que crea una matriz TF-IDF a partir de un conjunto de oraciones.

  from sklearn.feature_extraction.text import TfidfVectorizer

  vectorizador = TfidfVectorizer()
  matriz_tfidf = vectorizador.fit_transform(oraciones)
  return matriz_tfidf

def calcular_puntuacion_oracion(matriz_tfidf):#Función que calcula la puntuación de cada oración.

  puntuaciones_oracion = np.sum(matriz_tfidf, axis=1)
  return puntuaciones_oracion

# Ejemplo de uso
nombre_archivo = "textos_extraidos/texto_parrafos.txt"  # Cambiar por el nombre del archivo real
longitud_resumen = 5  # Cambiar por el número deseado de oraciones

resumen = resumen_texto(nombre_archivo, longitud_resumen)
print(resumen)
