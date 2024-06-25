import numpy as np
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import heapq
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Descargar recursos de nltk si es necesario
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def resumen_texto(nombre_archivo, longitud_resumen, archivo_salida):
    # Leer el archivo
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    # Limpiar y tokenizar el texto
    oraciones = tokenize.sent_tokenize(texto)  # Separar en oraciones

    # Preprocesar las oraciones
    oraciones_limpias = [preprocesar_oracion(oracion) for oracion in oraciones]

    # Calcular la importancia de cada oración
    matriz_tfidf = crear_matriz_tfidf(oraciones_limpias)
    puntuaciones_oracion = calcular_puntuacion_oracion(matriz_tfidf)

    # Seleccionar las oraciones más importantes
    oraciones_top = heapq.nlargest(longitud_resumen, range(len(puntuaciones_oracion)), key=puntuaciones_oracion.__getitem__)

    # Generar el resumen
    resumen = ' '.join([oraciones[i] for i in oraciones_top])

    # Utilizar un modelo avanzado para refinar el resumen
    resumen_refinado = refinar_resumen(resumen)

    # Guardar el resumen en un archivo
    with open(archivo_salida, 'w', encoding='utf-8') as archivo:
        archivo.write(resumen_refinado)

    return resumen_refinado

def preprocesar_oracion(oracion):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    palabras = tokenize.word_tokenize(oracion.lower())
    palabras_limpias = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra.isalnum() and palabra not in stop_words]
    return ' '.join(palabras_limpias)

def crear_matriz_tfidf(oraciones):
    vectorizador = TfidfVectorizer(ngram_range=(1, 3))  # Utilizar unigramas, bigramas y trigramas
    matriz_tfidf = vectorizador.fit_transform(oraciones)
    return matriz_tfidf

def calcular_puntuacion_oracion(matriz_tfidf):
    # Usar SVD para reducir la dimensionalidad y enfocarse en las características principales
    svd = TruncatedSVD(n_components=1, random_state=42)
    puntuaciones_oracion = svd.fit_transform(matriz_tfidf)
    return puntuaciones_oracion

def refinar_resumen(resumen):
    modelo = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(modelo)
    
    inputs = tokenizer([resumen], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = modelo.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    resumen_refinado = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return resumen_refinado

def responder_pregunta(nombre_archivo, pregunta):
    # Leer el archivo
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    # Crear un pipeline de preguntas y respuestas utilizando un modelo avanzado
    modelo = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=modelo, tokenizer=modelo)

    # Realizar la pregunta
    respuesta = qa_pipeline(question=pregunta, context=texto)
    return respuesta['answer']

# Ejemplo de uso
nombre_archivo = "textos_extraidos/texto_parrafos.txt"  # Cambiar por el nombre del archivo real
archivo_salida = "resumen_pagina_web/resumen.txt"  # Nombre del archivo donde se guardará el resumen
longitud_resumen = 5  # Cambiar por el número deseado de oraciones

resumen = resumen_texto(nombre_archivo, longitud_resumen, archivo_salida)
print("Resumen guardado en", archivo_salida)

while True:
    pregunta = input("Ingrese su pregunta (o 'salir' para terminar): ")
    if pregunta.lower() == 'salir':
        break
    respuesta = responder_pregunta(nombre_archivo, pregunta)
    print("Respuesta a la pregunta:")
    print(respuesta)
