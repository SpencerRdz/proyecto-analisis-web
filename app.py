from flask import Flask, request, render_template, jsonify
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import numpy as np
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import heapq
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def validar_url(url):
    try:
        response = requests.head(url)
        if response.status_code == 200:
            return True
        else:
            print(f"Error al validar la URL: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión al validar la URL: {e}")
        return False

def obtener_contenido_web(url):
    if not validar_url(url):
        return None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error al obtener la página: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None

def obtener_html_con_javascript(url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(2)

        html = driver.page_source
        driver.quit()

        return html
    except Exception as e:
        print(f"Error al obtener HTML con JavaScript: {e}")
        return None

def extraer_contenido(url, tipo_contenido):
    if tipo_contenido not in ('titulo', 'imagen', 'enlace', 'texto'):
        print(f"Tipo de contenido no válido: {tipo_contenido}")
        return None

    html = obtener_contenido_web(url)
    if not html:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    if tipo_contenido == 'texto':
        selector_css = 'p'
        texto_extraido = ' '.join([elemento.get_text() for elemento in soup.select(selector_css)])
        return texto_extraido

    elif tipo_contenido == 'titulo':
        selector_css = 'h1, h2, h3, h4, h5, h6'
        titulos = ' '.join([elemento.get_text() for elemento in soup.select(selector_css)])
        return titulos.strip() if titulos else None

    elif tipo_contenido == 'imagen':
        selector_css = 'img'
        imagenes = [elemento.get('src') for elemento in soup.select(selector_css) if elemento.get('src')]
        return imagenes if imagenes else None

    elif tipo_contenido == 'enlace':
        selector_css = 'a'
        enlaces = [elemento.get('href') for elemento in soup.select(selector_css) if elemento.get('href')]
        return enlaces if enlaces else None

def resumen_texto(nombre_archivo, longitud_resumen, archivo_salida):
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    oraciones = tokenize.sent_tokenize(texto)

    oraciones_limpias = [preprocesar_oracion(oracion) for oracion in oraciones]

    matriz_tfidf = crear_matriz_tfidf(oraciones_limpias)
    puntuaciones_oracion = calcular_puntuacion_oracion(matriz_tfidf)

    oraciones_top = heapq.nlargest(longitud_resumen, range(len(puntuaciones_oracion)), key=puntuaciones_oracion.__getitem__)

    resumen = ' '.join([oraciones[i] for i in oraciones_top])

    resumen_refinado = refinar_resumen(resumen)

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
    vectorizador = TfidfVectorizer(ngram_range=(1, 3))
    matriz_tfidf = vectorizador.fit_transform(oraciones)
    return matriz_tfidf

def calcular_puntuacion_oracion(matriz_tfidf):
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

def responder_pregunta(nombre_archivo, pregunta, umbral_confianza=0.3):
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    modelo = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=modelo, tokenizer=modelo)

    respuesta = qa_pipeline(question=pregunta, context=texto)

    if respuesta['score'] < umbral_confianza:
        return "No se han encontrado datos sobre tu pregunta"
    
    return respuesta['answer']

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    url = data.get('url')
    tipo_contenido = data.get('tipo_contenido')
    longitud_resumen = data.get('longitud_resumen')
    pregunta = data.get('pregunta')

    respuesta = {}
    
    if url and tipo_contenido:
        contenido = extraer_contenido(url, tipo_contenido)
        if tipo_contenido == 'texto' and contenido:
            with open('textos_extraidos/texto_parrafos.txt', 'w', encoding='utf-8') as f:
                f.write(contenido)
        respuesta['contenido'] = contenido

    if longitud_resumen:
        archivo_salida = 'resumen_pagina_web/resumen.txt'
        resumen = resumen_texto('textos_extraidos/texto_parrafos.txt', longitud_resumen, archivo_salida)
        respuesta['resumen'] = resumen

    if pregunta:
        respuesta_pregunta = responder_pregunta('textos_extraidos/texto_parrafos.txt', pregunta)
        respuesta['respuesta'] = respuesta_pregunta

    return jsonify(respuesta)

if __name__ == '__main__':
    app.run(debug=True)
