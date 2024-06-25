import numpy as np
from nltk import tokenize
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import spacy

def resumen_texto(nombre_archivo, longitud_resumen):
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            texto = archivo.read()
    except FileNotFoundError:
        print(f"Error: El archivo {nombre_archivo} no se encontró.")
        return None

    # Preprocesamiento del texto
    texto = texto.lower()
    oraciones = tokenize.sent_tokenize(texto)

    # Eliminar oraciones repetitivas
    oraciones_sin_repetir = eliminar_oraciones_repetitivas(oraciones)

    # Incorporar información contextual
    pesos_oracion = calcular_pesos_oracion(oraciones_sin_repetir)

    # Análisis semántico
    matriz_tfidf, vectorizador = crear_matriz_tfidf(oraciones_sin_repetir)
    vectores_oracion = calcular_vectores_oracion(matriz_tfidf, vectorizador)
    similitud_oracion = calcular_similitud_oracion(vectores_oracion)

    # Seleccionar oraciones relevantes
    puntuaciones_oracion = combinar_puntuaciones(pesos_oracion, matriz_tfidf, similitud_oracion)
    oraciones_top = heapq.nlargest(longitud_resumen, range(len(puntuaciones_oracion)), key=puntuaciones_oracion.__getitem__)

    # Generar resumen coherente
    resumen = generar_resumen_coherente([oraciones_sin_repetir[i] for i in oraciones_top])

    return resumen

def eliminar_oraciones_repetitivas(oraciones):
    oraciones_sin_repetir = []
    conjunto_hashes = set()

    for oracion in oraciones:
        vector_oracion = crear_vector_oracion(oracion)
        hash_oracion = generar_hash(vector_oracion)

        if hash_oracion not in conjunto_hashes:
            oraciones_sin_repetir.append(oracion)
            conjunto_hashes.add(hash_oracion)

    return oraciones_sin_repetir

def calcular_pesos_oracion(oraciones):
    pesos_oracion = []
    nlp = spacy.load("es_core_web_sm")

    for oracion in oraciones:
        documento = nlp(oracion)

        if documento.sents[0].is_sent_start:
            peso = 1.2
        elif documento.sents[-1].is_sent_end:
            peso = 1.2
        else:
            peso = 1.0

        for token in documento:
            if token.pos_ == "NOUN" and token.is_proper_noun:
                peso *= 1.1

        pesos_oracion.append(peso)

    return pesos_oracion

def crear_matriz_tfidf(oraciones):
    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(oraciones)
    return matriz_tfidf, vectorizador

def calcular_vectores_oracion(matriz_tfidf, vectorizador):
    modelo_word_embeddings = KeyedVectors.load_word2vec_format("modelos/fasttext.txt")

    vectores_oracion = []
    for fila_tfidf in matriz_tfidf.toarray():
        vector_oracion = np.zeros((len(modelo_word_embeddings.index_to_key),))
        for i, valor_tfidf in enumerate(fila_tfidf):
            if valor_tfidf > 0:
                palabra = vectorizador.get_feature_names_out()[i]
                if palabra in modelo_word_embeddings:
                    vector_oracion += valor_tfidf * modelo_word_embeddings[palabra]
        vectores_oracion.append(vector_oracion / np.linalg.norm(vector_oracion))

    return np.array(vectores_oracion)

def calcular_similitud_oracion(vectores_oracion):
    similitud_oracion = cosine_similarity(vectores_oracion)
    return similitud_oracion

def combinar_puntuaciones(pesos_oracion, matriz_tfidf, similitud_oracion):
    puntuaciones_oracion = []
    for i, peso in enumerate(pesos_oracion):
        puntuacion = np.sum(matriz_tfidf[i]) * np.mean(similitud_oracion[i]) * peso
        puntuaciones_oracion.append(puntuacion)

    return puntuaciones_oracion

def generar_resumen_coherente(oraciones_top):
    resumen = " ".join(oraciones_top)
    return resumen.strip()

# Ejemplo de uso
resumen = resumen_texto("textos_extraidos/texto_parrafos.txt", 5)
print(resumen)
