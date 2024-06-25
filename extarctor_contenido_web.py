import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

#siuuuu
def validar_url(url):#!Valida si la URL proporcionada tiene un formato correcto y es accesible.
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


def obtener_contenido_web(url):#!Función que obtiene el contenido HTML de una página web.
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


def obtener_html_con_javascript(url):#!Función que obtiene el HTML completo de una página web que utiliza JavaScript.
    
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Ejecutar Chrome en modo sin cabeza

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(2)  # Esperar a que se cargue la página

        html = driver.page_source
        driver.quit()

        return html
    except Exception as e:
        print(f"Error al obtener HTML con JavaScript: {e}")
        return None


def extraer_texto(html, selector_css):#!Función que extrae el texto de un contenido HTML.
    try:
        soup = BeautifulSoup(html, 'html.parser')
        elementos = soup.select(selector_css)
        texto = ' '.join([elemento.get_text() for elemento in elementos])
        return texto
    except Exception as e:
        print(f"Error al extraer texto del HTML: {e}")
        return None


def extraer_contenido(url, tipo_contenido, selector_css=None):#!Función que extrae contenido de una página web.
    if tipo_contenido not in ('titulo', 'imagen', 'enlace', 'texto'):
        print(f"Tipo de contenido no válido: {tipo_contenido}")
        return None

    html = obtener_contenido_web(url)
    if not html:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    if tipo_contenido == 'texto':
        if selector_css is None:
            selector_css = 'p'  # Extraer texto de párrafos por defecto
        texto_extraido = extraer_texto(html, selector_css)
        return texto_extraido

    elif tipo_contenido == 'titulo':
        selector_css = 'h1, h2, h3, h4, h5, h6'  # Extraer todos los títulos
        titulos = extraer_texto(html, selector_css)
        if titulos:
            return titulos.strip()  # Eliminar espacios al principio y final
        else:
            return None

    elif tipo_contenido == 'imagen':
        selector_css = 'img'  # Extraer todas las imágenes
        elementos = soup.select(selector_css)
        imagenes = []
        for elemento in elementos:
            src = elemento.get('src')
            if src:
                imagenes.append(src)
        if imagenes:
            return imagenes
        else:
            return None

    elif tipo_contenido == 'enlace':
        selector_css = 'a'  # Extraer todos los enlaces
        elementos = soup.select(selector_css)
        enlaces = []
        for elemento in elementos:
            href = elemento.get('href')
            if href:
                enlaces.append(href)
        if enlaces:
            return enlaces
        else:
            return None

    else:
        raise NotImplementedError(f"Tipo de contenido no implementado: {tipo_contenido}")


def guardar_texto_en_archivo(texto, nombre_archivo):#!Guarda texto en un archivo de texto.

    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(texto)
        print(f"Texto guardado exitosamente en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar el texto: {e}")


if __name__ == '__main__':
    # Ejemplo de uso
    url = input("ingreza un url: ")

    # Extraer texto de párrafos
    texto_parrafos = extraer_contenido(url, 'texto')
    if texto_parrafos:
        print("Texto de párrafos:")
        print(texto_parrafos)

        # Guardar texto en un archivo
        guardar_texto_en_archivo(texto_parrafos, 'textos_extraidos/texto_parrafos.txt')

    # Extraer títulos
    titulos = extraer_contenido(url, 'titulo')
    if titulos:
        print("\nTítulos:")
        for titulo in titulos:
            print(titulo)

    # Extraer imágenes (URLs)
    imagenes = extraer_contenido(url, 'imagen')
    if imagenes:
        print("\nImágenes (URLs):")
        for imagen in imagenes:
            print(imagen)

    # Extraer enlaces (URLs)
    enlaces = extraer_contenido(url, 'enlace')
    if enlaces:
        print("\nEnlaces (URLs):")
        for enlace in enlaces:
            print(enlace)
            #hola2727
