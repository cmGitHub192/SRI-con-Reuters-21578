#!/usr/bin/env python
# coding: utf-8

# # Sistema de Recuperación de Información basado en Reuters-21578
# Integrantes: Cristina Molina, Jair Sanchez
# 
# ## Descripción del Proyecto
# 
# Este proyecto se centra en el desarrollo de un Sistema de Recuperación de Información (SRI) utilizando el corpus Reuters-21578, un conjunto de datos ampliamente utilizado en la investigación de recuperación de información. El objetivo principal es implementar un sistema que permita realizar búsquedas eficientes y precisas dentro del corpus, utilizando técnicas modernas de procesamiento de texto y algoritmos de búsqueda.

# # Fases del Proyecto

# ## Fase 1: Adquisición de Datos

#  **Objetivo:**
# 
# El objetivo de esta fase es obtener, descomprimir y organizar el corpus Reuters-21578 de manera que esté listo para ser preprocesado en las siguientes fases del proyecto.
# 
# **Descripción**:
# 
# El corpus Reuters-21578 es un conjunto de datos ampliamente utilizado en la investigación de recuperación de información y procesamiento de lenguaje natural. Contiene artículos de noticias clasificados en varias categorías, y está disponible públicamente para su uso en investigación y desarrollo.
# 
# **Pasos para la Adquisición de Datos**
# 1. **Descarga del Corpus Reuters-21578:** El primer paso es descargar el corpus desde una fuente confiable. El corpus está disponible en varios sitios de la web, pero se recomienda obtenerlo desde el sitio original de la Universidad de Carnegie Mellon (CMU).
# 
#          URL de descarga: https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
#          
#   Pero para este proyecto se descargo la data directamente desde el repositorio proporcionado en el aula virtual
# 
# 2. **Descompresión y Organización de Archivos:** Una vez descargado el archivo comprimido, el siguiente paso es descomprimirlo y organizar los archivos en una estructura de directorios que facilite su acceso y manipulación.

# ## Fase 2: Preprocesamiento

# #### **Objetivo:**
# 
# El objetivo de este código es realizar el preprocesamiento de texto en archivos de texto plano ubicados en un directorio específico. Este preprocesamiento incluye la eliminación de palabras irrelevantes (stopwords), la conversión a minúsculas, la eliminación de caracteres no alfabéticos y números, la tokenización y el stemming de las palabras para reducirlas a su forma raíz.
# 
# #### **Descripción:**
# 
# El preprocesamiento de texto es una etapa fundamental en el procesamiento del lenguaje natural (NLP). Este proceso prepara el texto para su posterior análisis, eliminando ruido y normalizando el formato de las palabras. En este código, se utiliza para limpiar y estructurar el texto contenido en archivos de noticias de Reuters.
# 
# #### **Pasos para el preprocesamiento:**
# 
# 1. **Descarga de Recursos Necesarios**
#     - Se descarga el recurso 'punkt' de la biblioteca NLTK, que se utiliza para la tokenización de palabras.
# 
# 2. **Carga de Stopwords**
#     - Se carga un conjunto de stopwords desde un archivo externo. Las stopwords son palabras comunes que no aportan un significado relevante al análisis y se eliminarán durante el prep*
# 
# 3. **Preprocesamiento de Texto:**
#     - **Convertir el texto a minúsculas:** 
#       - Para garantizar la consistencia en el análisis, todo el texto se convierte a minúsculas. Esto es importante para que las palabras con diferencias de mayúsculas y minúsculas se traten de manera uniforme.
#     
#     - **Remover caracteres no alfabéticos y números:** 
#       - Se utiliza una expresión regular para eliminar caracteres que no son letras del alfabeto y números. Esto incluye signos de puntuación, símbolos y números que no son relevantes para el análisis semántico d
#         
#       - La expresión regular utilizada es r'[^a-z\s]'. Esta expresión regular busca coincidencias de cualquier carácter que no sea una letra minúscula del alfabeto inglés (a-z) ni un espacio en blanco (\s). El modificador ^ dentro de los corchetes [^] indica negación, por lo que la expresión coincide con cualquier carácter que no sea una letra minúscula o un espacio en blanco. Esto permite eliminar signos de puntuación, símbolos y números del texto
#       xto.
#     
#     - **Tokenización:** 
#       - El texto se divide en unidades significativas llamadas tokens utilizando la función `word_tokenize` de la biblioteca NLTK. Esta etapa es crucial para convertir el texto en una lista de palabras individuales que se pueden procesar y analizar por separado.
#     
#     - **Stemming:** 
#       - Se aplica stemming a cada token utilizando el Snowball Stemmer en inglés. El stemming es el proceso de reducir cada palabra a su forma raíz, eliminando sufijos y prefijos para capturar la esencia de la palabra. Por ejemplo, "running" se reduce a "run" y "walked" se r
#         
#       - Se elige Snowball Stemmer en inglés por su eficacia y precisión en este idioma, conservando la integridad semántica de las palabras y siendo menos agresivo en la reducción. Su amplio uso y respaldo en la industria lo convierten en una elección confiable para el preprocesamiento de texto en inglés.
#          a "walk".
#     
#     - **Eliminación de stopwords:** 
#       - Las stopwords, que son palabras comunes pero no informativas como "el", "la", "de", se eliminan del conjunto de tokens procesados. Esto se realiza comparando cada token con un conjunto predefinido de stopwords y eliminándo análisis de texto.de tokens procesados.
# 
# 4. **Guardado de los Resultados**
#     - Los documentos preprocesados se guardan en un archivo de texto llamado 'processed_documents.txt' en el mismo directorio que el código.
# 
# Al finalizar el proceso, los documentos preprocesados están listos para ser utilizados en análisis adicionales, como la construcción de modelos de aprendizaje automático o la creación de índices invertidos para la recuperación de información.
# 

# In[1]:


import os
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
# Download the 'punkt' resource
import nltk
nltk.download('punkt')


# In[2]:


# Función para cargar stopwords desde un archivo
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
         stopwords = set(word.strip() for word in file.readlines())
    return set(stopwords)


# In[3]:


# Función para preprocesar el texto
def preprocess_text(text, stopwords):
    # Convertir el texto a minúsculas
    text = text.lower()

    # Remover caracteres no alfabéticos y números
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenizar el texto en palabras
    tokens = word_tokenize(text)

    # Inicializar el stemmer
    stemmer = SnowballStemmer('english')

    # Aplicar stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Eliminar stopwords después del stemming
    cleaned_tokens = [token for token in stemmed_tokens if token not in stopwords]

    # Unir los tokens limpios en una cadena de texto
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

# Ruta del archivo de stopwords
stopwords_file = 'Proyecto_Data/reuters/stopwords.txt'

# Cargar stopwords
stopwords = load_stopwords(stopwords_file)

# Directorio donde se encuentran los archivos
CORPUS_DIR = 'Proyecto_Data/reuters/training'

# Diccionario para almacenar los textos procesados de todos los archivos
diccionario = {}

# Procesar cada archivo en el directorio
for filename in os.listdir(CORPUS_DIR):
    filepath = os.path.join(CORPUS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
        cleaned_text = preprocess_text(text, stopwords) 
        diccionario[filename] = cleaned_text


# In[4]:


# Guardar los resultados en un archivo
output_file = 'processed_documents.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for filename, text in diccionario.items():
        file.write(f"{filename}: {text}\n")

print(f"Processed documents saved to {output_file}")


# # Fase 3:  Representación de Datos en Espacio Vectorial
# 
# En esta fase, se lleva a cabo la representación de los datos textuales en un espacio vectorial para su posterior procesamiento y análisis. Se utilizan dos técnicas comunes: Bag of Words y TF-IDF (Term Frequency-Inverse Document Frequency).

# #### **Bag of Words (BoW)**
# 
# El enfoque de Bag of Words es una técnica simple pero poderosa para representar datos de texto. Consiste en crear un vector que contiene la frecuencia de ocurrencia de cada palabra en un documento. Este enfoque ignora el orden de las palabras y solo considera su presencia o ausencia en el documento.
# 
# - **Objetivo:**
#     - Convertir el corpus de documentos en una matriz donde cada fila representa un documento y cada columna representa una palabra, con el valor indicando la frecuencia de esa palabra en el documento.
# 
# - **Descripción:**
#     - Se utiliza la clase `CountVectorizer` de la biblioteca scikit-learn para vectorizar el corpus utilizando Bag of Words.
#     - Los documentos se convierten en una lista de textos y luego se vectorizan usando `CountVectorizer`.
#     - El resultado es una matriz donde cada fila representa un documento y cada columna representa una palabra del vocabulario, con los valores indicando la frecuencia de esa palabra en el documento.
# 
# - **Pasos:**
#     1. **Convertir el corpus a una lista de textos:** Se obtienen los textos de los documentos del diccionario.
#     2. **Vectorización usando Bag of Words:** Se utiliza `CountVectorizer` para transformar el corpus en una matriz de frecuencia de palabras.
# 

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

# Convertir el corpus a una lista de textos
corpus = list(diccionario.values())
# Vectorización usando Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)


# In[6]:


import pandas as pd

df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=diccionario.keys())
df_bow


# ### **TF-IDF (Term Frequency-Inverse Document Frequency)**
# 
# TF-IDF es otra técnica de representación de texto que tiene en cuenta tanto la frecuencia de ocurrencia de una palabra en un documento como la importancia de esa palabra en el conjunto de documentos. La importancia se basa en cuántas veces aparece una palabra en un documento (frecuencia de término) y en cuántos documentos contienen esa palabra (frecuencia inversa de documentos).
# 
# - **Objetivo:**
#     - Convertir el corpus de documentos en una matriz donde cada fila representa un documento y cada columna representa una palabra, con el valor indicando la importancia de esa palabra en el documento mediante el esquema TF-IDF.
# 
# - **Descripción:**
#     - Se utiliza la clase `TfidfVectorizer` de scikit-learn para vectorizar el corpus utilizando TF-IDF.
#     - Los documentos se convierten en una lista de textos y luego se vectorizan usando `TfidfVectorizer`.
#     - El resultado es una matriz donde cada fila representa un documento y cada columna representa una palabra del vocabulario, con los valores indicando la importancia de esa palabra en el documento mediante TF-IDF.
# 
# - **Pasos:**
#     1. **Convertir el corpus a una lista de textos:** Se obtienen los textos de los documentos del diccionario.
#     2. **Vectorización usando TF-IDF:** Se utiliza `TfidfVectorizer` para transformar el corpus en una matriz TF-IDF.

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Convertir el corpus a una lista de textos
corpus = list(diccionario.values())

# Vectorización usando TF-IDF
vectorizer = TfidfVectorizer()
Y = vectorizer.fit_transform(corpus)


# In[8]:


df_tf_idf = pd.DataFrame(Y.toarray(), columns=vectorizer.get_feature_names_out(), index=diccionario.keys())
df_tf_idf


# # Fase 4:  Indexación

# ### Objetivo:
# La fase de indexación tiene como objetivo crear y guardar un índice invertido a partir de los datos vectorizados en forma de DataFrames. Este índice invertido permite recuperar rápidamente los documentos que contienen un término específico y proporciona información sobre la frecuencia de ese término en cada documento.
# 
# ### Descripción:
# La indexación es una etapa crucial en el procesamiento de datos textuales. En esta fase, se construye un índice invertido que mapea cada término del vocabulario a la lista de documentos en los que aparece, junto con la frecuencia de ese término en cada documento. Este índice es fundamental para la recuperación eficiente de información y para la realización de consultas en grandes conjuntos de datos textuales.
# 
# ### Pasos:
# 1. **Crear el Índice Invertido:**
#     - Se define la función `crear_indice()` que recorre el DataFrame de términos y frecuencias para cada documento y construye un diccionario donde cada término del vocabulario se mapea a una lista de tuplas que contienen el identificador del documento y la frecuencia del término en ese documento.
#   
# 2. **Guardar el Índice en Archivos de Texto:**
#     - Se define la función `guardar_indice()` que guarda el índice invertido en archivos de texto. Para cada término, se escribe su identificador de documento y su frecuencia asociada en el archivo.
#   spondientes.
#   
# 4. **Crear y Guardar Índices para Bag of Words y TF-IDF:**
#     - Se crean los índices invertidos para los modelos Bag of Words (BoW) y TF-IDF, utilizando las funciones definidas anteriormente.
#   
# 5. **Visualizar los Índices Creados:**
#     - Se llama a la función `visualizar_indice()` para mostrar los índices invertidos como DataFrames de Pandas, lo que permite inspeccionar fácilmente la estructura y el contenido del índice.

# In[9]:


def crear_indice(df):
  indice_invertido = {}

  for columna in df.columns:
    for index, value in df[columna].items():
      if value != 0:
        if columna not in indice_invertido:
          indice_invertido[columna] = []
        indice_invertido[columna].append((index, value))

  return indice_invertido


# In[10]:


def guardar_indice(indice_invertido, directory, filename):
  if not os.path.exists(directory):
    os.makedirs(directory)

  filepath = os.path.join(directory, filename)

  with open(filepath, 'w') as file:
    for termino, documentos in indice_invertido.items():
      file.write(f"Termino: {termino}\n")
      for documento, frecuencia in documentos:
        file.write(f"Documento: {documento}, Frecuencia: {frecuencia}\n")
      file.write("\n")


# In[11]:


# Crear índices para Bag of Words y TF-IDF
indice_bow = crear_indice(df_bow)
indice_tf_idf = crear_indice(df_tf_idf)

# Guardar índices a archivos de texto
guardar_indice(indice_bow, 'Proyecto_Data/results', 'indice_bow.txt')
guardar_indice(indice_tf_idf, 'Proyecto_Data/results', 'indice_tf_idf.txt')


# In[12]:


def visualizar_indice(indice):
    datos = []
    for termino, docs in indice.items():
        datos.append({'Termino': termino, 'Documentos': docs})
    return pd.DataFrame(datos)


# In[13]:


visualizar_indice(indice_bow)


# In[14]:


visualizar_indice(indice_tf_idf)


# # Fase 5:  Diseño del Motor de Búsqueda

# ### Objetivo:
# La fase del diseño del Motor de Búsqueda se enfoca en la implementación de algoritmos de búsqueda para recuperar documentos relevantes basados en consultas de usuario. Se busca aplicar técnicas de similitud para calcular la relevancia de los documentos en función de las consultas ingresadas por el usuario.
# 
# ### Descripción:
# En esta fase, se implementan algoritmos de búsqueda que permiten recuperar documentos relevantes basados en consultas de usuario. Se utilizan dos enfoques principales: búsqueda por similitud de Jaccard y búsqueda por similitud coseno. Estos algoritmos evalúan la similitud entre el texto de la consulta y los documentos almacenados en el corpus, utilizando diferentes métricas para calcular la relevancia de cada documento.
# 
# ### Pasos:
# 1. **Carga del Índice Invertido:**
#     - Se define la función `separar_indice(filepath)` para cargar el índice invertido desde un archivo de texto. Esta función lee el archivo línea por línea, extrayendo la información de los términos y la frecuencia de cada documento.
#     
# 2. **Preprocesamiento de Consulta:**
#     - Se define la función `preprocesar_query(query)` para preprocesar la consulta del usuario. Esta función aplica el mismo preprocesamiento que se utilizó para los documentos del corpus, incluyendo la eliminación de stopwords y la tokenización.
#     
# 3. **Cálculo de Similitud de Jaccard:**
#     - Se implementa la función `busqueda_jaccard(query, inverted_index)` para realizar la búsqueda basada en la similitud de Jaccard. Esta función calcula la similitud entre la consulta y cada documento en el corpus utilizando el coeficiente de Jaccard, que mide la intersección entre los términos de la consulta y los términos del documento.
#     
# 4. **Cálculo de Similitud Coseno:**
#     - Se implementa la función `busqueda_coseno(query, inverted_index)` para realizar la búsqueda basada en la similitud coseno. Esta función calcula la similitud entre la consulta y cada documento en el corpus utilizando el producto escalar y la norma de los vectores de términos, lo que permite comparar la dirección y la magnitud de los vectores.
#     
# 5. **Selección de Resultados:**
#     - Se define la función `results(query, tv, tr)` para seleccionar los resultados de la búsqueda en función de los parámetros `tv` y `tr`, que indican el tipo de vectorización (BoW o TF-IDF) y el tipo de algoritmo de búsqueda (Jaccard o coseno). Esta función llama a las funciones de búsqueda correspondientes y devuelve los documentos más relevantes para la consulta.
#     
# 6. **Ejemplo de Uso:**
#     - Se proporciona un ejemplo de cómo utilizar las funciones de búsqueda para recuperar documentos relevantes en función de una consulta ingresada por el usuario. Esto incluye la llamada a la función `results()` con la consulta, la configuración de vectorización y el tipo de algoritmo de búsqueda.

# In[15]:


def separar_indice(filepath):
    try:
        print(f"Cargando índice invertido desde: {filepath}")
        inverted_index = {}
        with open(filepath, 'r', encoding='utf-8') as file:
            current_term = None
            for line in file:
                line = line.strip()
                if line.startswith("Termino:"):
                    current_term = line.split("Termino: ")[1]
                    inverted_index[current_term] = []
                elif line.startswith("Documento:"):
                    doc_info = line.split("Documento: ")[1]
                    doc_name, weight = doc_info.split(", Frecuencia: ")
                    inverted_index[current_term].append((doc_name, float(weight)))
        return inverted_index
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {filepath}")
        raise
    except Exception as e:
        print(f"Error al cargar el índice invertido desde {filepath}: {str(e)}")
        raise


# In[16]:


def prepocesar_query(query):
    query_prepocesada = preprocess_text(query, stopwords) 
    print(f"Consulta procesada: {query_prepocesada}")
    return query_prepocesada.split()


# In[17]:


def jaccard_similaridad(query_tokens, document_tokens):
    intersection = len(set(query_tokens) & set(document_tokens))
    union = len(set(query_tokens) | set(document_tokens))
    return intersection / union if union != 0 else 0


# In[18]:


import math

def coseno_similaridad(query_vector, doc_vector):
    dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
    query_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    doc_norm = math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0
    return dot_product / (query_norm * doc_norm)


# In[19]:


def busqueda_jaccard(query, inverted_index):
    try:
        CORPUS_DIR = os.path.join(os.getcwd(), 'Proyecto_Data', 'reuters', 'training')
        documents = {}
        for filename in os.listdir(CORPUS_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(CORPUS_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = preprocess_text(text, stopwords) 
                    documents[filename] = cleaned_text
        query_tokens = prepocesar_query(query)
        document_tokens = {doc_id: documents[doc_id].split() for doc_id in documents}
        results = []
        for doc_id in document_tokens:
            similarity = jaccard_similaridad(query_tokens, document_tokens[doc_id])
            results.append((doc_id, similarity))
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
        doc_ids = [doc_id for doc_id, _ in ranked_results]
        return doc_ids
    except Exception as e:
        print(f"Error in search_jaccard: {str(e)}")
        raise


# In[20]:


from collections import defaultdict

def busqueda_coseno(query, inverted_index):
    try:
        query_tokens = prepocesar_query(query)
        query_vector = {term: query_tokens.count(term) for term in query_tokens}
        document_vectors = defaultdict(dict)
        for term in query_tokens:
            if term in inverted_index:
                for doc_id, tfidf_score in inverted_index[term]:
                    document_vectors[doc_id][term] = tfidf_score
        scores = {}
        for doc_id in document_vectors:
            scores[doc_id] = coseno_similaridad(query_vector, document_vectors[doc_id])
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        doc_ids = [doc_id for doc_id, _ in ranked_results]
        return doc_ids
    except Exception as e:
        print(f"Error in search_cosine: {str(e)}")
        raise


# In[21]:


def results(query, tv, tr):
    try:
        if tv == "0" and tr == "1":
            inverted_index_loaded = separar_indice(os.path.join(os.getcwd(), 'Proyecto_Data', 'results', 'indice_tf_idf.txt'))
            results = busqueda_coseno(query, inverted_index_loaded)
        elif tv == "1" and tr == "0":
            inverted_index_loaded = separar_indice(os.path.join(os.getcwd(), 'Proyecto_Data', 'results', 'indice_bow.txt'))
            results = busqueda_jaccard(query, inverted_index_loaded)
        else:
            raise ValueError("Invalid combination of tv and tr values. Only BoW with Jaccard and TF-IDF with Cosine are allowed.")
        return results
    except Exception as e:
        print(f"Error in results function with query '{query}', tv='{tv}', tr='{tr}': {str(e)}")
        raise


# In[22]:


import os

# Ejemplo de llamada a la función results
query = "coffee"
tv = "0"  # Para TF-IDF con Coseno
tr = "1"  # Para TF-IDF con Coseno

try:
    # Llamada a la función results
    search_results = results(query, tv, tr)
    print("Resultados de la búsqueda:", search_results)
except ValueError as e:
    print(f"Error: {str(e)}")
except Exception as e:
    print(f"Error inesperado: {str(e)}")

