# Sistema de Recuperación de Información basado en Reuters-21578

Integrantes: Cristina Molina, Jair Sanchez

## Descripción del Proyecto

Este proyecto se centra en el desarrollo de un Sistema de Recuperación de Información (SRI) utilizando el corpus Reuters-21578, un conjunto de datos ampliamente utilizado en la investigación de recuperación de información. El objetivo principal es implementar un sistema que permita realizar búsquedas eficientes y precisas dentro del corpus, utilizando técnicas modernas de procesamiento de texto y algoritmos de búsqueda.

### Contexto y Motivación

La recuperación de información es un área fundamental en la ciencia de datos y la inteligencia artificial, ya que permite extraer información relevante de grandes volúmenes de datos no estructurados. El corpus Reuters-21578 contiene miles de artículos de noticias, lo que lo hace ideal para experimentar y evaluar diferentes técnicas de procesamiento y búsqueda de información.

### Objetivos Específicos

1. **Adquisición y Preparación de Datos**: Descargar, descomprimir y organizar los archivos del corpus Reuters-21578.
2. **Preprocesamiento de Datos**: Limpiar los datos, eliminar caracteres no deseados, tokenizar el texto, y aplicar técnicas de stemming y lematización.
3. **Vectorización de Textos**: Convertir los textos en vectores numéricos utilizando técnicas como Bag of Words (BoW) y TF-IDF.
4. **Indexación**: Construir un índice invertido que permita realizar búsquedas rápidas y eficientes.
5. **Implementación del Motor de Búsqueda**: Desarrollar la lógica para procesar consultas y rankear los resultados utilizando algoritmos de similitud como el coseno y Jaccard.
6. **Evaluación del Sistema**: Medir la efectividad del sistema mediante métricas como precisión, recall y F1-score.
7. **Interfaz de Usuario**: Crear una interfaz web intuitiva para que los usuarios puedan interactuar con el sistema y realizar búsquedas.

### Tecnologías Utilizadas

- **Python**: Para el preprocesamiento de datos y el desarrollo del motor de búsqueda.
- **JavaScript**: Para la implementación de la interfaz web.
- **Librerías de Python**: Numpy, Pandas, Scikit-learn, entre otras.

### Estructura del Proyecto

El proyecto está organizado en varias fases, cada una con tareas específicas:

1. **Adquisición de Datos**:
   - Descargar el corpus Reuters-21578.
   - Descomprimir y organizar los archivos.

2. **Preprocesamiento**:
   - Extracción del contenido relevante de los documentos.
   - Limpieza de datos y normalización de texto.
   - Tokenización, eliminación de stop words y aplicación de stemming o lematización.

3. **Representación de Datos en Espacio Vectorial**:
   - Vectorización de los textos utilizando técnicas como BoW y TF-IDF.
   - Evaluación de las técnicas de vectorización.

4. **Indexación**:
   - Construcción de un índice invertido para mapear términos a documentos.
   - Optimización de estructuras de datos para el índice.

5. **Diseño del Motor de Búsqueda**:
   - Desarrollo de la lógica para procesar consultas de usuarios.
   - Implementación de algoritmos de similitud y de ranking.

6. **Evaluación del Sistema**:
   - Definición de métricas de evaluación.
   - Pruebas con el conjunto de prueba del corpus y análisis de resultados.

7. **Interfaz Web de Usuario**:
   - Diseño y desarrollo de una interfaz web donde los usuarios puedan realizar búsquedas.
   - Implementación de características adicionales como filtros y opciones de visualización.

### Resultados Esperados

Al finalizar el proyecto, se espera contar con un sistema funcional de recuperación de información que permita realizar búsquedas eficientes en el corpus Reuters-21578. La evaluación del sistema deberá mostrar una alta precisión y recall, indicando la efectividad del SRI.

### Conclusión

Este proyecto proporciona una oportunidad práctica para aplicar y mejorar habilidades en el procesamiento de lenguaje natural, la recuperación de información y el desarrollo de aplicaciones web. Además, contribuye al entendimiento y aplicación de técnicas avanzadas de manejo de grandes volúmenes de datos textuales.
