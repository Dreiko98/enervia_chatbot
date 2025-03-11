# Chatbot para Enervia – Proyecto de Retrieval Augmented Generation (RAG)

Este proyecto implementa un chatbot para [Enervia: Soluciones Energéticas](https://enervia.net) utilizando técnicas de **Retrieval Augmented Generation** (RAG). El chatbot extrae información relevante de dos documentos PDF (un dossier unificado y un PDF de FAQs), los segmenta, limpia y genera embeddings con modelos gratuitos (SentenceTransformers). Luego, indexa estos embeddings usando FAISS para recuperar fragmentos relevantes a partir de una consulta del usuario. Finalmente, se utiliza la API de OpenAI (ChatCompletion con GPT-3.5-turbo) para generar respuestas contextualizadas basadas en la información recuperada.

## Tabla de Contenidos

- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Extracción y Segmentación](#extracción-y-segmentación)
  - [Generación de Embeddings e Indexación](#generación-de-embeddings-e-indexación)
  - [Consulta y Respuesta del Chatbot](#consulta-y-respuesta-del-chatbot)
- [Estructura del Código](#estructura-del-código)
- [Dependencias](#dependencias)
- [Consideraciones y Futuras Mejoras](#consideraciones-y-futuras-mejoras)
- [Contacto](#contacto)

## Características

- **Extracción de Texto desde PDFs:**  
  Usa `PyPDF2` para extraer el contenido de dos documentos (dossier unificado y FAQs).

- **Segmentación y Limpieza:**  
  Separa el contenido en fragmentos grandes basados en encabezados y subtítulos, corrigiendo errores de OCR o formateo.

- **Generación de Embeddings Gratuitos:**  
  Emplea la biblioteca `SentenceTransformers` (modelo `all-MiniLM-L6-v2`) para convertir cada fragmento en un vector semántico.

- **Indexación con FAISS:**  
  Almacena y gestiona los embeddings en un índice FAISS para búsquedas rápidas por similitud.

- **Respuesta Contextualizada con OpenAI Chat API:**  
  Recupera fragmentos relevantes a partir de la consulta del usuario y los utiliza en un prompt para generar respuestas usando GPT-3.5-turbo.

## Instalación

### Requisitos Previos

- Python 3.7 o superior.
- Acceso a la API de OpenAI (con la clave correspondiente).
- (Opcional) GPU para acelerar el procesamiento de embeddings, aunque no es indispensable.

### Dependencias

Instala las dependencias necesarias ejecutando:

```bash
pip install PyPDF2
pip install sentence-transformers
pip install faiss-cpu  # O faiss-gpu si dispones de GPU
pip install openai
```

Si utilizas la nueva API de Chat, asegúrate de tener la versión actualizada de `openai`.

## Uso

El flujo principal del proyecto se divide en tres pasos:

### Extracción y Segmentación

1. **Extraer Texto desde PDFs:**  
   El script `extraer_texto_pdf` utiliza `PyPDF2` para extraer el contenido de los PDFs.  
2. **Segmentación por Encabezados y Subtítulos:**  
   - `segmentar_por_encabezados`: Separa el contenido en secciones grandes usando patrones numéricos.
   - `segmentar_por_subtitulos_o_parrafos`: Divide cada sección en párrafos o fragmentos basados en saltos de línea dobles.
3. **División por Longitud y Limpieza:**  
   - `dividir_por_longitud`: Se asegura de que cada fragmento no supere un número máximo de palabras (por defecto 300).
   - `clean_text`: Corrige errores comunes y normaliza espacios.

### Generación de Embeddings e Indexación

1. **Generación de Embeddings:**  
   Se utiliza `SentenceTransformer` (modelo `all-MiniLM-L6-v2`) para generar embeddings de cada fragmento limpio.
2. **Indexación con FAISS:**  
   Se crea un índice FAISS (usando distancia Euclidiana L2) para almacenar todos los embeddings y permitir búsquedas por similitud.

### Consulta y Respuesta del Chatbot

1. **Búsqueda de Fragmentos Relevantes:**  
   La función `buscar_fragmentos` convierte la consulta del usuario en un embedding y busca en el índice FAISS los fragmentos más cercanos.
2. **Generación de Respuesta:**  
   Se concatenan los fragmentos relevantes y se envían junto con la consulta a la API de OpenAI (usando `ChatCompletion.create` con GPT-3.5-turbo) para obtener una respuesta contextualizada.

Ejemplo de uso en el código:

```python
# Ejemplo de consulta y respuesta:
consulta = "¿Con qué tipos de clientes trabaja Enervia?"
indices_similares, distancias = buscar_fragmentos(consulta, k=3)
fragmentos_relevantes = "\n".join(clean_fragments[idx] for idx in indices_similares)
respuesta = generar_respuesta_con_contexto(consulta, fragmentos_relevantes)
print("Chatbot:", respuesta)
```

## Estructura del Código

- **Extracción de Texto:**  
  - `extraer_texto_pdf`: Extrae contenido de PDFs.
- **Segmentación y Limpieza:**  
  - `segmentar_por_encabezados`
  - `segmentar_por_subtitulos_o_parrafos`
  - `dividir_por_longitud`
  - `clean_text`
- **Embeddings e Indexación:**  
  - Uso de `SentenceTransformer` para generar embeddings.
  - Creación del índice FAISS para almacenar y buscar vectores.
- **Consulta y Respuesta:**  
  - `buscar_fragmentos`: Recupera índices y distancias de fragmentos relevantes.
  - `generar_respuesta_con_contexto`: Llama a la API de OpenAI para generar respuestas.

## Dependencias

- [PyPDF2](https://pypi.org/project/PyPDF2/): Para extracción de texto de PDFs.
- [SentenceTransformers](https://www.sbert.net/): Para generación de embeddings.
- [FAISS](https://github.com/facebookresearch/faiss): Para indexación y búsqueda de vectores.
- [OpenAI](https://github.com/openai/openai-python): Para generación de respuestas con GPT-3.5-turbo.

## Consideraciones y Futuras Mejoras

- **Optimización de la Segmentación:**  
  Se puede mejorar la segmentación manualmente ajustando patrones y límites de palabras.
- **Actualización del Corpus:**  
  Con el tiempo, si se agregan nuevos documentos o se actualiza la información, será necesario reindexar los embeddings.
- **Escalabilidad:**  
  Aunque actualmente el corpus es pequeño (11 páginas), la estructura está pensada para escalar en caso de que se agregue más información.
- **Ajustes en el Prompt:**  
  La construcción del prompt para la API de OpenAI se puede refinar para obtener respuestas más precisas.
- **Interfaz Web:**  
  En el futuro se puede integrar este pipeline en una API web usando Flask o FastAPI para interactuar en tiempo real con los usuarios.

## Contacto

Si tienes alguna pregunta, sugerencia o contribución, no dudes en contactar a:

- [Germán Mallo Faure](https://germanmallo.com)
- Email: germanmallo04@gmail.com
