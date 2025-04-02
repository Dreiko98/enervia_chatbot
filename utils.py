import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

# 1. Función para leer PDFs
def extraer_texto_pdf(ruta_pdf):
    """
    Extrae y retorna el texto completo del PDF ubicado en 'ruta_pdf'.
    - Abre el archivo en modo binario.
    - Utiliza PyPDF2.PdfReader para leer cada página.
    - Extrae y concatena el texto de cada página.
    """
    texto_completo = ""
    with open(ruta_pdf, "rb") as archivo:
        lector = PyPDF2.PdfReader(archivo)
        for pagina in lector.pages:
            texto = pagina.extract_text()
            if texto:  # Verificar que se haya extraído texto
                texto_completo += texto + "\n"
    return texto_completo

# 2. Segmentación

def segmentar_por_encabezados(textos):
    """
    Divide el texto en fragmentos utilizando encabezados numéricos (Ej: "1. Introducción") como delimitadores.
    Retorna una lista de secciones grandes para cada documento.
    """
    secciones = []
    for texto in textos:
        # Patrón para encabezados tipo "1. " "2. " etc.
        pattern = r'(?=\n?\d+\.\s)'
        secciones_texto = re.split(pattern, texto)
        secciones_texto = [sec.strip() for sec in secciones_texto if sec.strip()]
        secciones.append(secciones_texto)
    return secciones

def segmentar_por_subtitulos_o_parrafos(texto):
    """
    Segmenta el texto en fragmentos usando doble salto de línea como delimitador.
    """
    pattern = r'(\n\s*\n)'  # doble salto de línea
    subfragmentos = re.split(pattern, texto)
    subfragmentos = [frag.strip() for frag in subfragmentos if frag.strip() and frag != pattern]
    return subfragmentos

def dividir_por_longitud(fragmentos, max_palabras=300):
    """
    Si un fragmento excede max_palabras, lo divide en trozos de tamaño máximo 'max_palabras'.
    Retorna una nueva lista de fragmentos.
    """
    resultado = []
    for frag in fragmentos:
        palabras = frag.split()
        if len(palabras) <= max_palabras:
            resultado.append(frag)
        else:
            for i in range(0, len(palabras), max_palabras):
                chunk = " ".join(palabras[i:i+max_palabras])
                resultado.append(chunk)
    return resultado

def segmentar_documento(textos, max_palabras=300):
    """
    Procesa una lista de textos (cada uno de un PDF) y retorna una lista de fragmentos finales.
    """
    # 1) Dividir cada documento por encabezados
    secciones_principales = segmentar_por_encabezados(textos)
    
    # 2) Aplanar las secciones y dividir cada una por subtítulos o párrafos
    subfragmentos_totales = []
    for secciones_doc in secciones_principales:  # Cada documento: lista de secciones
        for seccion in secciones_doc:             # Cada sección individual (string)
            subfragmentos = segmentar_por_subtitulos_o_parrafos(seccion)
            subfragmentos_totales.extend(subfragmentos)
    
    # 3) Dividir los subfragmentos largos por longitud
    fragmentos_finales = dividir_por_longitud(subfragmentos_totales, max_palabras)
    return fragmentos_finales

# 3. Función para limpiar texto
def clean_text(text):
    """
    Limpia el texto aplicando correcciones específicas y normalizando espacios.
    """
    corrections = {
        "Infor mación": "Información",
        "Sopor te": "Soporte",
        "Compr omiso": "Compromiso",
        "Pr oyectos": "Proyectos",
        "Cont enido": "Contenido",
        "T estimonios": "Testimonios",
        "Client e": "Cliente",
        "W eb": "Web",
        "Mant enimient o": "Mantenimiento",
        "T écnicas": "Técnicas",
        "Ener gías R enov ables": "Energías Renovables",
        "Pr esentaciones Int ernas": "Presentaciones Internas",
        "R euniones": "Reuniones",
        "Mark eting": "Marketing",
        "Objetiv os": "Objetivos"
    }
    for error, correct in corrections.items():
        text = text.replace(error, correct)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def limpiar_fragmentos(fragmentos):
    """
    Aplica la función de limpieza a cada fragmento.
    """
    return [clean_text(frag) for frag in fragmentos]

# 4. Generación de embeddings con SentenceTransformers
def generar_embeddings(fragmentos, model_name='all-MiniLM-L6-v2'):
    """
    Genera embeddings para cada fragmento utilizando un modelo preentrenado.
    Retorna un array de embeddings y el modelo cargado.
    """
    modelo = SentenceTransformer(model_name)
    embeddings = modelo.encode(fragmentos, convert_to_numpy=True)
    return embeddings, modelo

# 5. Crear índice con FAISS
def crear_indice(embeddings):
    """
    Crea y retorna un índice FAISS a partir de un array de embeddings.
    """
    emb_array = np.array(embeddings, dtype='float32')
    dimension = emb_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(emb_array)
    return index

# 6. Función para buscar fragmentos relevantes
def buscar_fragmentos(query, modelo, index, clean_fragments, k=3):
    """
    Retorna los índices y distancias de los k fragmentos más similares a la consulta.
    """
    query_embedding = modelo.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(query_embedding, k)
    return I[0], D[0]

if __name__ == "__main__":
    # Ejemplo de uso
    ruta_dossier = "static/guia_enervia.pdf"
    ruta_faq = "static/ENERVIA_faqs.pdf"

    texto_dossier = extraer_texto_pdf(ruta_dossier)
    texto_faq = extraer_texto_pdf(ruta_faq)
    
    textos_extraidos = [texto_dossier, texto_faq]
    fragmentos = segmentar_documento(textos_extraidos, max_palabras=300)
    clean_fragments = limpiar_fragmentos(fragmentos)
    
    print(f"Total de fragmentos: {len(clean_fragments)}")
    for i, frag in enumerate(clean_fragments, start=1):
        num_palabras = len(frag.split())
        print(f"Fragmento {i} ({num_palabras} palabras):\n{frag}\n{'-'*80}")
    
    embeddings, _ = generar_embeddings(clean_fragments)
    index = crear_indice(embeddings)
    print(f"Total de vectores indexados: {index.ntotal}")
