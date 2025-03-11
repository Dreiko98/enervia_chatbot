# app.py

from flask import Flask, render_template, request, jsonify
import openai
from utils import (
    extraer_texto_pdf, segmentar_documento, limpiar_fragmentos,
    generar_embeddings, crear_indice, buscar_fragmentos
)

# openai.api_key = "APIKEY"

app = Flask(__name__)

# Lee y procesa tus PDFs
ruta_dossier = "contenido_para_alimentar/guia_enervia.pdf"
ruta_faq = "contenido_para_alimentar/ENERVIA_faqs.pdf"

texto_dossier = extraer_texto_pdf(ruta_dossier)
texto_faq = extraer_texto_pdf(ruta_faq)

textos_extraidos = [texto_dossier, texto_faq]
fragmentos = segmentar_documento(textos_extraidos, max_palabras=300)
clean_fragments = limpiar_fragmentos(fragmentos)

# Genera embeddings e índice
embeddings, modelo = generar_embeddings(clean_fragments)
faiss_index = crear_indice(embeddings)  # <--- Cambia el nombre aquí

def generar_respuesta_con_contexto(query, fragmentos_relevantes):
    system_message = {
        "role": "system",
        "content": (
            "Eres un asistente virtual de Enervía. Usa la siguiente información para responder "
            "de forma clara y concisa a la pregunta del usuario. No inventes datos si no aparecen "
            "en los fragmentos.\n\n"
            f"Información relevante:\n{fragmentos_relevantes}"
        )
    }
    user_message = {
        "role": "user",
        "content": query
    }
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
        temperature=0.7,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"].strip()

@app.route("/")  # <--- Cambia el nombre de la función a home()
def home():
    return render_template("index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():
    query = request.form.get("query")
    indices_similares, _ = buscar_fragmentos(query, modelo, faiss_index, clean_fragments, k=3)
    fragmentos_relevantes = "\n".join(clean_fragments[idx] for idx in indices_similares)
    respuesta = generar_respuesta_con_contexto(query, fragmentos_relevantes)
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(debug=True)
