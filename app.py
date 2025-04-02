# app.py

from flask import Flask, render_template, request, jsonify
import openai
from utils import (
    extraer_texto_pdf, segmentar_documento, limpiar_fragmentos,
    generar_embeddings, crear_indice, buscar_fragmentos
)
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

app.secret_key = os.getenv("OPENAI_API_KEY")

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
    
    # Recuperar o inicializar el historial de conversación en la sesión
    if "conversation" not in session:
        session["conversation"] = []  # Lista de mensajes (cada mensaje es un dict)
    
    conversation = session["conversation"]
    
    # Obtener fragmentos relevantes (como ya lo tenías)
    indices_similares, _ = buscar_fragmentos(query, modelo, faiss_index, clean_fragments, k=3)
    fragmentos_relevantes = "\n".join(clean_fragments[idx] for idx in indices_similares)
    
    # Mensaje de sistema (contexto base)
    system_message = {
        "role": "system",
        "content": (
            "Eres un asistente virtual de Enervía. Usa la siguiente información para responder de forma "
            "clara y concisa a la pregunta del usuario. No inventes datos si no aparecen en los fragmentos.\n\n"
            f"Información relevante:\n{fragmentos_relevantes}"
        )
    }
    
    # Agregar el nuevo mensaje del usuario al historial
    user_message = {"role": "user", "content": query}
    conversation.append(user_message)
    
    # Limitar la conversación a los últimos 3 intercambios (o a 6 mensajes: 3 del usuario y 3 del asistente)
    max_mensajes = 6
    if len(conversation) > max_mensajes:
        conversation = conversation[-max_mensajes:]
    
    # Construir la lista final de mensajes, empezando con el mensaje del sistema
    messages = [system_message] + conversation

    # Llamar a la API de ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    assistant_response = response["choices"][0]["message"]["content"].strip()
    
    # Agregar la respuesta del asistente al historial
    conversation.append({"role": "assistant", "content": assistant_response})
    # Actualizar la sesión
    session["conversation"] = conversation
    
    return jsonify({"respuesta": assistant_response})

@app.route("/reset", methods=["POST"])
def reset():
    session["conversation"] = []
    return jsonify({"status": "ok"})



if __name__ == "__main__":
    print("Servidor iniciado")
    app.run(debug=True)
