import streamlit as st
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------
st.set_page_config(page_title="Asistente Legal", page_icon="💬")

# ------------------------------------------------------------------
# Bloqueo de seguridad por contraseña
# ------------------------------------------------------------------
def verificar_password():
    if st.session_state.get("password_correcta", False):
        return True

    def password_ingresada():
        if st.session_state.get("password_input") == st.secrets["ACCESO_PASSWORD"]:
            st.session_state["password_correcta"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correcta"] = False

    st.title("🔒 Acceso restringido")
    st.text_input(
        "Introduce la contraseña de acceso", 
        type="password", 
        on_change=password_ingresada, 
        key="password_input"
    )
    
    if "password_correcta" in st.session_state and not st.session_state["password_correcta"]:
        st.error("Contraseña incorrecta. Inténtalo de nuevo.")
    return False

if not verificar_password():
    st.stop()

# ------------------------------------------------------------------
# Clientes de Gemini y Pinecone (cacheados para velocidad)
# ------------------------------------------------------------------
@st.cache_resource
def obtener_cliente_gemini():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"], 
        http_options=HttpOptions(api_version="v1")
    )

@st.cache_resource
def obtener_indice_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index(st.secrets["PINECONE_INDEX_NAME"])

cliente_gemini = obtener_cliente_gemini()
indice_pinecone = obtener_indice_pinecone()

# ------------------------------------------------------------------
# Configuración del Motor RAG (Ultra-rápido y económico)
# ------------------------------------------------------------------
MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GENERACION = "gemini-3.5-flash-lite"  # ⚡ El modelo más rápido y resistente del catálogo
DIMENSION_EMBEDDING = 768
NUM_FRAGMENTOS_CONTEXTO = 2  # 📉 Reducido a 2 para procesar a máxima velocidad y ahorrar tokens

INSTRUCCION_SISTEMA = """
Eres un asistente legal sintético. Tu único objetivo es responder de forma directa, extremadamente breve y concisa.

Reglas obligatorias:
1. Responde en un máximo de 2 o 3 frases cortas usando EXCLUSIVAMENTE el CONTEXTO provisto.
2. Ve directo al grano. Elimina introducciones, saludos, fórmulas de cortesía ("Claro", "Basado en el contexto...") o conclusiones.
3. Si el CONTEXTO no contiene la respuesta exacta, di únicamente: "Información no disponible en el texto regulador." y detén tu respuesta.
4. Está estrictamente prohibido transcribir leyes enteras, citar artículos textualmente de forma extensa o mencionar tus fuentes de datos.
"""

# ------------------------------------------------------------------
# Funciones lógicas
# ------------------------------------------------------------------
def obtener_embedding(texto: str):
    resultado = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config={"output_dimensionality": DIMENSION_EMBEDDING},
    )
    # CORREGIDO: Accedemos al primer elemento de la lista devuelta por Google
    return resultado.embeddings[0].values

def buscar_contexto(vector_consulta) -> str:
    resultados = indice_pinecone.query(
        vector=vector_consulta, 
        top_k=NUM_FRAGMENTOS_CONTEXTO, 
        include_metadata=True
    )
    fragmentos = [
        match.metadata.get("texto", "") 
        for match in resultados.matches 
        if match.metadata and match.metadata.get("texto", "")
    ]
    return "\n\n---\n\n".join(fragmentos)

# ------------------------------------------------------------------
# Interfaz visual de Chat
# ------------------------------------------------------------------
st.title("💬 Asistente Legal")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar el historial de la sesión
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

pregunta_usuario = st.chat_input("Escribe tu consulta legal...")

if pregunta_usuario:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant"):
        # Espacio dinámico para el Streaming
        response_placeholder = st.empty()
        texto_acumulado = ""
        
        try:
            # 1. Recuperar contexto numérico
            vector_pregunta = obtener_embedding(pregunta_usuario)
            contexto = buscar_contexto(vector_pregunta)
            prompt = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta_usuario}"
            
            # 2. Configurar límites estrictos de tokens y creatividad a cero (precisión absoluta)
            configuracion_ia = GenerateContentConfig(
                system_instruction=INSTRUCCION_SISTEMA,
                max_output_tokens=150,  # 🛑 Límite de palabras estricto para economizar tokens
                temperature=0.0         # 🎯 Evita que la IA invente o decore las respuestas
            )
            
            # 3. Llamada en Streaming para respuesta instantánea
            response_stream = cliente_gemini.models.generate_content_stream(
                model=MODELO_GENERACION,
                contents=prompt,
                config=configuracion_ia
            )
            
            for chunk in response_stream:
                if chunk.text:
                    texto_acumulado += chunk.text
                    response_placeholder.markdown(texto_acumulado)
                
        except Exception as error:
            texto_acumulado = f"Consulta pausada por saturación en la red o error técnico. Por favor, reintenta en un momento. ({error})"
            response_placeholder.markdown(texto_acumulado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
