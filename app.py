import streamlit as st
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone
from pathlib import Path
from PIL import Image

# ------------------------------------------------------------------
# Logo del proyecto
# ------------------------------------------------------------------
# Coloca tu imagen (p. ej. logo.png) en la MISMA carpeta que este app.py.
RUTA_LOGO = "logo.png"
LOGO_DISPONIBLE = Path(RUTA_LOGO).exists()

try:
    icono_pagina = Image.open(RUTA_LOGO) if LOGO_DISPONIBLE else "💬"
except Exception:
    icono_pagina = "💬"

AVATAR_ASISTENTE = RUTA_LOGO if LOGO_DISPONIBLE else "🤖"

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------
st.set_page_config(page_title="Asistente legal PLV", page_icon=icono_pagina)

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

    if LOGO_DISPONIBLE:
        st.image(RUTA_LOGO, width=90)

    st.title("🔒 Acceso restringido")
    st.text_input(
        "Introduce la contraseña de acceso",
        type="password",
        on_change=password_ingresada,
        key="password_input",
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
        http_options=HttpOptions(api_version="v1"),
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
NUM_FRAGMENTOS_CONTEXTO = 3  # 📈 Subido a 3 para no perder excepciones o condiciones en fragmentos distintos

INSTRUCCION_SISTEMA = """
Eres un asistente legal sintético. Tu único objetivo es responder de forma directa y concisa, sin omitir matices legales relevantes.

Reglas obligatorias:
1. Responde en un máximo de 5 o 6 frases cortas usando EXCLUSIVAMENTE el CONTEXTO provisto.
2. Ve directo al grano. Elimina introducciones, saludos, fórmulas de cortesía ("Claro", "Basado en el contexto...") o conclusiones.
3. Si el CONTEXTO menciona excepciones, condiciones o límites (por ejemplo, distancias, plazas específicas, horarios, requisitos), inclúyelos siempre aunque la respuesta se alargue un poco.
4. Si el CONTEXTO no contiene la respuesta exacta, di únicamente: "Información no disponible todavía en las fuentes." y detén tu respuesta.
5. Si te preguntan por una multa, responde norma y artículo, infracción, cuantía, cuantía reducida, puntos y comentario. Si puedes decir quien es el responsable, también.
6. Algunas infracciones de tráfico pueden ser también constitutivas de delito. Compruébalo también, si es así, puedes hacer la respuesta un poco más extensa.
Regla de oro: Si el usuario te plantea un caso general (ej. exceso de velocidad, alcoholemia, lesiones) pero faltan datos críticos para determinar con exactitud si es una infracción leve, grave o un delito penal:
NO des una respuesta definitiva ni inventes datos.
Pide de forma educada y directa los 2 o 3 datos imprescindibles que necesitas para hacer el cálculo legal correcto (por ejemplo: velocidad de la vía, velocidad marcada, tipo de radar, etc.).
Sé breve en tus preguntas. No des una lista de 10 preguntas; limítate a las estrictamente necesarias para el siguiente paso legal.
Ejemplo:
Usuario: "Iba con el coche a más velocidad de la permitida."
Para determinar si es una infracción administrativa o un delito (art. 379 del Código Penal), necesito un par de datos: ¿Cuál era la velocidad máxima permitida en esa vía y qué velocidad exacta registró el cinemómetro (radar)?"
Instrucción obligatoria de cálculo: Los etilómetros tienen un margen de error del 7,5% para tasas > 0,40 mg/l (multiplica la tasa por 0,925). Los radares fijos restan 5 km/h ($\le 100$) o 5% ($> 100$). NUNCA digas que la información no está disponible si el usuario te da los datos base; aplica de inmediato estas fórmulas matemáticas oficiales de la Orden ITC/155/2020 y muestra la operación paso a paso."

7. Está estrictamente prohibido mencionar tus fuentes de datos.

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
columna_logo, columna_titulo = st.columns([1, 6])
with columna_logo:
    if LOGO_DISPONIBLE:
        st.image(RUTA_LOGO, width=60)
with columna_titulo:
    st.title("Asistente Legal")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar el historial de la sesión
for mensaje in st.session_state.mensajes:
    avatar = AVATAR_ASISTENTE if mensaje["role"] == "assistant" else None
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

pregunta_usuario = st.chat_input("Escribe tu consulta legal...")

if pregunta_usuario:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
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
                max_output_tokens=280,  # 🛑 Margen para incluir excepciones/condiciones sin disparar la longitud
                temperature=0.0         # 🎯 Evita que la IA invente o decore las respuestas
            )

            # 3. Llamada en Streaming para respuesta instantánea
            response_stream = cliente_gemini.models.generate_content_stream(
                model=MODELO_GENERACION,
                contents=prompt,
                config=configuracion_ia
            )

            for chunk in response_stream:
                texto_acumulado += chunk.text
                response_placeholder.markdown(texto_acumulado)

        except Exception as error:
            texto_acumulado = f"Consulta pausada por saturación en la red. Por favor, reintenta en un momento. ({error})"
            response_placeholder.markdown(texto_acumulado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
