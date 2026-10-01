import streamlit as st
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from pinecone import Pinecone
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

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
st.set_page_config(page_title="Asistente Legal", page_icon=icono_pagina)

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
NUM_FRAGMENTOS_CONTEXTO = 1  # 📈 el sistema recupera el artículo exacto que responde a tu pregunta de golpe, sin que le falte ningún matiz.
VENTANA_HISTORIAL = 8  # 🧠 Nº de mensajes previos (aprox. 4 turnos) que se envían como contexto conversacional
LIMITE_INACTIVIDAD = timedelta(minutes=60)  # ⏱️ Tras este tiempo sin interacción, se reinicia la conversación

INSTRUCCION_SISTEMA = """
¡ATENCIÓN! REGLA SUPREMA E INQUEBRANTABLE SOBRE PUNTOS: Para el artículo 118 del RGC (falta de guantes o calzado adecuado en motocicletas), la pérdida de puntos es SIEMPRE "0 puntos".

Eres un asistente legal experto. Tienes TRES fuentes de conocimiento, que no deben mezclarse:

A) CONTEXTO DOCUMENTAL Y MARCO GENERAL: fragmentos normativos recuperados (tráfico, VMP, espectáculos públicos, horarios, código penal y seguridad ciudadana).

B) FÓRMULAS Y BAREMOS FIJOS DE VELOCIDAD Y ALCOHOLEMIA (B.1 a B.4).

C) BAREMOS Y REGLAS CONDICIONALES PARA VMP Y VPL.

CLASIFICACIÓN PREVIA (Identifica el modo según la pregunta):

1. MODO TRÁFICO/MULTA (infracciones de circulación, velocidad, alcohol, VMP):
   - Usa obligatoriamente el formato de lista fija (Norma y artículo, Infracción, Cálculo, Cuantía, Puntos, Responsable, Comentario).

2. MODO HORARIOS Y LICENCIAS (establecimientos públicos, bares, pubs, horarios):
   - Responde de forma directa y concisa en un par de líneas (categoría, horario exacto de cierre y tiempo de desalojo). Prohibido usar plantilla de multas.

3. MODO CÓDIGO PENAL (delitos contra la seguridad vial, lesiones, desobediencia, etc.):
   - Responde en prosa jurídica clara y directa. Detalla el artículo del Código Penal, la conducta típica y las penas asociadas (prisión, multas en cuotas, trabajos en beneficio de la comunidad o privación del carné), sin inventar puntos ni reducciones administrativas.

4. MODO SEGURIDAD CIUDADANA (Ley Orgánica 4/2015):
   - Responde indicando el artículo, la descripción de la infracción (leve, grave o muy grave) y el rango de sanción económica o medidas accesorias (como incautaciones), de forma concisa y sin estructuras de tráfico.

Reglas de respuesta generales:
- REGLA DE ORO: Si faltan datos críticos para resolver un caso, pide los datos en una sola frase corta en lugar de inventar.
- Ve directo al grano. Elimina introducciones, saludos o fórmulas de cortesía.
- Si una materia no contempla puntos del carné (como en Código Penal o LO 4/2015), no menciones los puntos ni fuerces campos ajenos a la norma.
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

if "ultima_interaccion" not in st.session_state:
    st.session_state.ultima_interaccion = datetime.now()

if datetime.now() - st.session_state.ultima_interaccion > LIMITE_INACTIVIDAD:
    st.session_state.mensajes = []

st.session_state.ultima_interaccion = datetime.now()

with st.sidebar:
    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()

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
            prompt_actual = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta_usuario}"

            # 1b. Construir la ventana de historial conversacional (sin la pregunta actual,
            #     que ya está incluida en session_state.mensajes y se añade al final con su contexto)
            mensajes_previos = st.session_state.mensajes[:-1][-VENTANA_HISTORIAL:]
            contents_conversacion = []
            for mensaje_previo in mensajes_previos:
                rol_gemini = "model" if mensaje_previo["role"] == "assistant" else "user"
                contents_conversacion.append(
                    {"role": rol_gemini, "parts": [{"text": mensaje_previo["content"]}]}
                )
            contents_conversacion.append({"role": "user", "parts": [{"text": prompt_actual}]})

            # 2. Configurar límites estrictos de tokens y creatividad a cero (precisión absoluta)
            configuracion_ia = GenerateContentConfig(
                system_instruction=INSTRUCCION_SISTEMA,
                max_output_tokens=800,  # 🛑 Margen para cálculos paso a paso (corrección + tabla) sin cortar la respuesta
                temperature=0.4         # 🎯 Evita que la IA invente o decore las respuestas
            )

            # 3. Llamada en Streaming para respuesta instantánea
            response_stream = cliente_gemini.models.generate_content_stream(
                model=MODELO_GENERACION,
                contents=contents_conversacion,
                config=configuracion_ia
            )

            for chunk in response_stream:
                texto_acumulado += chunk.text
                response_placeholder.markdown(texto_acumulado)

        except Exception as error:
            texto_acumulado = f"Consulta pausada por saturación en la red. Por favor, reintenta en un momento. ({error})"
            response_placeholder.markdown(texto_acumulado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_acumulado})
