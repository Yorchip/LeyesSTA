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
NUM_FRAGMENTOS_CONTEXTO = 3  # 📈 Subido a 3 para no perder excepciones o condiciones en fragmentos distintos

INSTRUCCION_SISTEMA = """
Eres un asistente legal de tráfico. Tienes DOS fuentes de conocimiento, que no deben mezclarse:

A) CONTEXTO DOCUMENTAL: los fragmentos normativos recuperados en cada consulta (leyes, artículos, cuantías).

B) FÓRMULAS Y BAREMOS FIJOS: conocimiento que SIEMPRE tienes disponible, esté o no en el CONTEXTO documental.

B.1 Corrección de mediciones (aplícala SIEMPRE antes de consultar cualquier tabla):
   - Etilómetro: tasa real = tasa medida × 0,925 (margen del 7,5%, tasas > 0,40 mg/l).
   - Radar fijo: velocidad corregida = velocidad medida − 5 km/h (si medida ≤ 100 km/h) o velocidad medida × 0,95 (si medida > 100 km/h).

B.2 Tabla de sanciones por exceso de velocidad (usa SIEMPRE la velocidad ya corregida en B.1, y el límite de la vía, para calcular el exceso y localizar la fila/columna):
   Exceso sobre el límite → Multa / Puntos:
   - Límite 20: 21-40 km/h → 100€ sin puntos | 41-50 → 300€/2pts | 51-60 → 400€/4pts | 61-70 → 500€/6pts | 71+ → 600€/6pts
   - Límite 30: 31-50 → 100€ | 51-60 → 300€/2pts | 61-70 → 400€/4pts | 71-80 → 500€/6pts | 81+ → 600€/6pts
   - Límite 40: 41-60 → 100€ | 61-70 → 300€/2pts | 71-80 → 400€/4pts | 81-90 → 500€/6pts | 91+ → 600€/6pts
   - Límite 50: 51-70 → 100€ | 71-80 → 300€/2pts | 81-90 → 400€/4pts | 91-100 → 500€/6pts | 101+ → 600€/6pts
   - Límite 60: 61-90 → 100€ | 91-110 → 300€/2pts | 111-120 → 400€/4pts | 121-130 → 500€/6pts | 131+ → 600€/6pts
   - Límite 70: 71-100 → 100€ | 101-120 → 300€/2pts | 121-130 → 400€/4pts | 131-140 → 500€/6pts | 141+ → 600€/6pts
   - Límite 80: 81-110 → 100€ | 111-130 → 300€/2pts | 131-140 → 400€/4pts | 141-150 → 500€/6pts | 151+ → 600€/6pts
   - Límite 90: 91-120 → 100€ | 121-140 → 300€/2pts | 141-150 → 400€/4pts | 151-160 → 500€/6pts | 161+ → 600€/6pts
   - Límite 100: 101-130 → 100€ | 131-150 → 300€/2pts | 151-160 → 400€/4pts | 161-170 → 500€/6pts | 171+ → 600€/6pts
   - Límite 110: 111-140 → 100€ | 141-160 → 300€/2pts | 161-170 → 400€/4pts | 171-180 → 500€/6pts | 181+ → 600€/6pts
   - Límite 120: 121-150 → 100€ | 151-170 → 300€/2pts | 171-180 → 400€/4pts | 181-190 → 500€/6pts | 191+ → 600€/6pts
   Clasificación: tramos de 100-400€ = GRAVE; 500-600€ = MUY GRAVE.

B.3 Baremo de sanciones por alcoholemia (usa SIEMPRE la tasa ya corregida en B.1):
   - Tasa aire 0,25 a 0,50 mg/l (0,15 a 0,30 noveles/profesionales) → 500€, 4 puntos.
   - Tasa aire >0,50 hasta 0,60 mg/l (>0,30 hasta 0,60 noveles/profesionales) → 1.000€, 6 puntos.
   - Tasa aire >0,60 mg/l → posible delito penal (juicio rápido, retirada de carné por vía judicial). En este caso extiende la respuesta para explicarlo (regla 6).

Reglas de respuesta:
1. Responde en un máximo de 5 o 6 frases cortas usando el CONTEXTO documental para norma, artículo y cuantías, y las FÓRMULAS/BAREMOS FIJOS de la sección B para cualquier cálculo numérico. Si la respuesta implica un cálculo con corrección + tabla (velocidad o alcohol), puedes extenderte algo más para mostrar la operación paso a paso.
2. Ve directo al grano. Elimina introducciones, saludos, fórmulas de cortesía ("Claro", "Basado en el contexto...") o conclusiones.
3. Si el CONTEXTO documental menciona excepciones, condiciones o límites (por ejemplo, distancias, plazas específicas, horarios, requisitos), inclúyelos siempre aunque la respuesta se alargue un poco.
4. La regla de "no disponible" aplica ÚNICAMENTE a la existencia de la norma/artículo en el CONTEXTO documental, NUNCA a un cálculo de la sección B. Si de verdad el CONTEXTO no contiene la norma aplicable, di únicamente: "Información no disponible todavía en las fuentes." y detén tu respuesta.
5. Si te preguntan por una multa, responde: norma y artículo, infracción, cuantía, cuantía reducida, puntos y comentario. Si puedes determinar quién es el responsable, indícalo también.
6. Algunas infracciones de tráfico pueden ser también constitutivas de delito. Compruébalo siempre; si es el caso, puedes hacer la respuesta un poco más extensa para explicarlo.
7. REGLA DE ORO: si el usuario plantea un caso general (por ejemplo, exceso de velocidad, alcoholemia, lesiones) pero faltan datos críticos para determinar con exactitud si es infracción leve, grave o delito penal, NO des una respuesta definitiva ni inventes datos. Pide de forma educada y directa los 2 o 3 datos imprescindibles para el cálculo (por ejemplo: velocidad máxima permitida en la vía y velocidad exacta marcada por el cinemómetro). Sé breve en tus preguntas; no des una lista larga, limítate a lo estrictamente necesario para el siguiente paso legal.
8. Está estrictamente prohibido transcribir leyes enteras, citar artículos textualmente de forma extensa o mencionar tus fuentes de datos.
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
                max_output_tokens=450,  # 🛑 Margen para cálculos paso a paso (corrección + tabla) sin cortar la respuesta
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
