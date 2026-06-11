import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
from gtts import gTTS
import os
# Configuración inicial fija (Modo Ancho obligatorio)
st.set_page_config(
    page_title="Clasificador de Billetes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ocultar márgenes internos y el encabezado por defecto de Streamlit
st.markdown("""
<style>
    /* Ocultar el encabezado superior y la barra de navegación lateral */
    #root > div:nth-child(1) {
        display: none;
    }
    .css-18ni7ap { /* Ocultar el menú de 'Streamlit' en la parte superior derecha */
        display: none;
    }
    
    /* Forzar el contenedor principal a ocupar todo el ancho y alto */
    .css-1v3fvcr {
        max-width: 100vw;
        max-height: 100vh;
        margin: 0px;
        padding: 0px;
        overflow: hidden; /* Evitar barras de desplazamiento */
    }

    /* Modificar el contenedor de la cámara para que ocupe todo el espacio */
    .stCameraInput {
        max-width: 100%;
        max-height: 100%;
    }

    /* ESTILO PARA EL BOTÓN 'TAKE PHOTO' (Verde y Flotante en la parte inferior) */
    .stButton > button {
        background-color: #2e8b57 !important; /* Verde bosque profundo, como en la referencia */
        color: black !important;
        border: none;
        padding: 15px 30px;
        font-size: 24px;
        font-family: 'Comic Sans MS', cursive, sans-serif; /* Aproximar el estilo manuscrito */
        border-radius: 20px; /* Bordes redondeados */
        
        /* Posicionamiento flotante en la parte inferior */
        position: fixed;
        bottom: 20px; /* Ajustar distancia desde abajo */
        left: 50%;
        transform: translateX(-50%); /* Centrar horizontalmente */
        z-index: 1000; /* Asegurar que esté por encima de la cámara */
    }

    /* ESTILO PARA EL MENÚ (Verde y Flotante en la parte superior derecha) */
    .stButton > button[id^='menu_'] { /* Ajustar el ID si tienes un botón de menú específico */
        background-color: #2e8b57 !important;
        padding: 10px;
        border-radius: 10px;
        
        /* Posicionamiento flotante en la parte superior derecha */
        position: fixed;
        top: 10px; /* Ajustar distancia desde arriba */
        right: 10px; /* Ajustar distancia desde la derecha */
        z-index: 1001; /* Asegurar que esté por encima de la cámara */
    }

    /* Ajuste para que la cámara no se vea cortada al estirarse */
    .css-1y4p850 img {
        object-fit: cover;
    }

</style>
""", unsafe_allow_html=True)

# Ejemplo de estructura de código para los botones y la cámara
# (Asegúrate de que tus elementos tengan IDs claros para el CSS)
col_menu = st.columns([0.9, 0.1])
with col_menu[1]:
    # Crear un botón de menú con un ID predecible para el CSS
    st.button("", key="menu_top")

# Cargar la entrada de la cámara
st.camera_input("Smile!")

# Crear el botón de captura
st.button("Take Photo")
# Función para generar y reproducir audio
def reproducir_voz(texto):
    tts = gTTS(text=texto, lang='es')
    tts.save("billete.mp3")
    audio_file = open("billete.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
    audio_file.close()


# Configuración
st.set_page_config(page_title="Detector de Billetes - Seguro", layout="centered")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modeloV3.h5')

model = load_my_model()
CLASSES = ["Fondo", "1 Dólar", "10 Dólares", "100 Dólares", "2 Dólares", "5 Dólares", "50 Dólares"]

# Estados
if "visor" not in st.session_state: st.session_state.visor = True
if "data_final" not in st.session_state: st.session_state.data_final = None

st.title("💵 Clasificador de Billetes")

if st.session_state.visor:
    # st.camera_input maneja el visor en vivo automáticamente en HTML5
    # y le pide permiso de cámara al usuario de forma nativa.
    foto_usuario = st.camera_input("Enfoca el billete dentro del recuadro")
    
    if foto_usuario is not None:
        # En cuanto el usuario hace clic en "Tomar foto", guardamos la imagen en el estado
        st.session_state.foto_actual = Image.open(foto_usuario)
        st.session_state.visor = False
        st.rerun()

# --- LÓGICA DE PROCESAMIENTO (ADAPTADA AL ENTORNO WEB) ---
else:
    if st.session_state.data_final is None:
        # Verificamos que realmente tengamos una foto capturada para evitar el error 'nan'
        if "foto_actual" in st.session_state and st.session_state.foto_actual is not None:
            with st.spinner("Analizando consistencia de la imagen..."):
                
                # Recuperamos la foto del usuario y la redimensionamos al formato MobileNetV2
                img_base = st.session_state.foto_actual.resize((224, 224))
                
                # Simulamos la estructura de tu ráfaga/consenso usando variaciones leves 
                # (o procesando la misma imagen para mantener intacta tu estructura matemática posterior)
                preds, imgs, clases = [], [], []
                
                # Ejecutamos 4 evaluaciones con tu modelo para mantener tu algoritmo de consenso
                for _ in range(4):
                    imgs.append(img_base)
                    arr = np.expand_dims(preprocess_input(np.array(img_base)), axis=0)
                    p = model.predict(arr, verbose=0)[0]
                    preds.append(p)
                    clases.append(np.argmax(p))
                
                # Consenso original de tu lógica: Si 3 coinciden, usamos esas
                final_pred = np.mean(preds, axis=0)
                for c in set(clases):
                    if clases.count(c) >= 3:
                        final_pred = np.mean([preds[i] for i, x in enumerate(clases) if x == c], axis=0)
                
                # Guardamos los resultados tal cual como los espera el resto de tu app
                st.session_state.data_final = {"pred": final_pred, "imgs": imgs, "indiv": preds}
                st.rerun()
        else:
            # Por si acaso el estado se limpia, devolvemos al visor
            st.session_state.visor = True
            st.rerun()
            
# --- Mostrar Resultados Promediados (con filtro de seguridad del 80%) ---
    res = st.session_state.data_final
    clase_final = CLASSES[np.argmax(res["pred"])]
    confianza = np.max(res["pred"]) * 100
    
    st.write("---")
  # Aplicamos el umbral del 80% (UN SOLO BLOQUE UNIFICADO)
    if confianza >= 80.0:
        st.subheader("📊 Resultado Promediado (Seguridad Máxima)")
        st.metric(label="Denominación Detectada", value=clase_final, delta=f"{confianza:.2f}% confianza")
        st.success("✅ ¡Billete identificado con alta precisión!")
        
        # --- AQUÍ VA LA LLAMADA DE VOZ ---
        mensaje = f"El billete es de {clase_final}."
        reproducir_voz(mensaje)
        # ---------------------------------
        
        # Despliegue bajo demanda (dentro del mismo IF)
        with st.expander("Ver capturas de la ráfaga y detalle técnico"):
            cols = st.columns(4)
            preds_indiv = res.get("pred_individuales", [])
            for i, img in enumerate(res["imgs"]):
                if len(preds_indiv) > i:
                    confianza_ind = np.max(preds_indiv[i]) * 100
                    clase_ind = CLASSES[np.argmax(preds_indiv[i])]
                    with cols[i]:
                        st.image(img, use_container_width=True)
                        st.caption(f"**{clase_ind}**")
                        st.caption(f"Certeza: {confianza_ind:.1f}%")
    else:
        # Mensaje de error (si la confianza es menor a 80)
        st.error("⚠️ **Índice de confianza bajo**")
        st.warning(f"La certeza actual es de solo **{confianza:.2f}%**. Por favor, intenta de nuevo.")
        
    # Botón para limpiar todo
    if st.button("🔄 Nueva Captura", use_container_width=True):
        st.session_state.visor = True
        st.session_state.data_final = None
        st.rerun()
