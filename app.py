import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
from gtts import gTTS
import os

# 1. CONFIGURACIÓN INICIAL FIJA (Debe ser la primera instrucción de Streamlit)
st.set_page_config(
    page_title="Clasificador de Billetes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. INYECCIÓN DE CSS AVANZADO PARA PANTALLA COMPLETA VERTICAL (Estilo App Nativa)
st.markdown("""
    <style>
        /* Eliminar la interfaz web por defecto, el encabezado y el menú de Streamlit */
        header {
            visibility: hidden;
            height: 0px !important;
        }
        
        #root > div:nth-child(1) {
            display: none !important;
        }
        
        /* Forzar al contenedor principal a usar el 100% sin márgenes ni scrollbars */
        .main, .block-container {
            padding: 0rem !important;
            margin: 0rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }
        
        /* Ocultar etiquetas de texto automáticas del componente */
        .stCameraInput label {
            display: none !important;
        }
        
        /* Expandir el contenedor de la cámara al máximo de la pantalla */
        .stCameraInput > div {
            width: 100vw !important;
            max-width: 100vw !important;
            height: 100vh !important;
            padding: 0px !important;
            margin: 0px !important;
            background-color: #000000 !important;
        }

        /* Forzar al video de la cámara trasera a cubrir toda la pantalla vertical */
        .stCameraInput video {
            width: 100vw !important;
            height: 100vh !important;       
            object-fit: cover !important;    
            border-radius: 0px !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            z-index: 1 !important;           
        }
        
        /* Convertir el botón original de captura en un botón flotante abajo */
        .stCameraInput button {
            position: absolute !important;
            bottom: 0px !important;          
            left: 0px !important;
            width: 100vw !important;
            height: 14vh !important;         /* Espacio amplio adaptado al pulgar */
            background-color: rgba(0, 204, 102, 0.9) !important; /* Verde semitransparente llamativo */
            color: white !important;
            font-size: 26px !important;    
            font-weight: bold !important;
            border-radius: 0px !important;
            border: none !important;
            z-index: 10 !important;          /* Se dibuja por encima de la cámara */
            backdrop-filter: blur(5px) !important; 
        }
    </style>
""", unsafe_allow_html=True)

# Función para generar y reproducir audio
def reproducir_voz(texto):
    tts = gTTS(text=texto, lang='es')
    tts.save("billete.mp3")
    audio_file = open("billete.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
    audio_file.close()

# Carga optimizada del modelo de TensorFlow
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modeloV3.h5')

model = load_my_model()
CLASSES = ["Fondo", "1 Dólar", "10 Dólares", "100 Dólares", "2 Dólares", "5 Dólares", "50 Dólares"]

# Control de estados de la aplicación
if "visor" not in st.session_state: st.session_state.visor = True
if "data_final" not in st.session_state: st.session_state.data_final = None

# Título (Se mantiene oculto en el visor gracias al CSS de pantalla completa)
st.title("💵 Clasificador de Billetes")

# --- MÓDULO DE LA CÁMARA (VISOR ACTIVO) ---
if st.session_state.visor:
    foto_usuario = st.camera_input("Enfoca el billete dentro del recuadro")
    
    if foto_usuario is not None:
        st.session_state.foto_actual = Image.open(foto_usuario)
        st.session_state.visor = False
        st.rerun()

# --- MÓDULO DE PROCESAMIENTO E INFERENCIA ---
else:
    if st.session_state.data_final is None:
        if "foto_actual" in st.session_state and st.session_state.foto_actual is not None:
            with st.spinner("Analizando consistencia de la imagen..."):
                
                img_base = st.session_state.foto_actual.resize((224, 224))
                preds, imgs, clases = [], [], []
                
                # Ejecutamos las 4 evaluaciones utilizando tu estructura de ráfaga
                for _ in range(4):
                    imgs.append(img_base)
                    arr = np.expand_dims(preprocess_input(np.array(img_base)), axis=0)
                    p = model.predict(arr, verbose=0)[0]
                    preds.append(p)
                    clases.append(np.argmax(p))
                
                # Algoritmo de consenso original
                final_pred = np.mean(preds, axis=0)
                for c in set(clases):
                    if clases.count(c) >= 3:
                        final_pred = np.mean([preds[i] for i, x in enumerate(clases) if x == c], axis=0)
                
                st.session_state.data_final = {"pred": final_pred, "imgs": imgs, "indiv": preds}
                st.rerun()
        else:
            st.session_state.visor = True
            st.rerun()
            
    # --- INTERFAZ DE RESULTADOS (PANTALLA POST-CAPTURA) ---
    res = st.session_state.data_final
    clase_final = CLASSES[np.argmax(res["pred"])]
    confianza = np.max(res["pred"]) * 100
    
    st.write("---")

    # Filtro de seguridad unificado al 80%
    if confianza >= 80.0:
        st.subheader("📊 Resultado Promediado (Seguridad Máxima)")
        st.metric(label="Denominación Detectada", value=clase_final, delta=f"{confianza:.2f}% confianza")
        st.success("✅ ¡Billete identificado con alta precisión!")
        
        # Ejecución del asistente de voz por gTTS
        mensaje = f"El billete es de {clase_final}."
        reproducir_voz(mensaje)
        
        # Despliegue de analíticas de ráfaga
        with st.expander("Ver capturas de la ráfaga y detalle técnico"):
            cols = st.columns(4)
            preds_indiv = res.get("indiv", []) # Corrección de clave para evitar error nulo
            for i, img in enumerate(res["imgs"]):
                if len(preds_indiv) > i:
                    confianza_ind = np.max(preds_indiv[i]) * 100
                    clase_ind = CLASSES[np.argmax(preds_indiv[i])]
                    with cols[i]:
                        st.image(img, use_container_width=True)
                        st.caption(f"**{clase_ind}**")
                        st.caption(f"Certeza: {confianza_ind:.1f}%")
    else:
        st.error("⚠️ **Índice de confianza bajo**")
        st.warning(f"La certeza actual es de solo **{confianza:.2f}%**. Por favor, intenta de nuevo.")
        
    # Botón principal en la base para restablecer el flujo
    if st.button("🔄 Nueva Captura", use_container_width=True):
        st.session_state.visor = True
        st.session_state.data_final = None
        st.rerun()
