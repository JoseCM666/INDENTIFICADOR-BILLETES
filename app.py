import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
from gtts import gTTS
import os

# 1. CONFIGURACIÓN INICIAL ÚNICA
st.set_page_config(
    page_title="Clasificador de Billetes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS LIMPIO: Evita colapsos y asegura el flujo de la cámara
st.markdown("""
    <style>
        /* Ocultar interfaces web sobrantes */
        header, [data-testid="stHeader"], #root > div:nth-child(1) {
            visibility: hidden;
            height: 0px !important;
            display: none !important;
        }
        
        /* Contenedor principal expandido sin restricciones */
        .main, .block-container {
            padding: 0rem !important;
            margin: 0rem !important;
            max-width: 100% !important;
            min-height: 100vh !important;
        }
        
        .stCameraInput label {
            display: none !important;
        }
        
        /* Forzar al área de la cámara a ocupar la pantalla */
        .stCameraInput, .stCameraInput > div {
            width: 100vw !important;
            height: 100vh !important;
            margin: 0px !important;
            padding: 0px !important;
        }

        .stCameraInput video {
            width: 100vw !important;
            height: 100vh !important;
            object-fit: cover !important;
            border-radius: 0px !important;
            transform: scaleX(1) !important; 
        }
        
        /* BOTÓN NATIVO (Take Photo): Fijado en la base */
        .stCameraInput button {
            position: fixed !important;
            bottom: 20px !important;
            left: 5% !important;
            width: 90vw !important; 
            height: 70px !important;
            background-color: #00cc66 !important; 
            color: white !important;
            font-size: 24px !important;
            font-weight: bold !important;
            border-radius: 16px !important; 
            border: none !important;
            z-index: 999 !important; 
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4) !important;
        }

        /* 🔄 BOTÓN HTML FLOTANTE: Activar Cámara Trasera (Justo arriba de Take Photo) */
        .btn-voltear-fijo {
            position: fixed !important;
            bottom: 105px !important;
            left: 5% !important;
            width: 90vw !important;
            height: 65px !important;
            background-color: #2e8b57 !important;
            color: white !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border-radius: 16px !important;
            border: none !important;
            z-index: 9999 !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4) !important;
            display: flex;
            align-items: center;
            justify-content: center;
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

# Carga optimizada del modelo
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modeloV3.h5')

model = load_my_model()
CLASSES = ["Fondo", "1 Dólar", "10 Dólares", "100 Dólares", "2 Dólares", "5 Dólares", "50 Dólares"]

# Control de estados
if "visor" not in st.session_state: st.session_state.visor = True
if "data_final" not in st.session_state: st.session_state.data_final = None

# --- MÓDULO DE LA CÁMARA (VISOR ACTIVO) ---
if st.session_state.visor:
    
    # Inyección del botón HTML + JavaScript para forzar el cambio de cámara sin romper Streamlit
    st.markdown("""
        <button class="btn-voltear-fijo" onclick="voltearCamara()">
            🔄 Activar Cámara Trasera
        </button>
        
        <script>
            function voltearCamara() {
                // Busca el botón nativo que Streamlit genera para alternar cámaras
                const botones = document.querySelectorAll('button');
                for (let btn of botones) {
                    // Si el botón contiene el icono o texto de cambio de cámara, le da clic de inmediato
                    if (btn.innerHTML.includes('svg') && btn.outerHTML.includes('stCameraInput')) {
                        btn.click();
                        break;
                    }
                }
                // Oculta este botón una vez presionado para dejar libre el de captura
                document.querySelector('.btn-voltear-fijo').style.display = 'none';
            }
        </script>
    """, unsafe_allow_html=True)

    # Visor limpio
    foto_usuario = st.camera_input("Enfoca el billete")
    
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
                
                for _ in range(4):
                    imgs.append(img_base)
                    arr = np.expand_dims(preprocess_input(np.array(img_base)), axis=0)
                    p = model.predict(arr, verbose=0)[0]
                    preds.append(p)
                    clases.append(np.argmax(p))
                
                final_pred = np.mean(preds, axis=0)
                for c in set(clases):
                    if clases.count(c) >= 3:
                        final_pred = np.mean([preds[i] for i, x in enumerate(clases) if x == c], axis=0)
                
                st.session_state.data_final = {"pred": final_pred, "imgs": imgs, "indiv": preds}
                st.rerun()
        else:
            st.session_state.visor = True
            st.rerun()
            
    # --- INTERFAZ DE RESULTADOS ---
    res = st.session_state.data_final
    clase_final = CLASSES[np.argmax(res["pred"])]
    confianza = np.max(res["pred"]) * 100
    
    st.write("---")

    if confianza >= 80.0:
        st.subheader("📊 Resultado Promediado")
        st.metric(label="Denominación Detectada", value=clase_final, delta=f"{confianza:.2f}% confianza")
        st.success("✅ ¡Billete identificado!")
        
        mensaje = f"El billete es de {clase_final}."
        reproducir_voz(mensaje)
        
        with st.expander("Ver capturas de la ráfaga"):
            cols = st.columns(4)
            preds_indiv = res.get("indiv", [])
            for i, img in enumerate(res["imgs"]):
                if len(preds_indiv) > i:
                    confianza_ind = np.max(preds_indiv[i]) * 100
                    clase_ind = CLASSES[np.argmax(preds_indiv[i])]
                    with cols[i]:
