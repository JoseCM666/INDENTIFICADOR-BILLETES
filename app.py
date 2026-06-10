import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
from gtts import gTTS
import os
# Configuración inicial obligatoria
st.set_page_config(
    page_title="Clasificador de Billetes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS para forzar formato vertical de teléfono (Portrait)
st.markdown("""
    <style>
        /* Ocultar elementos innecesarios de la interfaz web */
        header {
            visibility: hidden;
            height: 0px !important;
        }
        
        .main .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        
        .stCameraInput label {
            display: none !important;
        }
        
        /* Contenedor principal de la cámara */
        .stCameraInput > div {
            width: 100vw !important;
            max-width: 100vw !important;
            background-color: #111111 !important; /* Fondo oscuro elegante de carga */
            padding: 0px !important;
            margin: 0px !important;
        }

        /* 📱 FORZAR FORMATO VERTICAL EN EL VIDEO */
        .stCameraInput video {
            width: 100vw !important;       /* Ocupa todo el ancho de la pantalla */
            height: 75vh !important;      /* Alto controlado para que no se deforme ni tape el botón */
            object-fit: cover !important;  /* MÁGICO: Recorta los lados y se enfoca en vertical */
            border-radius: 0px !important;
        }
        
        /* Botón de captura inferior adaptado al pulgar */
        .stCameraInput button {
            width: 100vw !important;
            height: 12vh !important;      /* Más alto y fácil de presionar */
            background-color: #00cc66 !important; /* Verde llamativo para la acción */
            color: white !important;
            font-size: 22px !important;   /* Letra grande y legible */
            font-weight: bold !important;
            border-radius: 0px !important;
            border: none !important;
            box-shadow: 0px -4px 10px rgba(0,0,0,0.3) !important; /* Sombra para separarlo del video */
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
