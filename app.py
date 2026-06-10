import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
from gtts import gTTS
import os

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

# --- VISOR Y DISPARO ---
if st.session_state.visor:
    placeholder = st.empty()
    # Botón grande y único
    if st.button("📸 CAPTURAR Y ANALIZAR", type="primary", use_container_width=True):
        st.session_state.visor = False
        st.rerun()
    
    cap = cv2.VideoCapture(0)
    while st.session_state.visor:
        ret, frame = cap.read()
        if ret:
            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(0.05)
    cap.release()

# --- LÓGICA DE PROCESAMIENTO (CON CONSENSO) ---
else:
    if st.session_state.data_final is None:
        with st.spinner("Analizando ráfaga de 4 fotos..."):
            cap = cv2.VideoCapture(0)
            preds, imgs, clases = [], [], []
            for _ in range(4):
                time.sleep(0.5)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224))
                    imgs.append(img)
                    arr = np.expand_dims(preprocess_input(np.array(img)), axis=0)
                    p = model.predict(arr, verbose=0)[0]
                    preds.append(p)
                    clases.append(np.argmax(p))
            cap.release()
            
            # Consenso: Si 3 coinciden, usamos esas
            final_pred = np.mean(preds, axis=0)
            for c in set(clases):
                if clases.count(c) >= 3:
                    final_pred = np.mean([preds[i] for i, x in enumerate(clases) if x == c], axis=0)
            
            st.session_state.data_final = {"pred": final_pred, "imgs": imgs, "indiv": preds}
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