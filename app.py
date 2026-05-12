import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# conf da pagina
st.set_page_config(page_title="Classificação de Emoções", layout="centered")

# Cache para evitar que a pagina recarrege 
@st.cache_resource

# função para carregar  o modelo
def load_emotion_model():
    return tf.keras.models.load_model('emotion_model.keras')

model = load_emotion_model()
class_names = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpresa', 'Neutro']

st.title("Chatbot - Classificação de Emoções")
st.warning("Nota: Este chatbot utiliza tecnologia CNN para análise de imagens. Os resultados são previsões automatizadas e podem conter erros.") 

# para inicializar o historico do chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Me envie uma foto de um rosto para que eu posso classificar :) "}
    ]
    
    
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# para mostrar o historico do chat
for message in st.session_state.messages:
    avatar = message.get("avatar")
    with st.chat_message(message["role"],avatar=avatar):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=300)
        
        if "chart_data" in message:
            st.info(f"Confiança detalhada: {message['confidence']:.2f}%")
            st.bar_chart(message["chart_data"])

# Lógica para aquisição da imagem, conversão para tons de cinza e redimensionamento conforme os requisitos de entrada de acordo com o modelo         
uploaded_file = st.file_uploader("Faça upload da sua imagem aqui para classifica-lá", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and uploaded_file.name != st.session_state.last_processed_file:
        image = Image.open(uploaded_file).convert('L')
        img_resized = image.resize((48, 48))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=(0, -1)) # (1, 48, 48, 1)

        # Predição
        prediction = model.predict(img_array, verbose=0)
        res_idx = np.argmax(prediction)
        emotion = class_names[res_idx]
        conf = np.max(prediction) * 100

        # Envia a imagem do usuario ao chat
        st.session_state.messages.append({
            "role": "user",
            "avatar":"🦖",
            "content": "Classifique esta imagem:", 
            "image": image
        })

        # monta a resposta que o chat responde
        resposta_ai = f"Pela minha análise, essa expressão parece ser de **{emotion}** (Confiança: {conf:.1f}%)."
            
        st.session_state.messages.append({
            "role": "assistant",
            "content": resposta_ai,
            "confidence": conf,
            "chart_data": dict(zip(class_names, prediction[0]))
        })
        
        st.session_state.last_processed_file = uploaded_file.name
        st.rerun()

