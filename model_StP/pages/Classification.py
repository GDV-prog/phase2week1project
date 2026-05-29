import streamlit as st
import torch
import torchvision.models as models
import os
from PIL import Image
from utils import load_image_from_url, predict

st.set_page_config(page_title="Классификатор погоды", page_icon="🌤️", layout="wide")
st.title("🌤️ Распознавание типов погоды")
st.write("Загружайте изображения локально или используйте прямые ссылки.")

@st.cache_resource
def load_weather_model():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    
    # СТРОГИЙ АЛФАВИТНЫЙ ПОРЯДОК КЛАССОВ ИЗ ВАШЕГО НОУТБУКА:
    classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
    
    # Точная структура с Dropout, как в обученной модели
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(num_ftrs, len(classes))
    )
    
    weights_path = "weights/resnet18.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        st.error(f"Файл весов не найден по пути: {weights_path}")
        
    model.eval()
    return model, classes

model, class_names = load_weather_model()

# --- Интерфейс ---
st.subheader("Источник данных")
upload_type = st.radio("Выберите метод загрузки картинки:", ("Локальный файл (Мультизагрузка)", "Прямая ссылка (URL)"))

images_to_process = []

if upload_type == "Локальный файл (Мультизагрузка)":
    uploaded_files = st.file_uploader(
        "Перетащите файлы сюда или нажмите обзор", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            images_to_process.append(Image.open(file).convert('RGB'))

elif upload_type == "Прямая ссылка (URL)":
    url_input = st.text_input("Вставьте прямую ссылку на изображение:")
    if url_input:
        img = load_image_from_url(url_input)
        if img:
            images_to_process.append(img)

# --- Вывод результатов ---
if images_to_process:
    st.markdown("---")
    st.subheader("Результаты распознавания")
    
    cols = st.columns(min(len(images_to_process), 3))
    for idx, img in enumerate(images_to_process):
        with cols[idx % 3]:
            st.image(img, use_container_width=True, caption=f"Изображение #{idx+1}")
            label, confidence, inf_time = predict(img, model, class_names)
            
            st.success(f"**Класс:** `{label}`")
            st.info(f"**Доверие сети:** {confidence:.2%}")
            st.warning(f"**Время инференса:** {inf_time:.4f} сек")