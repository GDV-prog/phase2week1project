import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import time
import os


@st.cache_resource
def load_weather_model():
    # ---------- ВСТАВЛЕННЫЙ БЛОК ----------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, "best_model.pth")
    class_file = os.path.join(SCRIPT_DIR, "class_names.txt")
    # --------------------------------------

    # Загрузка имён классов из txt-файла (построчно)
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        st.error(f"Файл {class_file} не найден! Использую стандартный список.")
        classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

    # Инициализация модели ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(num_ftrs, len(classes))
    )

    # Загрузка весов из best_model.pth
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success(f"Модель загружена из {model_path}")
    else:
        st.error(f"Файл весов не найден по пути: {model_path}")

    model.eval()
    return model, classes
