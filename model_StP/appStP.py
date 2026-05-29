import streamlit as st
import os

st.set_page_config(page_title="Аналитика модели Погоды", layout="wide")

st.title("Панель мониторинга и аналитики: Weather Image Recognition")
st.write("На этой странице представлена финальная статистика обучения архитектуры ResNet18.")

st.markdown("---")

# Метрики верхнего уровня на основе вашего успешного прогона
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Размер датасета", value="6,862")
with col2:
    st.metric(label="Количество классов", value="11 классов")
with col3:
    st.metric(label="Время обучения (5 эпох)", value="4 мин 23 сек")
with col4:
    # Здесь мы указываем вашу точную точность с графиков
    st.metric(label="Итоговая точность (Accuracy)", value="91.62%")

st.markdown("---")
st.subheader("Результаты и кривые обучения")

c1, c2 = st.columns(2)
with c1:
    if os.path.exists("learning_curves.png"):
        st.image("learning_curves.png", caption="Кривые потерь (Loss) и точности (Accuracy) по эпохам")
    else:
        st.warning("⚠️ Файл 'learning_curves.png' не найден в папке model_StP. Убедитесь, что сохранили его.")

with c2:
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Матрица ошибок (Confusion Matrix Heatmap)")
    else:
        st.warning("⚠️ Файл 'confusion_matrix.png' не найден в папке model_StP. Убедитесь, что сохранили его.")

st.markdown("---")
st.info("Перейдите в боковое меню слева на страницу **Weather Classification**, чтобы протестировать модель на пользовательских фотографиях!")