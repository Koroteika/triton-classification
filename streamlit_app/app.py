import os
import time
from typing import Any

import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

CLASS_TRANSLATIONS = {
    "Asian Green Bee-Eater": "Азиатская зеленая щурка",
    "Brown-Headed Barbet": "Буроголовый бородастик",
    "Common Kingfisher": "Обыкновенный зимородок",
}


st.set_page_config(
    page_title="Классификатор изображений Triton",
    page_icon="",
    layout="wide",
)


def get_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict_image(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    files = {"file": (file_name, file_bytes, "image/jpeg")}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def render_probabilities(probabilities: dict[str, float]) -> None:
    rows = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for class_name, score in rows:
        label = CLASS_TRANSLATIONS.get(class_name, class_name)
        st.progress(float(score), text=f"{label} ({class_name}): {score:.4f}")


st.title("Классификатор изображений Triton")

health = get_health()
if health and health.get("status") == "healthy":
    st.success(
        f"API доступен | модель: {health.get('model_name')} | вход: {health.get('input_name')} | выход: {health.get('output_name')}"
    )
else:
    st.error(f"API недоступен: {API_URL}")
    st.stop()

left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader(
        "Загрузите изображение птицы",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

with right:
    if uploaded_file is None:
        st.info("Загрузите изображение, чтобы выполнить классификацию.")
    else:
        if st.button("Выполнить классификацию", type="primary", use_container_width=True):
            started = time.perf_counter()
            try:
                result = predict_image(image_bytes, uploaded_file.name)
            except requests.RequestException as exc:
                st.error(f"Ошибка запроса к API: {exc}")
                st.stop()

            total_ms = (time.perf_counter() - started) * 1000
            prediction = result["predictions"][0]

            predicted_class = prediction["class_name"]
            predicted_label = CLASS_TRANSLATIONS.get(predicted_class, predicted_class)

            st.subheader(f"{predicted_label} ({predicted_class})")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Уверенность", f"{prediction['confidence']:.4f}")
            col_b.metric("Время Triton", f"{result['inference_ms']:.2f} мс")
            col_c.metric("Полный запрос", f"{total_ms:.2f} мс")

            st.write("Вероятности по классам")
            render_probabilities(prediction["probabilities"])

            with st.expander("JSON-ответ API"):
                st.json(result)
