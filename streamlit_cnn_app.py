# streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
from skimage.filters import threshold_otsu
import cv2
import os
import tensorflow as tf

# 모델 로드 (가장 최신 모델)
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

model_path = get_latest_model()
model = tf.keras.models.load_model(model_path) if model_path else None

# Title
st.title("CNN 숫자 예측기 (MNIST) - 개선 버전")
st.markdown("직접 입력한 숫자를 다양한 전처리로 안정적으로 인식합니다.")

# Canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=30,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas")

# Adaptive Thresholding
@st.cache_data
def apply_preprocessing(image_arr):
    results = {}

    # 히스토그램 평활화
    norm_img = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 전처리 방법들
    methods = {
        "Adaptive Gaussian": cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2),
        "Otsu": cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        "Manual 100": np.where(norm_img > 100, 0, 255).astype("uint8")
    }

    for key, img in methods.items():
        # 중심 이동
        cy, cx = center_of_mass(img)
        shift_y = int(round(img.shape[0] // 2 - cy))
        shift_x = int(round(img.shape[1] // 2 - cx))
        shifted = shift(img, shift=(shift_y, shift_x), mode='constant', cval=0)

        # 정규화 및 reshape
        norm = shifted.astype("float32") / 255.0
        reshaped = norm.reshape(1, 28, 28, 1)

        # 예측
        pred = model.predict(reshaped, verbose=0)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results[key] = {
            "processed": shifted,
            "prediction": pred_class,
            "confidence": confidence,
            "prob": pred[0]
        }

    return results

# 예측 실행
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data[:, :, 0]
    img = Image.fromarray(img.astype("uint8")).convert("L")  # 흑백
    img = ImageOps.invert(img).resize((28, 28))
    arr = np.array(img)

    # PNG 저장
    save_path = os.path.join("saved_models", "last_input.png")
    img.save(save_path)
    st.info(f"입력 이미지를 저장했습니다: {save_path}")

    # 시각화 - 히트맵
    st.subheader("입력 히트맵")
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='hot')
    st.pyplot(fig)

    # 다양한 전처리 → 예측
    results = apply_preprocessing(arr)

    # 결과 출력
    st.subheader("다중 전처리 예측 결과")
    best = max(results.items(), key=lambda x: x[1]['confidence'])
    best_label = best[1]['prediction']
    best_conf = best[1]['confidence']

    st.success(f"최종 예측: **{best_label}** (신뢰도: {best_conf:.2f})")
    st.bar_chart(best[1]['prob'])

    # 전처리별 비교
    for method, data in results.items():
        st.markdown(f"### {method} (예측: {data['prediction']}, 신뢰도: {data['confidence']:.2f})")
        st.image(data['processed'], width=140)

    # 비교용 X_test 이미지 시각화
    from tensorflow.keras.datasets import mnist
    (_, _), (X_test, y_test) = mnist.load_data()
    match_imgs = X_test[y_test == best_label][:5]
    st.subheader("정답 레이블과 같은 MNIST 이미지 샘플")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(match_imgs[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)

elif not model:
    st.warning("모델이 없습니다. 먼저 학습해 주세요.")
