import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import os
from scipy.ndimage import center_of_mass, shift

# ----------------------------
# 모델 로딩
# ----------------------------
def get_latest_model():
    MODEL_DIR = "saved_models"
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])


model_path = get_latest_model()
model = tf.keras.models.load_model(model_path) if model_path else None

# ----------------------------
# 타이틀
# ----------------------------
st.title("웹캠 숫자 인식기 (MNIST 기반 + 전처리 강화)")
st.markdown("흰 종이에 검은색 펜으로 숫자를 작성한 후 웹캠으로 촬영해보세요. 다양한 전처리 기법을 통해 인식률을 향상시킵니다.")

# ----------------------------
# 웹캠 입력
# ----------------------------
image_data = st.camera_input("숫자가 보이도록 웹캠으로 촬영")

# ----------------------------
# 다중 전처리 함수
# ----------------------------
@st.cache_data
def apply_preprocessing(image_arr):
    results = {}

    # 히스토그램 정규화
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

# ----------------------------
# 예측 처리
# ----------------------------
if image_data is not None and model:
    # 이미지 로드 및 Grayscale 변환
    image = Image.open(image_data)
    st.image(image, caption="입력 이미지", use_column_width=True)

    gray = ImageOps.grayscale(image)
    gray_np = np.array(gray)

    # 전처리 및 예측
    results = apply_preprocessing(gray_np)

    # 최종 결과 선택
    best = max(results.items(), key=lambda x: x[1]['confidence'])
    best_label = best[1]['prediction']
    best_conf = best[1]['confidence']

    st.subheader(f"✅ 최종 예측: **{best_label}** (신뢰도: {best_conf:.2f})")
    st.bar_chart(best[1]['prob'])

    # 전처리 방식별 출력
    st.subheader("🧪 전처리별 결과 비교")
    for method, data in results.items():
        st.markdown(f"### {method} (예측: {data['prediction']}, 신뢰도: {data['confidence']:.2f})")
        st.image(data['processed'], width=120, caption=method)

    # 히트맵 시각화
    st.subheader("🎨 히트맵 (Best Preprocessing 결과)")
    st.image(best[1]['processed'], width=150, clamp=True, channels="GRAY")

elif not model:
    st.warning("❌ 모델이 없습니다. 먼저 saved_models 폴더에 .keras 모델을 넣어주세요.")
else:
    st.info("먼저 웹캠으로 숫자를 촬영해주세요.")
