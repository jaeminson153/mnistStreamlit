# 웹캠 교수님 개선버전
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import os
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt
from collections import Counter

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
# 전처리 함수
# ----------------------------
def enhance_contrast(image_arr):
    norm = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    inverted = cv2.bitwise_not(blurred)
    return inverted

def preprocess_and_predict(image_arr):
    results = {}
    enhanced_img = enhance_contrast(image_arr)

    methods = {
        "Adaptive Gaussian": cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2),
        "Otsu": cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        "Manual 100": np.where(enhanced_img > 100, 0, 255).astype("uint8")
    }

    for method_name, binary_img in methods.items():
        # Adaptive 깨짐 방지: 픽셀 수 너무 적으면 제외
        if np.sum(binary_img > 0) < 20:
            continue

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary_img, kernel, iterations=1)

        cy, cx = center_of_mass(dilated)
        shift_y = int(round(dilated.shape[0] // 2 - cy))
        shift_x = int(round(dilated.shape[1] // 2 - cx))
        shifted = shift(dilated, shift=(shift_y, shift_x), mode='constant', cval=0)

        resized = cv2.resize(shifted, (28, 28), interpolation=cv2.INTER_AREA)
        norm = resized.astype("float32") / 255.0
        norm = np.clip(norm, 0.01, 1.0)
        reshaped = norm.reshape(1, 28, 28, 1)

        pred = model.predict(reshaped, verbose=0)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results[method_name] = {
            "processed": resized,
            "prediction": pred_class,
            "confidence": confidence,
            "prob": pred[0]
        }

    return results

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="웹캠 숫자 인식기", layout="centered")
st.title("📷 웹캠 숫자 인식기3 (MNIST 기반 최종 개선 버전)")
st.markdown("흰 종이에 **굵은 검정 펜**으로 숫자를 쓰고 웹캠으로 촬영해주세요.")

image_data = st.camera_input("숫자를 촬영하세요:")

if image_data and model:
    image = Image.open(image_data).convert("L")
    gray_np = np.array(image)

    results = preprocess_and_predict(gray_np)

    predictions = [v["prediction"] for v in results.values()]
    pred_counter = Counter(predictions)

    if len(pred_counter) > 1:
        st.warning("전처리별 예측이 일치하지 않습니다. 혼동 가능성이 있습니다.")

    # 다수결 기반 예측 선택
    voted_label, vote_count = pred_counter.most_common(1)[0]
    voted_candidates = {k: v for k, v in results.items() if v["prediction"] == voted_label}

    # 그 중 신뢰도 가장 높은 방식 선택
    best_method, best_result = max(voted_candidates.items(), key=lambda x: x[1]["confidence"])

    final_label = best_result["prediction"]
    final_conf = best_result["confidence"]
    final_prob = best_result["prob"]

    # ----------------------------
    # 결과 출력
    # ----------------------------
    st.subheader(f"최종 예측: **{final_label}** (신뢰도: {final_conf:.2f})")
    st.caption(f"사용된 전처리 방식: {best_method}")
    st.bar_chart(final_prob)

    # 전처리별 결과 시각화
    st.subheader("전처리별 예측 결과")
    for method, data in results.items():
        st.markdown(f"**{method}** - 예측: {data['prediction']}, 신뢰도: {data['confidence']:.2f}")
        st.image(data['processed'], width=120)

    # 히트맵
    st.subheader("입력 이미지 히트맵")
    fig, ax = plt.subplots()
    ax.imshow(gray_np, cmap='hot')
    ax.axis("off")
    st.pyplot(fig)

elif not model:
    st.warning("모델(.keras)이 없습니다. 먼저 학습한 모델을 `saved_models/` 폴더에 저장하세요.")
else:
    st.info("웹캠으로 숫자 이미지를 먼저 촬영해주세요.")
