import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import io
from scipy.ndimage import center_of_mass, shift
import os


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


# MODEL_PATH = "saved_models/mnist_cnn_model_latest.keras"  # 가장 최근 모델로 경로 수정
# model = tf.keras.models.load_model(MODEL_PATH)


# ----------------------------
# 타이틀
# ----------------------------
st.title("웹캠 숫자 인식기 (MNIST 기반)")
st.markdown("흰 종이에 검은색 펜으로 0~9 숫자를 작성 후 웹캠으로 촬영해보세요.")


# ----------------------------
# 웹캠 입력
# ----------------------------
image_data = st.camera_input("숫자가 보이도록 웹캠으로 촬영")


if image_data is not None:
    # 이미지 로드
    image = Image.open(image_data)
    st.image(image, caption="입력 이미지", use_column_width=True)


    # ----------------------------
    # 이미지 전처리
    # ----------------------------
    # RGB → 그레이스케일
    gray = ImageOps.grayscale(image)


    # numpy 변환
    gray_np = np.array(gray)


    # Adaptive 이진화 or Otsu
    _, bin_img = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 중심 정렬
    cy, cx = center_of_mass(bin_img)
    shift_y = int(bin_img.shape[0] // 2 - cy)
    shift_x = int(bin_img.shape[1] // 2 - cx)
    shifted_img = shift(bin_img, shift=(shift_y, shift_x), mode='constant', cval=0)


    # 리사이즈 → 28x28
    resized = Image.fromarray(shifted_img.astype("uint8")).resize((28, 28))


    # 정규화 및 차원 확장
    input_arr = np.array(resized).astype("float32") / 255.0
    input_arr = input_arr.reshape(1, 28, 28, 1)


    # ----------------------------
    # 예측
    # ----------------------------
    pred = model.predict(input_arr, verbose=0)
    pred_class = np.argmax(pred)


    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])


    # 히트맵 시각화
    st.subheader("입력 이미지 히트맵")
    st.image(input_arr.reshape(28, 28), width=150, clamp=True, channels="GRAY")


else:
    st.info("먼저 웹캠으로 숫자를 촬영해주세요.")




