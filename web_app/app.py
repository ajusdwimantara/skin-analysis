import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import threading
from tensorflow.lite.python.interpreter import Interpreter
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import tensorflow as tf
import openai
import os
import time

# LOCAL
from dotenv import load_dotenv
load_dotenv()  # This will read .env and set the environment variables


# Initialize client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or hardcode key for testing

# STREAMLIT
# client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_skin_report(top_labels):
    label_text = ', '.join([f'{label}: {prob*100:.2f}%' for label, prob in top_labels])

    # System message to guide GPT's behavior as a dermatologist
    system_message = {
        "role": "system",
        "content": (
            "You are a professional dermatologist. Your task is to analyze skin condition "
            "predictions and explain them clearly to non-medical users. Be concise, accurate, "
            "and informative, while maintaining a friendly and supportive tone."
        )
    }

    # User prompt with the predictions
    user_message = {
        "role": "user",
        "content": (
            f"The following are skin condition predictions with their score indicating how bad they experiencing it (0 will be just a few, 100 will be a lot):\n"
            f"{label_text}.\n\n"
            "Write a short paragraph summarizing the skin condition based on these predictions. "
            "Be informative and user-friendly. "
            "in the end ask the user to consult with Ori Skin for better explanation and what treatment they should have"
        )
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[system_message, user_message],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI API error: {e}"

    
# Lock for thread safety
lock = threading.Lock()
captured_frame = None
cropped_face = None

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels & Colors
MODEL_PATH = "detect.tflite"
LABEL_PATH = "labelmap.txt"
CLASSIFY_MODEL_PATH = "classify.h5"
MIN_CONF = 0.15
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
label_colors = {
    'acne': (0, 0, 255),
    'darkspot': (0, 255, 0),
    'wrinkle': (255, 0, 0),
    'oily': (0, 255, 255),
    'dry': (255, 0, 255),
}

# TFLite Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_height = input_details[0]['shape'][1]
in_width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)

# Load classification model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(CLASSIFY_MODEL_PATH)

loaded_model = load_model()
classify_labels = ['acne', 'darkspot', 'dry', 'normal', 'oily', 'other', 'wrinkle']

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
        overlay_path = "guidance_mobile2.jpeg"
        if not os.path.exists(overlay_path):
            print("⚠️ guidance.png NOT FOUND at path:", overlay_path)

        self.overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            print("⚠️ guidance.png exists but failed to load (not a valid image or corrupted)")
        elif self.overlay_img.shape[2] == 4:
            print("ℹ️ Detected 4-channel (RGBA) overlay — flattening to RGB.")
            self.overlay_img = self.flatten_overlay(self.overlay_img)
        else:
            print("ℹ️ Detected 3-channel (RGB) overlay — using as is.")

    def flatten_overlay(self, overlay_rgba, background_color=(0, 0, 0)):
        """Convert RGBA to RGB by blending with a background color."""
        overlay_rgb = overlay_rgba[:, :, :3].astype(float)
        alpha = overlay_rgba[:, :, 3].astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]  # Shape (H, W, 1)

        bg_color = np.full_like(overlay_rgb, background_color, dtype=float)

        blended = overlay_rgb * alpha + bg_color * (1 - alpha)
        return blended.astype(np.uint8)

    def overlay_guidance(self, bg, ov):
        ov = cv2.resize(ov, (bg.shape[1], bg.shape[0]))  # Match camera frame
        return cv2.addWeighted(bg, 0.8, ov, 0.2, 0)  # Semi-transparent blend

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror for webcam effect
        self.frame = img.copy()

        if self.overlay_img is not None:
            img = self.overlay_guidance(img, self.overlay_img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("📸 Skin Analyzer Prototype")

if 'stream_started' not in st.session_state:
    st.session_state.stream_started = False
if 'captured_face' not in st.session_state:
    st.session_state.captured_face = None
if 'face_detected' not in st.session_state:
    st.session_state.face_detected = False
if 'analyze' not in st.session_state:
    st.session_state.analyze = False

if not st.session_state.stream_started:
    if st.button("📹 Start Camera"):
        st.session_state.stream_started = True

ctx = None       
if st.session_state.stream_started and not st.session_state.face_detected:
    ctx = webrtc_streamer(
        key="stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1920},
                "height": {"ideal": 1080},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

def enhance_skin_image(img):
    # 1. Convert BGR to LAB color space (better for luminance adjustment)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Apply CLAHE to L-channel for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # 3. Merge back the LAB channels
    lab = cv2.merge((cl, a, b))

    # 4. Convert LAB back to BGR
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Sharpen image with a moderate kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)

    # 6. Increase saturation in HSV color space
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Increase saturation by 1.3 (30% more), cap max at 255
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    # Slightly increase value channel by 1.1 (10% brighter)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    hsv = hsv.astype(np.uint8)

    # 7. Convert back to BGR
    final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return final_img
        
# Button to capture + analyze
# Only show the capture button if face hasn't been detected yet
if not st.session_state.face_detected and ctx is not None and ctx.video_processor:
    if st.button("📸 Capture & Detect Face"):
        with lock:
            frame = ctx.video_processor.frame
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cropped_face = frame[y:y+h, x:x+w]

                    st.session_state.captured_face = cropped_face
                    st.session_state.face_detected = True
                    st.session_state.analyze = False

                    st.rerun()

                # # Enhance contrast and brightness
                # alpha = 1  # Contrast control (1.0-3.0)
                # beta = -10   # Brightness control (0-100)
                # face_for_detection = cv2.addWeighted(cropped_face, alpha, np.zeros(cropped_face.shape, cropped_face.dtype), 0, beta)

                # # Enhance sharpening
                # # Create the sharpening kernel
                # kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])

                # # Sharpen the image
                # face_for_detection = cv2.filter2D(face_for_detection, -1, kernel)

                # # Enhance colour
                # # Convert the image from BGR to HSV color space
                # image_hsv = cv2.cvtColor(face_for_detection, cv2.COLOR_RGB2HSV)

                # # Adjust the hue, saturation, and value of the image_hsv
                # # Adjusts the hue by multiplying it by 0.7
                # image_hsv[:, :, 0] = image_hsv[:, :, 0] * 1.5
                # # Adjusts the saturation by multiplying it by 1.5
                # image_hsv[:, :, 1] = image_hsv[:, :, 1] * 1
                # # Adjusts the value by multiplying it by 0.5
                # image_hsv[:, :, 2] = image_hsv[:, :, 2] * 0.5

                # # Convert the image back to BGR color space
                # face_for_detection = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

                # COMMENT DULU
                
if st.session_state.face_detected and st.session_state.captured_face is not None:
    st.image(st.session_state.captured_face, channels="BGR", caption="Captured Face")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Retake"):
            st.session_state.face_detected = False
            st.session_state.captured_face = None
            st.session_state.analyze = False
            st.rerun()  # 👈 Force update

    with col2:
        if st.button("✅ OK"):
            st.session_state.analyze = True

    with col3:
        if st.button("❌ Cancel"):
            st.session_state.face_detected = False
            st.session_state.captured_face = None
            st.session_state.analyze = False
            st.session_state.stream_started = False  # 👈 if you want to go back to "Start Camera"
            st.info("Canceled.")
            st.rerun()  # 👈 Force rerun to reflect reset

if st.session_state.analyze and st.session_state.captured_face is not None:
    # st.success("Analyzing…")
    with st.spinner("🧠 Analyzing your skin condition..."):
        progress_bar = st.progress(0)

        cropped_face = st.session_state.captured_face
        time.sleep(0.2)
        progress_bar.progress(10, text="🔧 Enhancing image...")
        processed_face = enhance_skin_image(cropped_face)

        time.sleep(0.2)
        progress_bar.progress(30, text="📐 Resizing + Normalizing...")
        resized = cv2.resize(processed_face, (in_width, in_height))
        input_data = np.expand_dims(resized, axis=0)
        if float_input:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        time.sleep(0.2)
        progress_bar.progress(50, text="📦 Running detection model...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        h, w, _ = processed_face.shape
        for i in range(len(scores)):
            if scores[i] > MIN_CONF:
                ymin = int(max(1, boxes[i][0] * h))
                xmin = int(max(1, boxes[i][1] * w))
                ymax = int(min(h, boxes[i][2] * h))
                xmax = int(min(w, boxes[i][3] * w))
                cls = labels[int(classes[i])]
                color = label_colors.get(cls, (255, 255, 255))
                cv2.rectangle(processed_face, (xmin, ymin), (xmax, ymax), color, 2)

        time.sleep(0.2)
        progress_bar.progress(70, text="🧠 Classifying skin type...")
        resized_face = cv2.resize(processed_face, (224, 224))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_face = np.expand_dims(normalized_face, axis=0)
        y_test = loaded_model.predict(input_face)

        top_indices = y_test[0].argsort()[-3:][::-1]
        top_labels = [
            (classify_labels[i], y_test[0][i])
            for i in top_indices if y_test[0][i] > 0.3
        ]

        time.sleep(0.2)
        progress_bar.progress(90, text="📋 Generating report...")
        if top_labels:
            description = generate_skin_report(top_labels)
            st.markdown("### 📋 Skin Condition Summary")
            st.markdown(description)
            st.markdown("### 🧠 Skin Type Predictions:")
            for label, prob in top_labels:
                st.markdown(f"- **{label}**: {prob * 100:.2f}%")
        else:
            st.markdown("⚠️ No confident prediction was found.")

        st.image(cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB), caption="Cropped Face", use_container_width=True)

        # Legend
        st.markdown("### 🔹 Description")
        patches = [
            Patch(color=np.array(label_colors[name][::-1]) / 255.0, label=name)
            for name in label_colors
        ]
        fig = plt.figure()
        plt.legend(handles=patches, loc='center', ncol=3)
        plt.axis('off')
        st.pyplot(fig)

        time.sleep(0.2)
        progress_bar.progress(100, text="✅ Done!")



# Show cropped face with detections
# if cropped_face is not None:
    # st.image(cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB), caption="Cropped Face", use_container_width=True)

    # # Legend
    # st.markdown("### 🔹 Description")
    # patches = [
    #     Patch(color=np.array(label_colors[name][::-1]) / 255.0, label=name)
    #     for name in label_colors
    # ]
    # fig = plt.figure()
    # plt.legend(handles=patches, loc='center', ncol=3)
    # plt.axis('off')
    # st.pyplot(fig)
