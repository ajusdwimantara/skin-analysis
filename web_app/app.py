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

# Lock for thread safety
lock = threading.Lock()
captured_frame = None
cropped_face = None

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels & Colors
MODEL_PATH = "detect.tflite"
LABEL_PATH = "labelmap.txt"
CLASSIFY_MODEL_PATH = "classify2.keras"
MIN_CONF = 0.2
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
loaded_model = tf.keras.models.load_model(CLASSIFY_MODEL_PATH)
classify_labels = ['acne', 'darkspot', 'dry', 'normal', 'oily', 'other', 'wrinkle']

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("📸 Real-time Face Crop + Skin Detection")

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

# Button to capture + analyze
if ctx.video_processor and st.button("📸 Capture & Detect Face"):
    with lock:
        frame = ctx.video_processor.frame
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped_face = frame[y:y+h, x:x+w]

                # Resize + Normalize for object detection
                resized = cv2.resize(cropped_face, (in_width, in_height))
                input_data = np.expand_dims(resized, axis=0)
                if float_input:
                    input_data = (np.float32(input_data) - 127.5) / 127.5

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                boxes = interpreter.get_tensor(output_details[1]['index'])[0]
                classes = interpreter.get_tensor(output_details[3]['index'])[0]
                scores = interpreter.get_tensor(output_details[0]['index'])[0]

                # Draw object detection results
                for i in range(len(scores)):
                    if scores[i] > MIN_CONF:
                        ymin = int(max(1, boxes[i][0] * h))
                        xmin = int(max(1, boxes[i][1] * w))
                        ymax = int(min(h, boxes[i][2] * h))
                        xmax = int(min(w, boxes[i][3] * w))
                        cls = labels[int(classes[i])]
                        color = label_colors.get(cls, (255, 255, 255))
                        cv2.rectangle(cropped_face, (xmin, ymin), (xmax, ymax), color, 2)

                # --- Classification ---
                resized_face = cv2.resize(cropped_face, (224, 224))
                normalized_face = resized_face.astype(np.float32) / 255.0
                input_face = np.expand_dims(normalized_face, axis=0)
                y_test = loaded_model.predict(input_face)
                top_indices = y_test[0].argsort()[-3:][::-1]
                top_labels = [(classify_labels[i], y_test[0][i]) for i in top_indices]

                st.markdown("### 🧠 Top Skin Type Predictions:")
                for label, prob in top_labels:
                    st.markdown(f"- **{label}**: {prob * 100:.2f}%")

# Show cropped face with detections
if cropped_face is not None:
    st.image(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB), caption="🧐 Cropped Face Detection", use_column_width=True)

    # Legend
    st.markdown("### 🔹 Legend")
    patches = [
        Patch(color=np.array(label_colors[name][::-1]) / 255.0, label=name)
        for name in label_colors
    ]
    fig = plt.figure()
    plt.legend(handles=patches, loc='center', ncol=3)
    plt.axis('off')
    st.pyplot(fig)
