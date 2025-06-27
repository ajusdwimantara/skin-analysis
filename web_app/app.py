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
    return tf.keras.models.load_model('classify.h5')

loaded_model = load_model()
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
if ctx.video_processor and st.button("📸 Capture & Detect Face"):
    with lock:
        frame = ctx.video_processor.frame
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped_face = frame[y:y+h, x:x+w]

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

                processed_face = enhance_skin_image(cropped_face)

                # Resize + Normalize for object detection
                resized = cv2.resize(processed_face, (in_width, in_height))
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
                        cv2.rectangle(processed_face, (xmin, ymin), (xmax, ymax), color, 2)

                # --- Classification ---
                resized_face = cv2.resize(processed_face, (224, 224))
                normalized_face = resized_face.astype(np.float32) / 255.0
                input_face = np.expand_dims(normalized_face, axis=0)
                y_test = loaded_model.predict(input_face)

                # Get top 3 predictions
                top_indices = y_test[0].argsort()[-3:][::-1]

                # Filter only predictions with prob > 30%
                top_labels = [
                    (classify_labels[i], y_test[0][i])
                    for i in top_indices if y_test[0][i] > 0.3
                ]

                # Display
                if top_labels:
                    st.markdown("### 🧠 Top Skin Type Predictions:")
                    for label, prob in top_labels:
                        st.markdown(f"- **{label}**: {prob * 100:.2f}%")
                else:
                    st.markdown("⚠️ No confident prediction (≥ 30%) was found.")

# Show cropped face with detections
if cropped_face is not None:
    st.image(cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB), caption="🧐 Cropped Face Detection", use_column_width=True)

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
