import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import threading

# Global lock to protect shared resources
lock = threading.Lock()
captured_frame = None  # Will store the last captured frame


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Store latest frame in global variable (thread-safe)
        global captured_frame
        with lock:
            self.frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("📸 Real-time Frame Capture from Webcam")

# Start video stream
ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Button to capture the frame
if ctx.video_processor:
    if st.button("📷 Capture Frame"):
        with lock:
            captured_frame = ctx.video_processor.frame.copy() if ctx.video_processor.frame is not None else None

# Display the captured frame
if captured_frame is not None:
    st.image(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB), caption="Captured Frame", use_column_width=True)
