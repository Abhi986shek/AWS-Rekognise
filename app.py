import streamlit as st
import cv2
import numpy as np
import boto3
from threading import Thread, Lock
from queue import Queue
import time
import logging
from datetime import datetime
import os
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 bucket configuration
S3_BUCKET_NAME = "videostreamlit"
VIDEO_FILENAME = "recorded_session.avi"
VIDEO_FPS = 15
VIDEO_CODEC = 'XVID'
AVATAR_PATH = "avatar.gif.webp"

# Session persistence constants
SESSION_FILE = "current_session.txt"
MAX_SESSION_DURATION = 1800  # 30 minutes

class FrameProcessor:
    def __init__(self):
        self.lock = Lock()
        self.detection_results = []
        self.running = False
        self.frame_queue = Queue(maxsize=1)
        self.detection_thread = None
        self.last_processed_time = 0
        self.processing_interval = 1.0

        try:
            self.rekognition = boto3.client(
                'rekognition',
                region_name="ap-south-1",
                aws_access_key_id="AKIA3C6FL24VFVSUZ2FM",
                aws_secret_access_key="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"
            )
            logger.info("AWS Rekognition initialized")
        except Exception as e:
            logger.error(f"Rekognition error: {e}")
            self.rekognition = None

    def start(self):
        if not self.rekognition:
            return
        self.running = True
        self.detection_thread = Thread(target=self._process_frames, daemon=True)
        self.detection_thread.start()

    def stop(self):
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()

    def update_frame(self, frame):
        now = time.time()
        if (now - self.last_processed_time) >= self.processing_interval:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (320, 240))
                _, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(buf.tobytes())
                self.last_processed_time = now
            except Exception as e:
                logger.warning(f"Frame queue error: {e}")

    def _process_frames(self):
        while self.running:
            try:
                frame_bytes = self.frame_queue.get(timeout=1)
                response = self.rekognition.detect_faces(Image={'Bytes': frame_bytes}, Attributes=['ALL'])
                with self.lock:
                    self.detection_results = response.get('FaceDetails', [])
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except Exception as e:
                if "timeout" not in str(e).lower():
                    logger.warning(f"Detection error: {e}")

    def get_results(self):
        with self.lock:
            return self.detection_results.copy()

def draw_face_info(frame, face, idx):
    h, w = frame.shape[:2]
    box = face['BoundingBox']
    left = int(box['Left'] * w)
    top = int(box['Top'] * h)
    right = left + int(box['Width'] * w)
    bottom = top + int(box['Height'] * h)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    gender = face.get('Gender', {}).get('Value', 'Unknown')
    emotions = sorted(face.get('Emotions', []), key=lambda e: e['Confidence'], reverse=True)
    top_emotion = emotions[0] if emotions else {'Type': 'Unknown', 'Confidence': 0}
    pose = face.get('Pose', {})
    looking = "Yes" if abs(pose.get('Pitch', 20)) < 15 and abs(pose.get('Yaw', 20)) < 15 else "No"

    label = f"Person {idx+1}: {gender} | {top_emotion['Type']} ({top_emotion['Confidence']:.0f}%)"
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    return {
        "id": idx + 1,
        "gender": gender,
        "looking": looking,
        "emotion": top_emotion['Type'],
        "confidence": top_emotion['Confidence'],
        "bbox": (left, top, right, bottom)
    }

def main():
    st.set_page_config(page_title="Face Match & Emotion", layout="wide")
    st.title("🎯 Real-time Emotion Detection + Face Match")

    st.markdown("""
    <style>
        .st-emotion-camera {
            width: 100% !important;
            height: 480px !important;
            object-fit: cover !important;
        }
    </style>
    """, unsafe_allow_html=True)

    for key in ['processor', 'run', 'detected_faces', 'uploaded_image', 'match_results', 'last_frame', 'cap', 'do_match', 'writer', 'video_path', 'persistent_end_time']:
        if key not in st.session_state:
            if key in ['processor', 'cap', 'uploaded_image', 'writer', 'video_path']:
                st.session_state[key] = None
            elif key in ['run', 'do_match']:
                st.session_state[key] = False
            elif key == 'persistent_end_time':
                st.session_state[key] = 0.0
            else:
                st.session_state[key] = {}

    # Restore persistent session if available
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            line = f.read().strip()
            if line:
                prev_path, prev_end = line.split(",")
                if float(prev_end) > time.time():
                    st.session_state.video_path = prev_path
                    st.session_state.persistent_end_time = float(prev_end)

    if not st.session_state.video_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.video_path = f"{timestamp}_{VIDEO_FILENAME}"
        st.session_state.persistent_end_time = time.time() + MAX_SESSION_DURATION
        with open(SESSION_FILE, "w") as f:
            f.write(f"{st.session_state.video_path},{st.session_state.persistent_end_time}")

    if st.session_state.processor is None:
        st.session_state.processor = FrameProcessor()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Live Camera")
        frame_placeholder = st.empty()
        status_text = st.empty()
        if st.button("▶️ Start Camera", disabled=st.session_state.run):
            st.session_state.run = True
            st.session_state.processor.start()

            with st.spinner("🎥 Opening camera..."):
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

                start_time = time.time()
                while not cap.isOpened():
                    if time.time() - start_time > 5:
                        st.error("❌ Failed to open camera.")
                        st.session_state.run = False
                        break
                    time.sleep(0.1)

                st.session_state.cap = cap

                fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                st.session_state.writer = cv2.VideoWriter(st.session_state.video_path, fourcc, VIDEO_FPS, (640, 480))

        if st.button("⏹ Stop Camera", disabled=not st.session_state.run):
            st.session_state.run = False
            st.session_state.processor.stop()

            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None

            if st.session_state.writer:
                st.session_state.writer.release()
                st.session_state.writer = None

                try:
                    s3 = boto3.client(
                        's3',
                        region_name="ap-south-1",
                        aws_access_key_id="AKIA3C6FL24VFVSUZ2FM",
                        aws_secret_access_key="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"
                    )
                    s3.upload_file(st.session_state.video_path, S3_BUCKET_NAME, os.path.basename(st.session_state.video_path))
                    st.success(f" Video uploaded to S3: {S3_BUCKET_NAME}/{os.path.basename(st.session_state.video_path)}")
                    os.remove(st.session_state.video_path)
                except Exception as e:
                    st.error(f" Failed to upload video: {e}")

            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)

    with col2:
        st.subheader("🤖 AI Avatar")
        avatar_placeholder = st.empty()

        if os.path.exists(AVATAR_PATH):
            with open(AVATAR_PATH, "rb") as f:
                avatar_bytes = f.read()
                encoded = base64.b64encode(avatar_bytes).decode()
                avatar_html = f"""
                    <div style="width:100%; height:480px; display:flex; justify-content:center; align-items:center; overflow:hidden;">
                        <img src='data:image/webp;base64,{encoded}' style='width:100%; height:100%; object-fit:contain;' />
                    </div>
                """
                avatar_placeholder.markdown(avatar_html, unsafe_allow_html=True)
        else:
            avatar_placeholder.warning("⚠️ 'avatar.gif.webp' not found!")

        st.subheader("🖼️ Face Matching")
        uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_np = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, uploaded_buf = cv2.imencode('.jpg', rgb)
            st.session_state.uploaded_image = uploaded_buf.tobytes()
            st.image(rgb, use_container_width=True)

        match_result = st.empty()
        if st.button("🔍 Find Match"):
            if not st.session_state.run:
                match_result.warning("Start the camera first.")
            elif not st.session_state.uploaded_image:
                match_result.warning("Upload an image.")
            else:
                st.session_state.do_match = True

    with st.sidebar:
        st.subheader("📋 Face Info")
        sidebar = st.empty()

    if st.session_state.run and st.session_state.cap and st.session_state.cap.isOpened():
        cap = st.session_state.cap

        try:
            while st.session_state.run and cap.isOpened() and time.time() < st.session_state.persistent_end_time:
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.05)

                if not ret:
                    status_text.error("⚠️ Could not read from camera.")
                    st.session_state.run = False
                    break

                st.session_state.last_frame = frame.copy()
                st.session_state.processor.update_frame(frame)
                faces = st.session_state.processor.get_results()
                disp = frame.copy()
                face_info_list = []

                for idx, face in enumerate(faces):
                    info = draw_face_info(disp, face, idx)
                    face_info_list.append(info)

                st.session_state.detected_faces = face_info_list
                frame_placeholder.image(disp, channels="BGR", use_container_width=True)

                if st.session_state.writer:
                    st.session_state.writer.write(frame)

                with sidebar.container():
                    if face_info_list:
                        st.success(f"Detected {len(face_info_list)} face(s)")
                        for face in face_info_list:
                            with st.expander(f"👤 Person {face['id']}"):
                                st.metric("Gender", face['gender'])
                                st.metric("Looking", face['looking'])
                                st.metric("Emotion", f"{face['emotion']} ({face['confidence']:.1f}%)")
                                if face['id'] in st.session_state.match_results:
                                    st.success(f"Match: {st.session_state.match_results[face['id']]:.1f}%")
                    else:
                        st.warning("No faces detected.")

                if st.session_state.do_match:
                    best_match = None
                    best_similarity = 0
                    h, w = st.session_state.last_frame.shape[:2]

                    for face in face_info_list:
                        l, t, r, b = face['bbox']
                        padding = 0.05
                        pad_x = int((r - l) * padding)
                        pad_y = int((b - t) * padding)
                        l = max(0, l - pad_x)
                        r = min(w, r + pad_x)
                        t = max(0, t - pad_y)
                        b = min(h, b + pad_y)
                        cropped = st.session_state.last_frame[t:b, l:r]

                        if cropped.shape[0] < 100 or cropped.shape[1] < 100:
                            continue

                        try:
                            rgb_face = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                            _, buf = cv2.imencode('.jpg', rgb_face)
                            resp = st.session_state.processor.rekognition.compare_faces(
                                SourceImage={'Bytes': st.session_state.uploaded_image},
                                TargetImage={'Bytes': buf.tobytes()},
                                SimilarityThreshold=80
                            )
                            matches = resp.get("FaceMatches", [])
                            if matches:
                                sim = matches[0]['Similarity']
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_match = face['id']
                        except Exception as e:
                            logger.warning(f"Matching error: {e}")

                    if best_match:
                        st.session_state.match_results[best_match] = best_similarity
                        match_result.success(f"✅ Best match: Person {best_match} ({best_similarity:.1f}%)")
                    else:
                        match_result.error("❌ No match found.")

                    st.session_state.do_match = False

                time_left = max(0, int(st.session_state.persistent_end_time - time.time()))
                status_text.info(f"📸 Running | {len(face_info_list)} face(s) detected | ⏳ {time_left // 60:02}:{time_left % 60:02} left")
                time.sleep(0.05)
        finally:
            st.session_state.processor.stop()
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.writer:
                st.session_state.writer.release()
                st.session_state.writer = None
                try:
                    s3 = boto3.client(
                        's3',
                        region_name="ap-south-1",
                        aws_access_key_id="AKIA3C6FL24VFVSUZ2FM",
                        aws_secret_access_key="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"
                    )
                    s3.upload_file(st.session_state.video_path, S3_BUCKET_NAME, os.path.basename(st.session_state.video_path))
                    st.success(f"✅ Video uploaded to S3: {S3_BUCKET_NAME}/{os.path.basename(st.session_state.video_path)}")
                    os.remove(st.session_state.video_path)
                except Exception as e:
                    st.error(f"❌ Failed to upload video: {e}")
            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)
            st.session_state.run = False

if __name__ == "__main__":
    main()
