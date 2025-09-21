---

# Real-time Emotion Detection & Face Matching with AWS Rekognition

A **Streamlit-based application** for real-time emotion detection, facial recognition, and face matching using **AWS Rekognition**. The app integrates live camera streaming, AWS Rekognition analysis, video recording with session persistence, and S3 upload functionality.

---

## Features

* **Real-time Face Detection**
  Detects multiple faces from a live camera feed and draws bounding boxes.

* **Emotion Analysis**
  Displays top emotion and confidence for each detected face.

* **Face Attribute Analysis**
  Shows gender and whether the person is looking at the camera.

* **Face Matching**
  Upload an image to compare against faces detected in the live stream using AWS Rekognition’s `CompareFaces` API.

* **Video Recording with Persistence**
  Records live camera sessions and persists video for up to 30 minutes, even if the app is refreshed.

* **S3 Upload**
  Automatically uploads recorded videos to a configured AWS S3 bucket.

* **AI Avatar Display**
  Shows a static AI avatar GIF in the interface alongside the camera feed.

---

## Demo

The app provides a two-column interface:

1. **Live Camera Feed** – Shows the camera stream, detected faces, and overlays bounding boxes with face info.
2. **AI Avatar & Face Matching** – Displays an avatar and allows uploading an image to find face matches.

The **sidebar** shows detailed information for each detected face, including gender, emotion, confidence, and camera-facing status.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/repo-name.git
cd repo-name
```

2. **Install dependencies**

```bash
pip install streamlit opencv-python boto3 numpy
```

3. **AWS Configuration**

Update the `FrameProcessor` class with your **AWS credentials** (or configure via environment variables / AWS CLI):

```python
self.rekognition = boto3.client(
    'rekognition',
    region_name="ap-south-1",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)
```

Set your **S3 bucket name** in `S3_BUCKET_NAME`.

4. **Run the app**

```bash
streamlit run app.py
```

---

## Usage

1. Start the camera using the **▶️ Start Camera** button.
2. Observe live face detection and emotion analysis on the left column.
3. Upload a reference face image for matching on the right column.
4. Press **🔍 Find Match** to detect the best match among detected faces.
5. Stop the camera and upload the video to S3 using **⏹ Stop Camera**.

---

## File Structure

* `app.py` – Main Streamlit application.
* `avatar.gif.webp` – AI avatar image for the interface.
* `current_session.txt` – Tracks video session persistence.

---

## Configuration

* **Video Settings**

```python
VIDEO_FILENAME = "recorded_session.avi"
VIDEO_FPS = 15
VIDEO_CODEC = 'XVID'
MAX_SESSION_DURATION = 1800  # 30 minutes
```

* **Face Detection & Emotion**

Uses AWS Rekognition `detect_faces` API with `Attributes=['ALL']`.

* **Face Matching**

Uses AWS Rekognition `compare_faces` API with a similarity threshold of 80%.

---

## Dependencies

* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Boto3](https://boto3.amazonaws.com/)

---

## Notes

* Ensure your webcam is accessible and not used by other applications.
* AWS credentials must have **Rekognition** and **S3 upload** permissions.
* Uploaded videos are removed locally after successful S3 upload.

---

