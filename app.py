import sys
import os
from gtts import gTTS
import base64
import io
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from PIL import Image

from src.violation_clasess import VIOLATION_CLASSES

def play_voice_alert(text="Violation detected"):
    tts = gTTS(text=text, lang='en')

    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)

    b64 = base64.b64encode(audio_bytes.getvalue()).decode()

    audio_html = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)
def generate_heatmap(image, points):
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    # Add intensity at each violation point
    for (x, y) in points:
        cv2.circle(heatmap, (x, y), 40, 1, -1)

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply color
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay with original image
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))




# -----------------------------
# Paths
# -----------------------------
#MODEL_PATH = r"D:\Final year project traffic violation detection\weights\best (1).pt"
#OUTPUT_DIR = r"D:\Final year project traffic violation detection\outputs\results"
MODEL_PATH = "weights/best.pt"

#os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚦AI-Powered Traffic Violation Detection System")

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
) # file upload option on web interface


# Detect Button

detect_btn = st.button("🔍 Detect Violation")


if uploaded_file is not None:

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, use_container_width=True)
    st.markdown(
        "<h3 style='text-align: center;'>Uploaded Image</h3>",
        unsafe_allow_html=True
    )

    if detect_btn:

        # Run YOLO detection
        results = model(img, conf=0.4)

        violation_found = False
        violation_details = []
        vehicle_count=0
        violation_count=0
        violation_points=[]
        
        

        # Process detections
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = VIOLATION_CLASSES.get(cls_id, "Unknown")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    #count vehilces
                    if "Motorcycle" in label or "Bicycle" in label:
                        vehicle_count+=1
                     # Store violation info
                    if label != "Unknown":
                        violation_count+=1
                        violation_found = True
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        violation_points.append((cx, cy))
                        violation_details.append({
                            "type": label,
                            "confidence": conf
                        })

                    # Red for violation, Green otherwise
                    color = (0, 0, 255) if label != "Unknown" else (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        img,
                        f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        # Show Result Image
        st.image(img, use_container_width=True)
        st.markdown(
            "<h3 style='text-align: center;'>Detection Result</h3>",
            unsafe_allow_html=True
        )
        # -----------------------------
        # Display Violation Status
        # -----------------------------
        st.subheader("Detection Result")

        if violation_found:
            st.error("🚨 Violation Detected")
            play_voice_alert("Violation detected in the above traffic image")  # 🔊 SOUND ALERT ADDED HERE

            for v in violation_details:
                st.write(f"Violation detected: **{v['type']}**")
                st.write(f"Confidence score of violation: **{v['confidence']:.2f}**")
                st.write("---")
            #show total violations
            st.warning(f"🚨 Total Violations Detected: {violation_count} ")
            st.info(f"🚗 Total Vehicles Detected: {vehicle_count}")

        else:
            st.success("✅ No Traffic Violation Detected")
            # 🔊 Voice alert for NO violation
            play_voice_alert("No violation is detected in the given traffic image")
         # 🔥 Show Heatmap
        if violation_points:
            heatmap_img = generate_heatmap(img.copy(), violation_points)
            st.image(heatmap_img, use_container_width=True)
            st.markdown(
                "<h3 style='text-align: center;'>Violation Heatmap</h3>",
                unsafe_allow_html=True
            )
       
elif detect_btn:
    st.warning("⚠️ Please upload an image before detection.")


