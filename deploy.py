import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="YOLO Parking Spot Detection",
    layout="wide"
)

st.title("🟢 YOLO Parking Spot Detection on Video")

# ----------------------------
# Video path
# ----------------------------
video_path = "29_05_20201.mp4"

if not os.path.exists(video_path):
    st.error(f"Video file {video_path} not found in working directory.")
else:
    # Show original video
    st.subheader("Original Video")
    st.video(video_path)

    # ----------------------------
    # Load YOLO model
    # ----------------------------
    model = YOLO("best.pt")  # your trained YOLO weights

    if st.button("Run Parking Spot Detection"):
        st.info("Processing video... this may take some time depending on length.")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Prepare temporary output video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # Streamlit frame for live display
        stframe = st.empty()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO prediction
            results = model.predict(frame, verbose=False)[0]

            # Draw boxes
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # If class is occupied, draw in blue
                color = (255, 0, 0) if label.lower() == "occupied" else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Write frame to output
            out.write(frame)

            # Show frame in Streamlit
            stframe.image(frame, channels="BGR")

            progress_bar.progress((i + 1) / frame_count)

        cap.release()
        out.release()
        st.success("Video processing completed!")

        # Show processed video
        st.subheader("Processed Video with Bounding Boxes")
        st.video(temp_file.name)

        # Download button
        st.download_button(
            label="Download Processed Video",
            data=open(temp_file.name, "rb"),
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
