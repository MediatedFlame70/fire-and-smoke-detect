from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Fire and Smoke Detection", page_icon="🔥", layout="wide")

st.title("Fire and Smoke Detection (YOLOv8)")
st.caption("Upload an image or video to run fire/smoke detection in near real-time.")

weights_path = Path("output/yolov8/best_model/best.pt")
if not weights_path.exists():
    st.error("Model weights not found. Train first with: python train_yolov8.py")
    st.stop()

model = YOLO(str(weights_path))

conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

if file is not None:
    suffix = Path(file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = Path(tmp.name)

    if suffix in {".jpg", ".jpeg", ".png"}:
        result = model.predict(source=str(tmp_path), conf=conf, imgsz=640, save=False, verbose=False)[0]
        plotted = result.plot()
        st.image(plotted, caption="Detection Output", channels="BGR", use_container_width=True)
    else:
        with st.spinner("Running video inference..."):
            pred = model.predict(
                source=str(tmp_path),
                conf=conf,
                imgsz=640,
                save=True,
                project="output/yolov8/predictions",
                name="streamlit_video",
                exist_ok=True,
                verbose=False,
            )
            out_file = Path(pred[0].save_dir) / tmp_path.name
            if out_file.exists():
                st.video(str(out_file))
            else:
                st.warning("Video was processed but output file was not found.")

    tmp_path.unlink(missing_ok=True)
