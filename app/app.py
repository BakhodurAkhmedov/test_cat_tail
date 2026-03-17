"""Gradio web app for cat tail segmentation demo."""

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights/best.pt"

parser = argparse.ArgumentParser(description="Cat tail segmentation web app")
parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Model weights path")
parser.add_argument("--port", type=int, default=7860, help="Server port")
args = parser.parse_args()

model = YOLO(args.weights)


def segment_image(image: np.ndarray, confidence: float = 0.3) -> np.ndarray:
    """Run segmentation on a single image."""
    results = model.predict(image, conf=confidence, verbose=False)
    result_img = results[0].plot()

    if len(results[0].boxes) == 0:
        cv2.putText(result_img, "No tail detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return result_img


def segment_video(video_path: str, confidence: float = 0.3) -> str:
    """Run segmentation on video, return path to annotated video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

    cap.release()
    writer.release()
    return out_path


with gr.Blocks(title="Cat Tail Segmentation") as demo:
    gr.Markdown("# Cat Tail Segmentation")
    gr.Markdown("Upload a photo or video of a cat and the model will segment its tail.")

    confidence = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence threshold")

    with gr.Tabs():
        with gr.Tab("Image"):
            img_input = gr.Image(label="Upload cat photo")
            img_output = gr.Image(label="Result")
            img_btn = gr.Button("Segment")
            img_btn.click(segment_image, inputs=[img_input, confidence], outputs=img_output)

        with gr.Tab("Video"):
            vid_input = gr.Video(label="Upload video")
            vid_output = gr.Video(label="Result")
            vid_btn = gr.Button("Segment")
            vid_btn.click(segment_video, inputs=[vid_input, confidence], outputs=vid_output)

if __name__ == "__main__":
    demo.launch(server_port=args.port)
