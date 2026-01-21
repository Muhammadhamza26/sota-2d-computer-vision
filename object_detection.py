"""
object_detection.py

A demonstration of 2D object detection using YOLO26 from the Ultralytics package.
This script downloads a sample image, loads a pre-trained YOLO26 model, runs inference, and visualizes the results.
YOLO26 is the latest state-of-the-art YOLO model (released in 2025) featuring an end-to-end, NMS-free architecture that improves accuracy and speed.
"""

import os
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    """Run object detection on a sample image using YOLO26"""
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download sample image if it doesn't exist
    import urllib.request
    sample_url = "https://ultralytics.com/images/bus.jpg"
    sample_image_path = output_dir / "sample.jpg"
    if not sample_image_path.exists():
        urllib.request.urlretrieve(sample_url, sample_image_path)
        print(f"Downloaded sample image to {sample_image_path}")

    # Load pre-trained YOLO26 model (nano version for speed)
    model = YOLO("yolo26n.pt")

    # Run inference and save results
    results = model.predict(source=str(sample_image_path), save=True, project=str(output_dir), name="predict")
    print("Detection completed. Output saved in:", output_dir)


if __name__ == "__main__":
    main()
