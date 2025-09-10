import streamlit as st
import numpy as np
from rasterio.mask import mask
import rioxarray
import json
import matplotlib.pyplot as plt
import cv2
import supervision as sv

# image tiling
from supervision import InferenceSlicer, OverlapFilter, BoxAnnotator
from ultralytics import YOLO

# load model
yolo_model = YOLO("best.pt")

# streamlit
st.title("Mask TIF with GeoJSON")

uploaded_tif = st.file_uploader("Upload TIF file", type=["tif", "tiff"])
uploaded_geojson = st.file_uploader("Upload GeoJSON file", type=["geojson"])


# --- Processing ---
if uploaded_geojson and uploaded_tif:
    try:
        # Load GeoJSON
        if isinstance(uploaded_geojson, str):  # from Drive
            with open(uploaded_geojson) as f:
                data = json.load(f)
        else:  # from uploader
            data = json.load(uploaded_geojson)

        crs = data["crs"]["properties"]["name"]
        geoms = [feat["geometry"] for feat in data["features"]]

        # Load raster
        rds = rioxarray.open_rasterio(uploaded_tif)

        # Clip raster
        clipped = rds.rio.clip(geoms, crs, drop=False)

        # Convert to numpy (Y, X, Bands)
        img = clipped.transpose("y", "x", "band").values

        # Normalize to 0–255 if needed
        if img.dtype != "uint8":
            img_min, img_max = img.min(), img.max()
            img = ((img - img_min) / (img_max - img_min) * 255).astype("uint8")

        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Clipped Orthophoto (RGB)")

        # Save figure with DPI (resolution control)
        plt.savefig("clipped_result_with_coords8Ha.jpg", dpi=1000, bbox_inches="tight")
        plt.close()

        image = cv2.imread("clipped_result_with_coords8Ha.jpg")

        # YOLO Prediction 
        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = yolo_model(image_slice, verbose=False, conf=0.4, iou=0.5, agnostic_nms=True)[0]
            return sv.Detections.from_ultralytics(result)

        # for image tiling, handling overlap, slicing, and also stitch the image back to the original size
        slicer = InferenceSlicer(
            slice_wh=(640, 640),
            overlap_ratio_wh=None,
            overlap_wh = (int(640 * 0.3), int(640 * 0.3)),  # = (192, 192)
            callback=callback,
            overlap_filter=OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.1
        )

        # Run slicing inference
        detections = slicer(image)
        # Filter to only class ID 0 (palm trees)
        detections = detections[detections.class_id == 0]

        # Draw bounding boxes using BoxAnnotator
        box_annotator = BoxAnnotator(color=sv.Color.GREEN, thickness=5)
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.subheader("Clipped Orthophoto with YOLO Prediction")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(annotated_image)
        ax.set_title(f"{len(detections)} Palm tree Detected")
        ax.axis("off")
        st.pyplot(fig)

        # accuracy for detected palm tree
        total = len(detections)
        result = total / 842 * 100
        st.title(f'Accuracy = {result:.2f}%')

    except Exception as e:
        st.error(f" Error: {e}")

else:
    st.info("Please upload or link both a **GeoJSON** and a **TIFF raster**.")


