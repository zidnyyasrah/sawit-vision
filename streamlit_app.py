import streamlit as st
import rioxarray
import json
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Raster Clipper", layout="wide")

# Title
st.title("🗺️ Raster Clipper App")
st.write("Upload a **GeoJSON file** (boundary) and a **TIFF raster file** (orthophoto).")
st.write("The app will clip the raster using the boundary and let you download the result as a JPG.")

# File uploaders in two columns
col1, col2 = st.columns(2)
with col1:
    geojson_file = st.file_uploader("📂 Upload GeoJSON", type=["geojson"])
with col2:
    tif_file = st.file_uploader("🌍 Upload TIFF Raster", type=["tif", "tiff"])

# Process once both files are uploaded
if geojson_file and tif_file:
    try:
        # Load GeoJSON
        data = json.load(geojson_file)
        crs = data["crs"]["properties"]["name"]
        geoms = [feat["geometry"] for feat in data["features"]]

        # Load raster
        rds = rioxarray.open_rasterio(tif_file)

        # Clip raster
        clipped = rds.rio.clip(geoms, crs, drop=False)

        # Convert to numpy (Y, X, Bands)
        img = clipped.transpose("y", "x", "band").values

        # Normalize to 0–255 if needed
        if img.dtype != "uint8":
            img_min, img_max = img.min(), img.max()
            img = ((img - img_min) / (img_max - img_min) * 255).astype("uint8")

        # --- Display clipped image ---
        st.subheader("📸 Clipped Orthophoto")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Clipped Orthophoto (RGB)")
        st.pyplot(fig)

        # --- Save as JPG for download ---
        buf = BytesIO()
        save_fig, save_ax = plt.subplots(figsize=(12, 12))
        save_ax.imshow(img)
        save_ax.set_xlabel("Longitude")
        save_ax.set_ylabel("Latitude")
        save_ax.set_title("Clipped Orthophoto (RGB)")
        save_fig.savefig(buf, format="jpg", dpi=600, bbox_inches="tight")
        buf.seek(0)
        plt.close(save_fig)

        st.download_button(
            label="📥 Download Clipped JPG",
            data=buf,
            file_name="clipped_result.jpg",
            mime="image/jpeg"
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Please upload both a **GeoJSON** and a **TIFF raster** to continue.")
