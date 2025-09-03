import streamlit as st
import rioxarray
import json
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import requests
import tempfile
import os

st.set_page_config(page_title="Raster Clipper", layout="wide")

st.title("🗺️ Raster Clipper App")
st.write("Upload or link a **GeoJSON** and a **TIFF raster** to clip the raster and export as JPG.")

# --- Upload Options ---
st.sidebar.header("Upload Options")
upload_method = st.sidebar.radio("Choose file source:", ["Local Upload", "Google Drive Link"])

geojson_file, tif_file = None, None

if upload_method == "Local Upload":
    col1, col2 = st.columns(2)
    with col1:
        geojson_file = st.file_uploader("📂 Upload GeoJSON", type=["geojson"])
    with col2:
        tif_file = st.file_uploader("🌍 Upload TIFF Raster", type=["tif", "tiff"])

elif upload_method == "Google Drive Link":
    geojson_url = st.text_input("🔗 Paste GeoJSON Google Drive link")
    tif_url = st.text_input("🔗 Paste TIFF Google Drive link")

    def download_from_drive(url, suffix):
        if "drive.google.com" not in url:
            st.error("❌ Not a valid Google Drive link")
            return None
        try:
            file_id = url.split("/d/")[1].split("/")[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.get(direct_url, stream=True)
            if response.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(response.content)
                tmp.close()
                return tmp.name
            else:
                st.error("⚠️ Failed to download file.")
                return None
        except Exception as e:
            st.error(f"⚠️ Error parsing link: {e}")
            return None

    if geojson_url:
        geojson_file = download_from_drive(geojson_url, ".geojson")
    if tif_url:
        tif_file = download_from_drive(tif_url, ".tif")

# --- Processing ---
if geojson_file and tif_file:
    try:
        # Load GeoJSON
        if isinstance(geojson_file, str):  # from Drive
            with open(geojson_file) as f:
                data = json.load(f)
        else:  # from uploader
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
        ax.set_title("Clipped Orthophoto (RGB)")
        st.pyplot(fig)

        # --- Save as JPG ---
        buf = BytesIO()
        save_fig, save_ax = plt.subplots(figsize=(12, 12))
        save_ax.imshow(img)
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
    st.info("👆 Please upload or link both a **GeoJSON** and a **TIFF raster**.")
