import streamlit as st
from PIL import Image
import tempfile
import os

# ---------------------------
# BACKEND IMPORTS
# ---------------------------
from style_transfer import (
    apply_style_transfer,
    apply_fast_nst,
    stylize_video,
    batch_process_folder
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Style Transfer Studio",
    layout="wide",
)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🎨 Controls")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Image", "Video", "Batch Processing"]
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Fast NST (Real-time)", "Slow NST (High Quality)"]
)

style_strength = st.sidebar.slider(
    "Style Strength (Slow NST only)",
    0.1, 1.5, 1.0, step=0.1
)

st.sidebar.info("⚡ Fast NST ignores style strength (real-time model)")

# ---------------------------
# MAIN UI
# ---------------------------
st.title("✨ AI Style Transfer Studio")
st.markdown("Upload content and style image to generate artistic output")

col1, col2 = st.columns(2)

# ---------------------------
# CONTENT UPLOAD
# ---------------------------
with col1:
    if mode == "Image":
        content_file = st.file_uploader(
            "Upload Content Image",
            type=["jpg", "jpeg", "png"]
        )
    elif mode == "Video":
        content_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov"]
        )
    else:
        content_file = st.file_uploader(
            "Upload ZIP of Images",
            type=["zip"]
        )

# ---------------------------
# STYLE UPLOAD (ONLY ONE – FAST & STABLE)
# ---------------------------
with col2:
    style_file = st.file_uploader(
        "Upload Style Image",
        type=["jpg", "jpeg", "png"]
    )

# ---------------------------
# GENERATE
# ---------------------------
st.markdown("### 🚀 Generate Output")
generate_btn = st.button("Start Processing")

# =====================================================
# PROCESSING
# =====================================================
if generate_btn:

    if not content_file or not style_file:
        st.error("Please upload both content and style files.")
        st.stop()

    style_img = Image.open(style_file).convert("RGB")

    # =================================================
    # IMAGE MODE
    # =================================================
    if mode == "Image":

        content_img = Image.open(content_file).convert("RGB")
        st.info("Processing image...")

        if model_type == "Fast NST (Real-time)":
            output = apply_fast_nst(
                content_img,
                style_img
            )
        else:
            output = apply_style_transfer(
                content_img,
                [style_img],
                strength=style_strength
            )

        before, after = st.columns(2)
        with before:
            st.image(content_img, caption="Original Image")

        with after:
            st.image(output, caption="Stylized Output")

        # DOWNLOAD
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        output.save(tmp.name)

        with open(tmp.name, "rb") as f:
            st.download_button(
                "⬇️ Download Image",
                f,
                "styled_output.png",
                "image/png"
            )

    # =================================================
    # VIDEO MODE (FAST ONLY)
    # =================================================
    elif mode == "Video":

        st.info("Processing video (this may take some time)...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(content_file.read())
            video_path = temp_video.name

        output_path = stylize_video(
            input_video_path=video_path,
            style_image=style_img,
            output_path="styled_video.mp4"
        )

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Video",
                f,
                "styled_video.mp4",
                "video/mp4"
            )

    # =================================================
    # BATCH MODE
    # =================================================
    elif mode == "Batch Processing":

        st.info("Processing batch images...")

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "input.zip")

        with open(zip_path, "wb") as f:
            f.write(content_file.read())

        out_dir = batch_process_folder(
            input_zip_or_folder=zip_path,
            style_image_or_pil=style_img,
            model="fast" if model_type == "Fast NST (Real-time)" else "slow",
            style_strength=style_strength
        )

        st.success("Batch processing completed!")
        st.write(f"Output saved in folder: `{out_dir}`")
