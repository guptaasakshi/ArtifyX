"""
style_transfer.py

Backend helpers for Neural Style Transfer project.
Provides:
- apply_style_transfer(...)             # slow VGG19 based NST (high quality)
- apply_fast_nst(...)                   # TF-Hub arbitrary fast stylization
- blend_styles(...) / prepare_blended_style(...)   # multi-style utilities
- save_style_filter(...) / load_style_filter(...)
- stylize_video(...)                    # video style transfer using TF-Hub fast model
- batch_process_folder(...)             # batch processing for a folder/zip of images
- upscale_image(...)                    # optional TF-Hub ESRGAN upscale wrapper

Usage:
    from style_transfer import apply_style_transfer, apply_fast_nst, stylize_video, ...
"""

import os
import shutil
import tempfile
from typing import List, Union, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import imageio
from tqdm import tqdm

# ----------------------------
# CONFIG / LAYERS
# ----------------------------
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
CONTENT_LAYER = 'block5_conv2'


# ----------------------------
# UTIL: PIL / TENSOR converters
# ----------------------------
def pil_to_tensor(img_pil: Image.Image, max_dim: int = None) -> tf.Tensor:
    """PIL -> batched float32 tensor in [0,1] shape (1,H,W,3)."""
    if max_dim is not None:
        long_dim = max(img_pil.size)
        if long_dim > max_dim:
            scale = max_dim / long_dim
            new_w = int(img_pil.size[0] * scale)
            new_h = int(img_pil.size[1] * scale)

            # FIXED here
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = np.array(img_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return tf.constant(arr, dtype=tf.float32)


def tensor_to_pil(tensor: Union[tf.Tensor, np.ndarray]) -> Image.Image:
    """Batched or unbatched tensor/np array to PIL.Image (RGB)."""
    if isinstance(tensor, tf.Tensor):
        arr = tensor.numpy()
    else:
        arr = np.array(tensor)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype('uint8')
    return Image.fromarray(arr)


def load_image(path_or_file, max_dim: int = 512) -> Image.Image:
    """
    Load from path or file-like (streamlit upload).
    Returns PIL image resized keeping aspect ratio (if max_dim provided).
    """
    if isinstance(path_or_file, (str, os.PathLike)):
        img = Image.open(str(path_or_file)).convert('RGB')
    else:
        img = Image.open(path_or_file).convert('RGB')

    long_dim = max(img.size)
    if max_dim is not None and long_dim > max_dim:
        scale = max_dim / long_dim
        new_w = int(img.size[0] * scale)
        new_h = int(img.size[1] * scale)

        # FIXED here
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


# ----------------------------
# VGG19 feature extractor (slow NST)
# ----------------------------
def load_vgg19_extractor():
    """Return a model that outputs style layer activations + content activation."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    outputs.append(vgg.get_layer(CONTENT_LAYER).output)
    model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model


def preprocess_for_vgg(img_tensor: tf.Tensor) -> tf.Tensor:
    """Expect img in [0,1], float32 batched -> convert to VGG-preprocessed float32."""
    # VGG preprocess expects pixels in range 0..255 then subtract mean and BGR order
    x = img_tensor * 255.0
    return tf.keras.applications.vgg19.preprocess_input(x)


def deprocess_from_vgg(processed: tf.Tensor) -> Image.Image:
    """Reverse VGG preprocess output to PIL image (expects processed tensor)."""
    # processed is batched tensor similar to Keras preprocessing output
    arr = processed.numpy().squeeze()
    # reverse preprocess_input: from BGR with mean subtracted back to RGB [0,255]
    arr[..., 0] += 103.939
    arr[..., 1] += 116.779
    arr[..., 2] += 123.68
    arr = arr[..., ::-1]  # BGR -> RGB
    arr = np.clip(arr, 0, 255).astype('uint8')
    return Image.fromarray(arr)


def gram_matrix(feature_map: tf.Tensor) -> tf.Tensor:
    """Compute Gram matrix for style feature map (assumes shape (1,H,W,C))."""
    # shape: (1, H, W, C)
    x = tf.squeeze(feature_map, axis=0)
    c = tf.shape(x)[-1]
    x_flat = tf.reshape(x, [-1, c])  # (H*W, C)
    gram = tf.matmul(x_flat, x_flat, transpose_a=True)
    n = tf.cast(tf.shape(x_flat)[0], tf.float32)
    return gram / (n + 1e-8)


# ----------------------------
# Save / Load style filters
# ----------------------------
def save_style_filter(style_features: dict, filename: str = "style_filter.npz"):
    """
    Save style feature gram matrices (dict layer -> tensor) to .npz
    style_features: dict{layer_name: tf.Tensor}
    """
    save_dict = {}
    for k, v in style_features.items():
        if isinstance(v, tf.Tensor):
            save_dict[k] = v.numpy()
        else:
            save_dict[k] = np.array(v)
    np.savez(filename, **save_dict)
    return filename


def load_style_filter(filename: str = "style_filter.npz") -> dict:
    """
    Load saved style grams. Returns dict layer->tf.Tensor
    """
    data = np.load(filename)
    loaded = {k: tf.constant(data[k]) for k in data.files}
    return loaded


# ----------------------------
# Extract style features (grams) from an image path or PIL
# ----------------------------
def extract_style_features_from_image(style_image: Union[str, Image.Image],
                                      extractor_model=None,
                                      max_dim: int = 512) -> dict:
    """
    Return dict of layer->gram_matrix for the style image.
    """
    if extractor_model is None:
        extractor_model = load_vgg19_extractor()
    if isinstance(style_image, (str, os.PathLike)):
        pil = load_image(style_image, max_dim=max_dim)
    else:
        pil = style_image
    t = pil_to_tensor(pil, max_dim=max_dim)
    t_vgg = preprocess_for_vgg(t)
    outputs = extractor_model(t_vgg)
    style_outputs = outputs[:len(STYLE_LAYERS)]
    grams = {STYLE_LAYERS[i]: gram_matrix(style_outputs[i]) for i in range(len(STYLE_LAYERS))}
    return grams


# ----------------------------
# Multi-style blending utilities
# ----------------------------
def blend_grams(all_style_grams: List[List[tf.Tensor]], weights: List[float] = None) -> List[tf.Tensor]:
    """
    all_style_grams: list of styles; each style is list of grams per layer.
    returns: blended grams list (one per layer)
    """
    n_styles = len(all_style_grams)
    if n_styles == 0:
        raise ValueError("No style grams provided")
    if weights is None:
        weights = np.ones(n_styles, dtype=np.float32) / n_styles
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-12)
    n_layers = len(all_style_grams[0])
    blended = []
    for li in range(n_layers):
        accum = None
        for si in range(n_styles):
            g = all_style_grams[si][li]
            if accum is None:
                accum = weights[si] * g
            else:
                accum = accum + weights[si] * g
        blended.append(accum)
    return blended


def prepare_blended_style(style_image_paths: List[Union[str, Image.Image]],
                          extractor_model=None,
                          weights: List[float] = None,
                          max_dim: int = 512) -> List[tf.Tensor]:
    """
    Compute grams for each style image and return a blended gram list (one per style layer).
    """
    if extractor_model is None:
        extractor_model = load_vgg19_extractor()
    all_grams = []
    for p in style_image_paths:
        grams_dict = extract_style_features_from_image(p, extractor_model, max_dim=max_dim)
        # order grams per STYLE_LAYERS
        grams_list = [grams_dict[name] for name in STYLE_LAYERS]
        all_grams.append(grams_list)
    blended = blend_grams(all_grams, weights=weights)
    return blended


# ----------------------------
# Slow NST (VGG19 based) — runs optimization on the image tensor
# ----------------------------
def apply_style_transfer(content_pil: Image.Image,
                         style_pils: List[Image.Image],
                         strength: float = 1.0,
                         iterations: int = 100,
                         learning_rate: float = 0.02) -> Image.Image:
    """
    Slow VGG-based style transfer.
    content_pil: PIL image
    style_pils: list of PIL style images (only first is used here for basic NST)
    strength: scaling factor to number of optimization iterations
    iterations: base iterations (will be scaled by strength)
    Returns a PIL Image (RGB) of stylized result.
    """
    # Prepare images
    content = pil_to_tensor(content_pil, max_dim=512)  # [1,H,W,3] float32 [0,1]
    style = pil_to_tensor(style_pils[0], max_dim=512)

    extractor = load_vgg19_extractor()

    # Prepare targets
    content_vgg = preprocess_for_vgg(content)
    style_vgg = preprocess_for_vgg(style)

    outputs_style = extractor(style_vgg)
    style_targets = [gram_matrix(out) for out in outputs_style[:len(STYLE_LAYERS)]]

    outputs_content = extractor(content_vgg)
    content_target = outputs_content[-1]  # content activation

    # Initialize generated image variable (we optimize directly in pixel space)
    generated = tf.Variable(content * 255.0, dtype=tf.float32)  # VGG preprocessing expects 0..255 in preprocess_for_vgg

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # compute total steps from strength
    steps = max(1, int(iterations * float(strength)))

    for i in range(steps):
        with tf.GradientTape() as tape:
            gen_vgg = preprocess_for_vgg(generated / 255.0)  # generated scaled back to [0,1] then to vgg
            outputs = extractor(gen_vgg)

            # style loss
            s_loss = 0.0
            gen_style_feats = outputs[:len(STYLE_LAYERS)]
            for gf, st in zip(gen_style_feats, style_targets):
                g_gram = gram_matrix(gf)
                s_loss += tf.reduce_mean(tf.square(g_gram - st))

            s_loss = s_loss / float(len(STYLE_LAYERS))

            # content loss
            gen_content = outputs[-1]
            c_loss = tf.reduce_mean(tf.square(gen_content - content_target))

            # combined (alpha beta): alpha for style, beta for content
            alpha = 1.0  # style weight scale (can be parameterized)
            beta = 1.0   # content weight
            total_loss = alpha * s_loss + beta * c_loss

        grads = tape.gradient(total_loss, generated)
        opt.apply_gradients([(grads, generated)])
        # clip to valid pixel range
        generated.assign(tf.clip_by_value(generated, 0.0, 255.0))

        if (i + 1) % max(1, steps // 10) == 0:
            print(f"[NST] Step {i+1}/{steps} -> loss: {total_loss.numpy():.4f}")

    # convert to PIL
    final = generated / 255.0
    out_pil = tensor_to_pil(final)
    return out_pil


# ----------------------------
# Fast NST (TF-Hub) — arbitrary style transfer
# ----------------------------
# We load the hub model once on module import (if possible) to reduce overhead
try:
    _hub_nst_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
except Exception as e:
    _hub_nst_model = None
    # Do not raise here — app can call load_pretrained_fast_model() later

def load_pretrained_fast_model():
    global _hub_nst_model
    if _hub_nst_model is None:
        _hub_nst_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    return _hub_nst_model


def apply_fast_nst(content: Union[str, Image.Image],
                   style: Union[str, Image.Image],
                   content_max_dim: int = 512,
                   style_max_dim: int = 256,
                   strength: float = 1.0) -> Image.Image:

    model = load_pretrained_fast_model()

    if isinstance(content, (str, os.PathLike)):
        content_pil = load_image(content, max_dim=content_max_dim)
    else:
        content_pil = content

    if isinstance(style, (str, os.PathLike)):
        style_pil = load_image(style, max_dim=style_max_dim)
    else:
        style_pil = style

    content_t = pil_to_tensor(content_pil, max_dim=content_max_dim)
    style_t   = pil_to_tensor(style_pil, max_dim=style_max_dim)

    stylized = model(content_t, style_t)[0]

    # 🔥 STYLE STRENGTH CONTROL
    # make sure both tensors are same size
    stylized = tf.image.resize(
            stylized,
        (tf.shape(content_t)[1], tf.shape(content_t)[2])
    )


    output = content_t + strength * (stylized - content_t)
    output = tf.clip_by_value(output, 0.0, 1.0)

    return tensor_to_pil(output)



# ----------------------------
# Video style transfer (fast per-frame via hub)
# ----------------------------
def extract_frames(video_path: str, max_frames: int = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # BGR uint8
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return frames, fps, (width, height)


def frames_to_video(frames_bgr: List[np.ndarray], out_path: str, fps: float):
    # frames are BGR uint8
    if len(frames_bgr) == 0:
        raise ValueError("No frames to write")
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    return out_path


def stylize_video(input_video_path: str,
                  style_image: Union[str, Image.Image],
                  output_path: str = "stylized_video.mp4",
                  frame_limit: int = None,
                  smoothing: bool = True,
                  content_max_dim: int = 512):
    """
    Stylize a video using TF-Hub fast NST per-frame.
    Returns path to output video.
    """
    model = load_pretrained_fast_model()
    # load style tensor once
    if isinstance(style_image, (str, os.PathLike)):
        style_pil = load_image(style_image, max_dim=256)
    else:
        style_pil = style_image
    style_t = pil_to_tensor(style_pil, max_dim=256)

    frames, fps, (w, h) = extract_frames(input_video_path, max_frames=frame_limit)

    stylized_frames_bgr = []
    for f in tqdm(frames, desc="Stylizing frames"):
        # f is BGR uint8 — convert to PIL RGB
        img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        content_t = pil_to_tensor(pil, max_dim=content_max_dim)
        out = model(content_t, style_t)[0]  # (1,H,W,3) float32
        # to frame
        out_np = out.numpy().squeeze()
        out_np = np.clip(out_np * 255.0, 0, 255).astype('uint8')
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        stylized_frames_bgr.append(out_bgr)

    if smoothing:
        # simple temporal smoothing: average with previous
        smoothed = [stylized_frames_bgr[0]]
        for i in range(1, len(stylized_frames_bgr)):
            prev = smoothed[-1].astype(np.float32)
            cur = stylized_frames_bgr[i].astype(np.float32)
            blended = (0.7 * cur + 0.3 * prev).astype('uint8')
            smoothed.append(blended)
        stylized_frames_bgr = smoothed

    frames_to_video(stylized_frames_bgr, output_path, fps)
    return output_path


# ----------------------------
# Batch processing (process a folder/zip of images)
# ----------------------------
def batch_process_folder(input_zip_or_folder: str,
                         style_image_or_pil: Union[str, Image.Image],
                         out_dir: str = "batch_out",
                         model: str = "fast",
                         style_strength: float = 1.0):
    """
    Accepts either a folder path or zip file path.
    For zip: extracts into tempdir then processes.
    model: "fast" or "slow"
    Returns path to out_dir (created).
    """
    # Prepare input files list
    tmp = None
    if os.path.isfile(input_zip_or_folder) and input_zip_or_folder.lower().endswith('.zip'):
        tmp = tempfile.mkdtemp()
        import zipfile
        with zipfile.ZipFile(input_zip_or_folder, 'r') as z:
            z.extractall(tmp)
        root = tmp
    else:
        root = input_zip_or_folder

    # gather images
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []
    for root_dir, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root_dir, fn))

    os.makedirs(out_dir, exist_ok=True)

    for path in tqdm(files, desc="Batch stylize"):
        try:
            if model == "fast":
                out_pil = apply_fast_nst(path, style_image_or_pil)
            else:
                content_pil = load_image(path, max_dim=512)
                if isinstance(style_image_or_pil, (str, os.PathLike)):
                    style_pil = load_image(style_image_or_pil, max_dim=512)
                else:
                    style_pil = style_image_or_pil
                out_pil = apply_style_transfer(content_pil, [style_pil], strength=style_strength)
            # save
            fname = os.path.basename(path)
            out_path = os.path.join(out_dir, fname)
            out_pil.save(out_path)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if tmp:
        shutil.rmtree(tmp)
    return out_dir


# ----------------------------
# ESRGAN Upscale wrapper (TF-Hub)
# ----------------------------
try:
    _esrgan = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
except Exception:
    _esrgan = None


def load_esrgan_model():
    global _esrgan
    if _esrgan is None:
        _esrgan = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    return _esrgan


def upscale_image(input_img: Union[str, Image.Image], output_path: str = "upscaled.png"):
    """
    Upscale an image using TF-Hub ESRGAN model (if available).
    input_img: path or PIL
    """
    model = load_esrgan_model()
    if isinstance(input_img, (str, os.PathLike)):
        pil = load_image(input_img)
    else:
        pil = input_img
    lr = pil_to_tensor(pil, max_dim=None)  # keep original size
    sr = model(lr)
    if isinstance(sr, dict):
        sr = sr[list(sr.keys())[0]]
    sr_np = sr.numpy().squeeze()
    sr_np = np.clip(sr_np * 255.0, 0, 255).astype('uint8')
    out_pil = Image.fromarray(sr_np)
    out_pil.save(output_path)
    return output_path


# ----------------------------
# Small test helper (not run automatically)
# ----------------------------
def _quick_test():
    """Quick self test if running manually (not auto-run)."""
    print("Quick test: ensure hub model load works (may download models).")
    try:
        _ = load_pretrained_fast_model()
        print("Fast NST model loaded.")
    except Exception as e:
        print("Fast model load failed:", e)
    try:
        _ = load_esrgan_model()
        print("ESRGAN loaded.")
    except Exception as e:
        print("ESRGAN load failed (optional):", e)


if __name__ == "__main__":
    print("style_transfer.py module. Import functions from this file in your app.")
    # do NOT auto-run heavy operations here.
