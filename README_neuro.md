# 🎨 ArtifyX — Neural Style Transfer Engine

ArtifyX is an advanced AI-powered image and video stylization system that transforms ordinary visuals into expressive digital artwork. It leverages deep neural networks and real-time optimization pipelines to deliver fast, high-quality artistic transformations while preserving structural integrity.

---

## ✨ Core Highlights
- ⚡ Fast Arbitrary Neural Style Transfer
- 🧠 Dual-Phase Neural Stylization Engine
- 🖼️ Image & 🎥 Video Stylization Support
- 🎚️ Adjustable Style Strength Control
- 🚀 Streamlit-based Interactive UI
- 📈 Optimized for CPU (GPU optional)

---

## 🧠 Under the Hood (Models & Techniques)

ArtifyX is built upon a **Dual-Phase Neural Stylization Engine**, consisting of:

### Phase 1 — Adaptive Feature Alignment
- Extracts high-level semantic features from content images
- Aligns artistic representations without destroying structure

### Phase 2 — Dynamic Texture Synthesis
- Injects stylistic textures using neural feature blending
- Preserves edges while enabling chromatic abstraction

### 🔬 Models Used
- **TensorFlow Hub – Magenta Arbitrary Image Stylization**
- **Convolutional Neural Networks (CNN)**
- **Perceptual Feature Matching**
- **Dynamic Tensor Rescaling Pipeline**

---

## 📁 Project Structure

```
ArtifyX/
│
├── app.py                    # Streamlit application
├── style_transfer.py          # Core NST logic
├── NeuroStyleX_NST.ipynb      # Research & experimentation notebook
├── requirements.txt           # Dependencies
├── .gitignore
│
├── neuro/                     # Virtual environment (not uploaded to GitHub)
│
├── assets/
│   ├── content.jpg
│   ├── style.jpg
│
├── outputs/
│   ├── final_output.png
│   ├── stylized_output.mp4
│
└── README.md
```

> ⚠️ Note: Virtual environment folders (Lib/, Scripts/, Include/) are intentionally excluded from GitHub.

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/ArtifyX.git
cd ArtifyX
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
streamlit run app.py
```

---

## 🧪 Notebook Usage
Use `NeuroStyleX_NST.ipynb` for:
- Experimenting with styles
- Understanding NST internals
- Custom tuning & research

---

## ⚙️ Performance Notes
- Optimized for **fast inference**
- Image resizing prevents shape mismatch errors
- CPU friendly (GPU boosts performance further)

---

## 📌 Future Enhancements
- Batch image processing
- GPU auto-detection
- Style preset library
- Web deployment support

---

## 👩‍💻 Author
**Sakshi Gupta**  
B.Tech | AI • Data Science • Full Stack

---

## 📜 License
This project is licensed for educational and research purposes.