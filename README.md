# 🩺 Automated Retinopathy of Prematurity (ROP) Clinical Screening Suite

An end-to-end **Deep Learning-based clinical screening application** designed to detect **Retinopathy of Prematurity (ROP)** from infant retinal fundus images. The system uses **Transfer Learning with ResNet50** to classify retinal images and provides predictions through an interactive **Gradio** web interface.

---## 🛠️ Tech Stack

- 🐍 **Programming Language:** Python
- 🧠 **Core AI Framework:** TensorFlow 2.x & Keras API
- 🏗️ **Model Architecture:** ResNet50 (Transfer Learning)
- 🖼️ **Image Processing:** OpenCV (cv2) & TensorFlow Image Processing (`tf.image`)
- 🤖 **Machine Learning:** Scikit-learn
- 🌐 **Web Interface:** Gradio
- 📊 **Data Processing:** NumPy & Pandas
- 💻 **Development Environment:** Google Colab
- ⚡ **Hardware:** NVIDIA T4 GPU (Google Colab)
- 📂 **Dataset Source:** Kaggle Retinal Fundus Image Dataset

# 📊 Model Performance

- **Model:** ResNet50 Transfer Learning
- **Image Size:** 224 × 224 pixels
- **Classification:** Binary (ROP / Healthy Retina)
- **Framework:** TensorFlow & Keras
- **Inference:** Real-time prediction using Gradio

> *Performance may vary depending on the dataset and training configuration.*
                            
## 📂 Project Structure
ROP-Detection-Using-Deep-Learning/
│
├── app.py                  # Gradio web application
├── main_training.py        # Model training script
├── README.md
├── requirements.txt
├── resnet50_rop_weights.h5
└── sample_images/
```

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ROP-Detection-Using-Deep-Learning.git
```

Move to the project folder:

```bash
cd ROP-Detection-Using-Deep-Learning
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```
# 📈 Future Enhancements

- Multi-class ROP severity classification
- Explainable AI (Grad-CAM)
- Cloud deployment
- Mobile application support
- Improved model performance using larger datasets

---
# 👩‍💻 Author
**Anjali Singh**
B.Tech – Computer Science Engineering

# 📄 License
This project is licensed under the **MIT License**.
⭐ If you found this project helpful, please consider giving it a **Star** on GitHub
