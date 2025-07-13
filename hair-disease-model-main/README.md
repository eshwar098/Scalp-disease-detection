
```markdown
# 🧠 Hair Disease Classification using CNN | Streamlit Web App

This project is a deep learning-based web application built with **TensorFlow**, **Keras**, and **Streamlit** that classifies hair/scalp diseases from uploaded images. The model predicts one of the following conditions:

- Alopecia  
- Baldness  
- Dandruff  

It is deployed using **Streamlit Cloud** or **Hugging Face Spaces**, with the model stored on **Hugging Face Hub** for dynamic loading.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://scalp-disease-detection.streamlit.app/)

---

## 🧰 Tech Stack

- Python **3.11.8**  
- TensorFlow **2.19.0**  
- Keras **3.10.0**  
- Streamlit  
- Hugging Face Hub (for model hosting)  

---

## 📂 Project Structure

```

hair-disease-classifier/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version pinning
├── Hair\_Disease.h5       # CNN model (downloaded from Hugging Face)
├── upload/               # Uploaded images (runtime only)

````

---

## ⚙️ Setup Instructions (Run Locally)

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR-USERNAME/hair-disease-classifier.git
cd hair-disease-classifier
````

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📦 Model Hosting

The `.h5` model is hosted on Hugging Face and downloaded at runtime using:

```
https://huggingface.co/ravikanth27/hair-disease-model/resolve/main/Hair_Disease.h5
```

---

## 💡 Features

* Upload image files (`.jpg`, `.png`, etc.)
* On-device prediction using CNN model
* Softmax confidence score with predicted label
* Lightweight and fast UI with Streamlit

---

## 📌 Requirements

* Python **3.11.8**
* TensorFlow **2.19.0**
* Keras **3.10.0**
* Streamlit
* NumPy
* Requests

---

## 🧠 Model Info

The model is a **Convolutional Neural Network (CNN)** trained on custom hair disease image data (180x180 resolution). It uses softmax activation in the output layer to predict one of the three classes.

---

## 🙏 Acknowledgments

* [Streamlit](https://streamlit.io)
* [Hugging Face](https://huggingface.co)
* [TensorFlow](https://www.tensorflow.org)
* \[Your Dataset Source]

---

## 📬 Contact

Made with ❤️ by **Ravikanth Kothagudem**
📧 Email: [your-email@example.com](mailto:23r25a3302@mlrit.ac.in)
🔗 GitHub: [@yourusername](https://github.com/23r25a3302)

