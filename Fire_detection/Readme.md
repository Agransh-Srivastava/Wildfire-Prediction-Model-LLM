# 🔥 Wildfire Detection using Areal Images

This project combines **YOLOv8** for real-time wildfire (fire and smoke) detection and **ChatGPT** for generating professional emergency reports based on detection outputs.

---
## 📊 Dataset: [D-Fire](https://github.com/gaiasd/DFireDataset?tab=readme-ov-file)

- **Name:** D-Fire: An Image Dataset for Fire and Smoke Detection
- **Authors:** Researchers from Gaia, Solutions on Demand (GAIA)
- **Description:**  
  D-Fire is a curated image dataset specifically designed for fire and smoke detection tasks.  
  It contains **over 21,000 images** covering diverse fire and smoke occurrences suitable for machine learning, deep learning, and object detection applications.
-[Download D-Fire dataset](https://drive.google.com/drive/folders/1DWgsQLVgkkLM8m-VcugHNpD5WYDbjYp5)
- **Usage in this Project:**  
  The D-Fire dataset is used to train the YOLOv8 model to accurately detect fire and smoke regions in images. It is organized into `train/`, `val/`, and `test/` subsets as per YOLO data format.

---

---

## 📂 Project Structure

```
D-fire/            # Dataset (train/val/test folders inside)
├── train/
├── val/
├── test/

data.yaml          # YOLOv8 dataset configuration file
Training.py        # YOLOv8 model training script
best.pt            # Trained best YOLOv8 model weights
app.py             # Streamlit app for detection and reporting
README.md          # Project instructions and documentation
requirements.txt   # Python package dependencies
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Agransh-Srivastava/Wildfire-Prediction-Model-LLM.git
cd Wildfire-Prediction-Model-LLM
git checkout In_image_fire_detection
cd Fire_detection

```

### 2. Install Required Python Packages

```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use

### 🔹 A. (Optional) Retrain YOLOv8 Model

If you wish to retrain the YOLOv8 model on the dataset:

```bash
python Training.py
```

This will generate new model weights saved in the project directory.

### 🔹 B. Run the Streamlit App

Launch the Streamlit application:

```bash
streamlit run app.py
```

**In the app you can:**
- Upload an image containing fire/smoke.
- View YOLOv8 detection outputs (bounding boxes for fire/smoke).
- Generate an emergency fire situation report powered by ChatGPT.

---

## 🛠 Important Notes

- **OpenAI API Key:**  
  Make sure to replace `"your-openai-api-key-here"` inside `app.py` with your actual OpenAI API key.

- **Model Weights Path:**  
  Ensure the `best.pt` path inside `app.py` matches your saved model weights location.

- **Dataset Structure:**  
  `D-fire/` should contain `train/`, `val/`, and `test/` subdirectories along with a proper `data.yaml`.

---

## 📋 Example Outputs

- 🖼️ YOLOv8 detection output with fire/smoke bounding boxes.
- 📋 ChatGPT-generated emergency report including:
  - Severity Estimation
  - Resource Recommendations
  - Evacuation Advisory
  - Action Priority
  - Professional Emergency Situation Reporting

---

## 📄 Requirements

The following Python packages are needed:

- `ultralytics`
- `openai`
- `streamlit`
- `pillow`
- `numpy`

You can install them via:

```bash
pip install -r requirements.txt
```

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 💬 Acknowledgements

This project is built using open-source tools like Ultralytics YOLOv8, OpenAI's GPT models, and Streamlit.

---
