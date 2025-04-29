import streamlit as st
import openai
from ultralytics import YOLO
import json
from PIL import Image
import tempfile

# === SETTINGS ===
OPENAI_API_KEY = ""  # <<<<<<<<<<<<<< Set your API Key here go to openaiplatforms.com to get your api key
YOLO_MODEL_PATH = "best.pt"  # <<<<<<<<<<<<<< Set your model path
CONFIDENCE_THRESHOLD = 0.3
OPENAI_MODEL = "gpt-4o"

# === FUNCTIONS ===

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def run_detection(model, image_path):
    results = model(image_path)
    return results

def extract_detections(results, model, confidence_threshold):
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf.item())
        if conf >= confidence_threshold:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            bbox = [float(x) for x in box.xyxy[0]]  # [x1, y1, x2, y2]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox
            })
    return detections

def build_prompt(detections):
    prompt = f"""
You are an expert wildfire incident analyst.

Based on the following fire and smoke detections (label, confidence, bounding boxes):

{json.dumps(detections, indent=2)}

Please provide:
1. **Severity Estimation**: How severe is the fire?
2. **Resource Recommendation**: How many fire trucks, firefighters, and resources are needed?
3. **Evacuation Advisory**: Should civilians be evacuated? How far?
4. **Action Priority**: Is immediate action needed or monitoring sufficient?
5. **Situation Reporting**: Write a professional short report for emergency responders.

Be specific but concise. Assume your answer will be sent to a real fire emergency team.
"""
    return prompt

def call_chatgpt_api(prompt, api_key, model="gpt-4o"):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fire emergency response expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800
    )
    return response['choices'][0]['message']['content']

# === STREAMLIT APP ===

st.set_page_config(page_title="Wildfire Emergency Analyzer", layout="centered")
st.title("🔥 Wildfire Emergency Analyzer")

st.write("Upload an image showing smoke or fire. The app will detect fire/smoke areas and generate an emergency report using ChatGPT.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("Analyze"):
        with st.spinner("Detecting fire and smoke..."):
            model = load_yolo_model(YOLO_MODEL_PATH)
            results = run_detection(model, tmp_path)
            detections = extract_detections(results, model, CONFIDENCE_THRESHOLD)

        if detections:
            # Save and show YOLO detection output
            annotated_image = results[0].plot()  # Draws boxes
            annotated_pil_image = Image.fromarray(annotated_image)

            st.subheader("📷 Detection Output")
            st.image(annotated_pil_image, caption="YOLOv8 Detected Fire/Smoke", use_column_width=True)

            with st.spinner("Asking ChatGPT for Emergency Response Analysis..."):
                prompt = build_prompt(detections)
                analysis = call_chatgpt_api(prompt, OPENAI_API_KEY, OPENAI_MODEL)

            st.success("✅ Emergency Report Ready!")

            st.subheader("📋 Emergency Fire Situation Report")
            for section in analysis.split('\n\n'):
                st.markdown(section)

        else:
            st.warning("⚠️ No fire or smoke detected with sufficient confidence.")
