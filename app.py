import streamlit as st
import pandas as pd
import io
import re
import folium
from streamlit_folium import folium_static
import numpy as np
import os
from datetime import datetime, timedelta
import sys
import requests
from math import radians, cos, sin, asin, sqrt
import json
import openai

# Add scripts directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# Import after adding to path
from scripts.rag_retriever import handle_user_query, is_prediction_query, extract_location_from_query, get_predictions_for_location, extract_time_from_query

# Set page config
st.set_page_config(page_title="Wildfire Prediction System", page_icon="🔥", layout="wide")

# Set paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'predictions')

# Create predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# === SETTINGS ===
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"
CONFIDENCE_THRESHOLD = 0.3

# Function to build prompt for ChatGPT
def build_prompt(prediction, probability, acres_pred, detections, location=None, weather=None):
    weather_info = ""
    if weather:
        weather_info = f"""
Weather conditions:
- Temperature Max: {weather.get('tempmax', 'N/A')} °C  
- Temperature Min: {weather.get('tempmin', 'N/A')} °C  
- Precipitation: {weather.get('precip', 'N/A')} mm  
- Wind Speed: {weather.get('windspeed', 'N/A')} km/h  
"""
    
    location_info = f"Location: {location.title()}" if location else ""
    
    return f"""
You are an expert wildfire incident analyst.

{location_info}

Fire prediction results:
- Predicted Fire Start: {'Yes' if prediction == 1 else 'No'}
- Probability of Wildfire: {probability:.2%}
- Estimated Acres Burned: {acres_pred:,.0f} acres

{weather_info}

Detected fire/smoke objects:
{json.dumps(detections, indent=2)}

Please provide:
1. **Severity Estimation**: How severe is the fire?
2. **Resource Recommendation**: How many fire trucks, firefighters, and resources are needed?
3. **Evacuation Advisory**: Should civilians be evacuated? How far?
4. **Action Priority**: Is immediate action needed or monitoring sufficient?
5. **Situation Reporting**: Write a professional short report for emergency responders.

Be specific but concise. Assume your answer will be sent to a real fire emergency team.
"""

# Function to call ChatGPT API
def call_chatgpt_api(prompt, api_key, model="gpt-4o"):
    openai.api_key = api_key
    try:
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
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return "Error generating report. Please try again later."

# ------- ADD NAVIGATION HERE -------
# Sidebar navigation
page = st.sidebar.radio(
    "Select Module:",
    [
        "🔥 Wildfire Prediction System",  # Combined module
        "🖼️ Fire/Smoke Image Detection"
    ]
)

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This integrated system combines:
- Natural language wildfire prediction
- Location-based weather & fire forecasting 
- Image-based fire detection

Version 1.0
""")

# ------- COMBINED FUNCTION FOR CHATBOT AND MAP -------
def wildfire_prediction_system():
    """Combined function for chatbot and map-based prediction"""
    
    st.title("🔥 Wildfire Prediction System")
    
    # Create tabs for the different views
    tab1, tab2 = st.tabs(["Chatbot Interface", "Interactive Map"])
    
    with tab1:
        st.header("Wildfire Prediction Chatbot")
        st.write("Ask questions about wildfire risks, predictions, and historical data.")
        
        # Initialize session state variables for the chatbot
        if 'processed_questions' not in st.session_state:
            st.session_state.processed_questions = {}
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        if 'current_answer' not in st.session_state:
            st.session_state.current_answer = ""
        if 'has_table' not in st.session_state:
            st.session_state.has_table = False
        if 'excel_data' not in st.session_state:
            st.session_state.excel_data = None
        if 'map_data' not in st.session_state:
            st.session_state.map_data = None
        if 'has_map' not in st.session_state:
            st.session_state.has_map = False
        if 'is_prediction' not in st.session_state:
            st.session_state.is_prediction = False
        if 'prediction_location' not in st.session_state:
            st.session_state.prediction_location = None
        if 'prediction_time' not in st.session_state:
            st.session_state.prediction_time = None
        if 'prediction_data' not in st.session_state:
            st.session_state.prediction_data = None

        # Chatbot input
        user_question = st.text_input("Ask a wildfire prediction or analysis question:", key="chatbot_question_input")
        
        # Process chatbot query
        if user_question and (user_question != st.session_state.current_question):
            with st.spinner("Thinking..."):
                # Update the current question
                st.session_state.current_question = user_question
                
                # Check if this is a prediction query
                st.session_state.is_prediction = is_prediction_query(user_question)
                
                # If it's a prediction, extract location and time
                if st.session_state.is_prediction:
                    st.session_state.prediction_location = extract_location_from_query(user_question)
                    st.session_state.prediction_time = extract_time_from_query(user_question)
                    
                    # Get prediction data
                    if st.session_state.prediction_location:
                        st.session_state.prediction_data = get_predictions_for_location(
                            st.session_state.prediction_location, 
                            st.session_state.prediction_time
                        )
                
                # Process the question
                answer = handle_user_query(user_question, top_k=5)
                st.session_state.current_answer = answer
                
                # Extract table and prepare Excel data
                structured_df = extract_table_from_answer(answer)
                st.session_state.has_table = structured_df is not None and not structured_df.empty
                
                if st.session_state.has_table:
                    st.session_state.excel_data = convert_df_to_excel(structured_df)
                    
                    # Extract location data for map
                    has_map, map_df, lat_col, lon_col = extract_location_data(structured_df)
                    st.session_state.has_map = has_map
                    
                    if has_map:
                        # Create the map
                        st.session_state.map_data = create_wildfire_map(map_df, lat_col, lon_col)
                else:
                    st.session_state.excel_data = create_text_excel(user_question, answer)
                    st.session_state.has_map = False

        # Display chatbot results
        if st.session_state.current_question:
            st.markdown(f"**You:** {st.session_state.current_question}")
            st.markdown(f"**Bot:** {st.session_state.current_answer}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.session_state.is_prediction:
                    st.markdown("### 🔮 Wildfire Prediction Analysis")
                    
                    if st.session_state.prediction_data:
                        pred_data = st.session_state.prediction_data
                        location = pred_data.get('location', pred_data.get('region', 'this area'))
                        time_display = format_time_info(st.session_state.prediction_time)
                        prob = pred_data.get('probability', pred_data.get('avg_probability', 0)) * 100
                        risk = pred_data.get('risk_category', 'Unknown')
                        
                        risk_color = "green"
                        if prob > 25: risk_color = "blue"
                        if prob > 50: risk_color = "orange"
                        if prob > 75: risk_color = "red"
                        
                        st.markdown(f"**Location:** {location.title()}")
                        st.markdown(f"**Prediction Period:** {time_display}")
                        st.markdown(f"**Risk Category:** {risk}")
                        st.progress(prob/100, text=f"Risk Probability: {prob:.1f}%")
                        st.markdown(f"<span style='color:{risk_color};font-weight:bold;font-size:18px;'>Risk Level: {risk}</span>", unsafe_allow_html=True)
                        
                        if 'weather_data' in pred_data:
                            st.markdown("### 🌤️ Weather Conditions")
                            weather = pred_data['weather_data']
                            st.markdown(f"Temperature: {weather['temperature']} | " +
                                        f"Humidity: {weather['humidity']} | " +
                                        f"Wind: {weather['wind']}")
                        
                        if 'drought_data' in pred_data:
                            st.markdown("### 💧 Drought Conditions")
                            drought = pred_data['drought_data']
                            st.markdown(f"**Current Status:** {drought['level']} Drought")
                            st.markdown(f"**Impact on Fire Risk:** {drought['impact']}")
                        
                        month = st.session_state.prediction_time.get('month') if st.session_state.prediction_time else None
                        if month:
                            if month in [6, 7, 8, 9]:
                                st.info("☀️ Summer months typically have higher wildfire risk in California due to dry conditions.")
                            elif month in [12, 1, 2]:
                                st.info("❄️ Winter months typically have lower wildfire risk in California due to precipitation.")
                        
                        # Add Emergency Report Generation with ChatGPT
                        if prob > 60:  # Only for high-risk predictions
                            st.markdown("### 🚨 Emergency Response Report")
                            
                            if st.button("Generate Emergency Report", key="chatbot_report_btn"):
                                with st.spinner("Generating emergency report using AI..."):
                                    # Create mock detections since we don't have image data here
                                    mock_detections = []
                                    
                                    # Use estimated acreage based on probability
                                    acres_pred = max(100, int(prob * 100))  # Simple scaling
                                    
                                    # Get weather dict if available, otherwise None
                                    weather_dict = pred_data.get('weather_data', None)
                                    
                                    # Build prompt for ChatGPT
                                    prompt = build_prompt(1, prob/100, acres_pred, mock_detections, location, weather_dict)
                                    
                                    # Call ChatGPT API
                                    report = call_chatgpt_api(prompt, OPENAI_API_KEY, OPENAI_MODEL)
                                    
                                    # Display the report
                                    for section in report.split('\n\n'):
                                        st.markdown(section)
                
                if st.session_state.has_map:
                    st.markdown("### 🗺️ Wildfire Prediction Map")
                    folium_static(st.session_state.map_data, width=700)
                elif st.session_state.has_table:
                    st.markdown("### 📊 Wildfire Prediction Data")
                    st.dataframe(extract_table_from_answer(st.session_state.current_answer))
            
            with col2:
                st.markdown("### 📄 Download Report")
                
                if st.session_state.excel_data:
                    filename = 'wildfire_prediction_report.xlsx' if st.session_state.has_table else 'wildfire_text_report.xlsx'
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=st.session_state.excel_data,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key="chatbot_download_btn"
                    )
                    
                    if not st.session_state.has_table:
                        st.info("ℹ️ No structured wildfire prediction table was detected, but a text report has been created for download.")
    
    with tab2:
        st.header("Weather & Fire Prediction Map")
        st.write("Click a location on the map and choose a date for specific weather and fire risk predictions.")
        
        # Fix package check
        missing_packages = []
        try:
            import xgboost
        except ImportError:
            missing_packages.append("xgboost")
        try:
            import sklearn
        except ImportError:
            missing_packages.append("scikit-learn")
        
        if missing_packages:
            st.error(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
            st.info("Please install the missing packages using pip:")
            st.code(f"pip install {' '.join(missing_packages)}")
        else:
            # Map with popup for clicking
            m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)
            folium.LatLngPopup().add_to(m)
            from streamlit_folium import st_folium
            click_data = st_folium(m, height=500, width=700, returned_objects=["last_clicked"])
            
            # Date input
            selected_date = st.date_input("📅 Select date for weather data", datetime.today(), key="map_date_input")
            
            # Process clicked location
            if click_data and click_data.get("last_clicked"):
                lat = click_data["last_clicked"]["lat"]
                lon = click_data["last_clicked"]["lng"]
                st.markdown(f"✅ You clicked at: **Lat:** {lat:.4f}, **Lon:** {lon:.4f}")
                
                # Find nearest coordinate
                nearest_name, nearest_coords, min_dist = None, None, float("inf")
                for name, (clat, clon) in ca_counties.items():
                    dist = haversine(lat, lon, clat, clon)
                    if dist < min_dist:
                        nearest_name = name
                        nearest_coords = (clat, clon)
                        min_dist = dist
                        
                st.markdown(f"📍 Nearest known location: **{nearest_name.title()}** ({nearest_coords[0]}, {nearest_coords[1]})")
                
                # Build datetime string
                query_time = "12:00:00"
                query_datetime = datetime.combine(selected_date, datetime.strptime(query_time, "%H:%M:%S").time())
                dt_string = query_datetime.strftime('%Y-%m-%dT%H:%M:%S')
                
                # Query weather API
                base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
                api_key = "ZYKHPURRJDNR5K6GQKCL5MH6Y"  # Replace with your actual key if needed
                fields = "tempmax,tempmin,precip,windspeed"
                location = f"{nearest_coords[0]},{nearest_coords[1]}"
                url = f"{base_url}/{location}/{dt_string}"
                params = {
                    "key": api_key,
                    "unitGroup": "metric",
                    "include": "hours",
                    "elements": fields,
                    "contentType": "json"
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        days = data.get("days", [{}])
                        hours = days[0].get("hours", [])
                        
                        # Try to get exact 12:00:00 hour data safely
                        hour_data = next((h for h in hours if h.get("datetime") == query_time), None)
                        
                        if hour_data:
                            source = "Hourly (12:00:00)"
                            weather = hour_data
                        else:
                            source = "Daily summary (fallback)"
                            weather = days[0]  # fallback if no hour match
                        
                        st.subheader(f"🌤️ Weather Data at 12:00:00 ({source})")
                        st.markdown(f"""
                            - **Temperature Max:** {weather.get('tempmax', 'N/A')} °C  
                            - **Temperature Min:** {weather.get('tempmin', 'N/A')} °C  
                            - **Precipitation:** {weather.get('precip', 'N/A')} mm  
                            - **Wind Speed:** {weather.get('windspeed', 'N/A')} km/h  
                        """)
                        
                        # Fire prediction section
                        try:
                            import joblib
                            import pandas as pd
                            
                            # Load the model files
                            model_path = os.path.join(MODEL_DIR, "xgb_fire_classifier.pkl")
                            
                            if os.path.exists(model_path):
                                model = joblib.load(model_path)
                                
                                # Prepare input features
                                month = selected_date.month
                                day_of_year = selected_date.timetuple().tm_yday
                                
                                # Get season
                                def get_season(month):
                                    if month in [12, 1, 2]:
                                        return "Winter"
                                    elif month in [3, 4, 5]:
                                        return "Spring"
                                    elif month in [6, 7, 8]:
                                        return "Summer"
                                    else:
                                        return "Fall"
                                
                                season = get_season(month)
                                
                                # Create DataFrame for prediction
                                input_df = pd.DataFrame([{
                                    'MAX_TEMP': weather.get('tempmax', 0) * 9/5 + 32,
                                    'MIN_TEMP': weather.get('tempmin', 0) * 9/5 + 32,
                                    'PRECIPITATION': weather.get('precip', 0),
                                    'AVG_WIND_SPEED': weather.get('windspeed', 0),
                                    'MONTH': month,
                                    'DAY_OF_YEAR': day_of_year,
                                    'SEASON': season
                                }])
                                
                                # Make prediction
                                prediction = model.predict(input_df)[0]
                                probability = model.predict_proba(input_df)[0][1]
                                
                                st.subheader("🔥 Fire Start Prediction")
                                st.markdown(f"**Prediction:** {'Yes' if prediction == 1 else 'No'}")
                                st.markdown(f"**Probability of wildfire:** {probability:.2%}")
                                
                                # If fire predicted, estimate acres burned
                                acres_pred = 0
                                if prediction == 1:
                                    regression_model_path = os.path.join(MODEL_DIR, "randomforest_model.pkl")
                                    if os.path.exists(regression_model_path):
                                        regression_model = joblib.load(regression_model_path)
                                        
                                        # Prepare input for regression model
                                        reg_input = pd.DataFrame([{
                                            'tempmin': weather.get('tempmin', 0),
                                            'tempmax': weather.get('tempmax', 0),
                                            'precip': weather.get('precip', 0),
                                            'windspeed': weather.get('windspeed', 0)
                                        }])
                                        
                                        # Predict burned acreage
                                        log_acres_pred = regression_model.predict(reg_input)[0]
                                        acres_pred = np.expm1(log_acres_pred)
                                        
                                        st.subheader("🔥 Predicted Acres Burned")
                                        st.markdown(f"Estimated wildfire size: **{acres_pred:,.0f} acres**")
                                
                                # Add Emergency Report Generation with ChatGPT if prediction is high risk
                                if prediction == 1 and probability > 0.4:
                                    st.markdown("### 🚨 Emergency Response Report")
                                    
                                    if st.button("Generate Emergency Report", key="map_report_btn"):
                                        with st.spinner("Generating emergency report using AI..."):
                                            # Mock detections since we don't have image data here
                                            mock_detections = []
                                            
                                            # Build prompt for ChatGPT
                                            prompt = build_prompt(prediction, probability, acres_pred, mock_detections, nearest_name, weather)
                                            
                                            # Call ChatGPT API
                                            report = call_chatgpt_api(prompt, OPENAI_API_KEY, OPENAI_MODEL)
                                            
                                            # Display the report
                                            st.success("✅ Emergency Report Ready!")
                                            for section in report.split('\n\n'):
                                                st.markdown(section)
                                
                            else:
                                st.warning(f"Model file not found at {model_path}")
                                st.info("Please ensure the xgb_fire_classifier.pkl file is in your models directory.")
                                
                        except Exception as e:
                            st.error(f"Error in fire prediction: {str(e)}")
                            
                    elif response.status_code == 401:
                        st.error("❌ API Error: Invalid API key. Please check your Visual Crossing API key.")
                    elif response.status_code == 429:
                        st.error("❌ API Error: Too many requests. The API usage limit has been reached.")
                    else:
                        st.error(f"❌ API Error: {response.status_code} — {response.text}")
                except requests.exceptions.Timeout:
                    st.error("❌ API request timed out. Please try again later.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API request failed: {str(e)}")
                except ValueError as e:
                    st.error(f"❌ Error parsing API response: {str(e)}")
            else:
                st.info("Click on the map to select a location.")

def fire_detection_module():
    """Function for fire/smoke detection in images using YOLO"""
    
    st.title("🖼️ Fire/Smoke Image Detection")
    st.write("Upload an image. The app will detect fire/smoke using a YOLOv8 model and generate an emergency report.")
    
    # Check for ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        st.error("⚠️ The ultralytics package is not installed")
        st.info("Please install it with: `pip install ultralytics`")
        return
        
    from PIL import Image
    import tempfile
    
    # Define model path
    yolo_model_path = os.path.join(MODEL_DIR, "best.pt")
    
    # Check if model exists with better error message
    if not os.path.exists(yolo_model_path):
        st.warning(f"⚠️ YOLO model not found at {yolo_model_path}")
        st.info("""
        Please download the model file (best.pt) from the GitHub repository:
        https://github.com/Agransh-Srivastava/Wildfire-Prediction-Model-LLM/tree/In_image_fire_detection/Fire_detection
        
        And place it in your models directory.
        """)
        return
            
    # Cache the model loading to avoid reloading on each rerun
    @st.cache_resource
    def load_yolo_model(model_path):
        return YOLO(model_path)
            
    def run_detection(model, image_path):
        return model(image_path)
            
    def extract_detections(results, model, confidence_threshold=0.3):
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf.item())
            if conf >= confidence_threshold:
                cls_id = int(box.cls.item())
                label = model.names[cls_id]
                bbox = [float(x) for x in box.xyxy[0]]
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox
                })
        return detections
            
    # File uploader for image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecated parameter
        
        # Location selection (optional)
        st.subheader("Location Information (Optional)")
        location_options = [""] + sorted(list(ca_counties.keys()))
        selected_location = st.selectbox("Select location (if known):", location_options, key="location_selector")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
                
        if st.button("Analyze Image", key="analyze_image_btn"):
            with st.spinner("Running detection..."):
                try:
                    model = load_yolo_model(yolo_model_path)
                    results = run_detection(model, tmp_path)
                    detections = extract_detections(results, model, CONFIDENCE_THRESHOLD)
                    
                    if detections:
                        # Draw boxes and display
                        annotated_image = results[0].plot()
                        annotated_pil_image = Image.fromarray(annotated_image)
                        st.image(annotated_pil_image, caption="Detection Result", use_container_width=True)  # Fixed deprecated parameter
                        
                        st.subheader("🔍 Detection Summary")
                        for d in detections:
                            st.markdown(f"- **{d['label']}** — Confidence: {d['confidence']:.2f}")
                            
                        # Count fire and smoke instances
                        fire_count = sum(1 for d in detections if d['label'].lower() == 'fire')
                        smoke_count = sum(1 for d in detections if d['label'].lower() == 'smoke')
                        
                        # Provide severity assessment
                        if fire_count > 0 and smoke_count > 0:
                            st.warning("⚠️ Both fire and smoke detected. This indicates an active wildfire.")
                        elif fire_count > 0:
                            st.warning("⚠️ Fire detected without significant smoke. This could be an early-stage wildfire.")
                        elif smoke_count > 0:
                            st.warning("⚠️ Smoke detected without visible fire. This may indicate a wildfire in the vicinity.")
                        
                        # Generate emergency report using ChatGPT
                        st.subheader("📋 Emergency Report")
                        
                        # Mock prediction values
                        prediction = 1 if fire_count > 0 or smoke_count > 0 else 0
                        probability = max(0.8 if fire_count > 0 else 0.5 if smoke_count > 0 else 0.3, max([d['confidence'] for d in detections]))
                        
                        # Estimate acres based on detection counts
                        acres_pred = 0
                        if fire_count > 0:
                            # Simple heuristic
                            acres_pred = fire_count * 500  # 500 acres per fire detection
                        elif smoke_count > 0:
                            acres_pred = smoke_count * 300  # 300 acres per smoke detection
                        
                        if st.button("Generate Emergency Report", key="image_report_btn"):
                            with st.spinner("Generating emergency report using ChatGPT..."):
                                # Build prompt for ChatGPT
                                prompt = build_prompt(prediction, probability, acres_pred, detections, selected_location)
                                
                                # Call ChatGPT API
                                report = call_chatgpt_api(prompt, OPENAI_API_KEY, OPENAI_MODEL)
                                
                                # Display the report
                                st.success("✅ Emergency Report Ready!")
                                for section in report.split('\n\n'):
                                    st.markdown(section)
                    else:
                        st.success("✅ No fire or smoke detected with high confidence.")
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

# Preserve all the helper functions from the original app
def extract_table_from_answer(answer_text):
    """
    Extract structured table from LLM answer.
    Enhanced to better handle Llama3's output formats and preserve all rows.
    """
    try:
        # Try parsing HTML tables first
        tables = pd.read_html(io.StringIO(answer_text), flavor='bs4')
        if tables and len(tables) > 0:
            return tables[0]
    except Exception:
        pass  # Ignore HTML parsing errors

    try:
        # Try parsing Markdown table with improved pattern
        # This pattern looks for proper markdown tables with header separator line
        table_pattern = r"(\|.+\|[\r\n|\n])((?:\|[\-:]+\|[\r\n|\n]))((?:\|.+\|[\r\n|\n])+)"
        match = re.search(table_pattern, answer_text, re.DOTALL)
        if match:
            table_text = match.group(0)
            
            # Manual parsing to preserve all rows
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            
            # Get headers
            header_line = lines[0]
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            # Skip separator line
            data = []
            for line in lines[2:]:  # Skip header and separator
                if '|' in line:
                    # Split by pipe and clean cells
                    cells = line.split('|')
                    # Remove empty cells from start/end (artifacts of split)
                    cells = [c.strip() for c in cells[1:-1] if c is not None]
                    # Make sure we have a cell for each header
                    while len(cells) < len(headers):
                        cells.append('')
                    data.append(cells)
            
            # Create DataFrame without dropping ANY rows
            df = pd.DataFrame(data, columns=headers)
            df = df.replace('', pd.NA)  # Convert empty strings to NA for better handling
            
            # Only drop columns that are completely empty
            df = df.dropna(axis=1, how='all')
            
            # DO NOT drop any rows
            return df
    except Exception as e:
        print(f"Error in main extraction: {e}")
        pass  # Try alternative method if this fails

    try:
        # Fallback: Parse any table-like structure line by line
        lines = [line.strip() for line in answer_text.split('\n') if '|' in line]
        if len(lines) >= 3:  # Need at least header, separator, and one data row
            # First line is header
            header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
            
            # Data rows (skip header and potential separator line)
            data_rows = []
            for line in lines[2:]:
                if '|' in line:
                    # Get cells, skip empty first/last cells from split
                    cells = [cell.strip() for cell in line.split('|')]
                    # Remove empty elements from start/end
                    while cells and not cells[0]:
                        cells.pop(0)
                    while cells and not cells[-1]:
                        cells.pop()
                    
                    if cells:  # Only add non-empty rows
                        # Ensure row has right number of columns
                        while len(cells) < len(header_cells):
                            cells.append('')
                        if len(cells) > len(header_cells):
                            cells = cells[:len(header_cells)]
                        
                        data_rows.append(cells)
            
            if data_rows:
                return pd.DataFrame(data_rows, columns=header_cells)
    except Exception as e:
        print(f"Error in fallback extraction: {e}")
        pass

    return None  # No structured table found

def convert_df_to_excel(df):
    """
    Convert dataframe to Excel with improved formatting and summary sheet.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data - preserve all rows
        df.to_excel(writer, index=False, sheet_name='Wildfire Predictions', na_rep='')
        
        # Format worksheet to better display the data
        workbook = writer.book
        worksheet = writer.sheets['Wildfire Predictions']
        
        # Auto-fit column widths
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).apply(len).max(), len(str(col))) + 2
            worksheet.set_column(i, i, max_len)
        
        # Create summary sheet with basic statistics for numeric columns
        try:
            # Identify numeric columns for summary
            numeric_cols = []
            for col in df.columns:
                try:
                    # Try to convert to numeric, ignoring errors
                    pd.to_numeric(df[col], errors='coerce')
                    # Only include if majority of values are numeric
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > len(df) / 2:
                        numeric_cols.append(col)
                except:
                    continue
            
            if numeric_cols:
                # Create summary DataFrame with basic statistics
                summary_data = {
                    'Metric': ['Count', 'Average', 'Min', 'Max', 'Sum']
                }
                
                for col in numeric_cols:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    summary_data[col] = [
                        numeric_values.count(),
                        numeric_values.mean(),
                        numeric_values.min(),
                        numeric_values.max(),
                        numeric_values.sum()
                    ]
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        except Exception as e:
            print(f"Error creating summary: {e}")
            # If summary creation fails, just continue without it
            pass
    
    processed_data = output.getvalue()
    return processed_data

def create_text_excel(question, answer):
    """
    Create an Excel file from text response when no table is detected.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main response sheet
        pd.DataFrame({
            'Question': [question],
            'Response': [answer]
        }).to_excel(writer, sheet_name='Llama3 Response', index=False)
        
        # Format worksheet
        workbook = writer.book
        worksheet = writer.sheets['Llama3 Response']
        
        # Set column widths
        worksheet.set_column(0, 0, 20)  # Question column
        worksheet.set_column(1, 1, 80)  # Response column
        
        # Simple key points extraction for summary sheet
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        key_points = [s for s in sentences if len(s.split()) > 5][:5]  # First 5 substantial sentences
        
        if key_points:
            pd.DataFrame({
                'Key Points': key_points
            }).to_excel(writer, sheet_name='Summary', index=False)
            
            # Format summary sheet
            summary_worksheet = writer.sheets['Summary']
            summary_worksheet.set_column(0, 0, 80)  # Key points column width
    
    return output.getvalue()

def extract_location_data(df):
    """
    Extract location data from dataframe for mapping.
    Returns tuple of (has_map_data, processed_df_with_coords).
    """
    # Make a copy to avoid modifying the original
    map_df = df.copy()
    
    # Look for location-related columns
    location_columns = []
    coordinate_columns = {'lat': None, 'lon': None}
    
    # Check for exact latitude/longitude columns
    for col in map_df.columns:
        col_lower = col.lower()
        if 'latitude' in col_lower or 'lat' == col_lower:
            coordinate_columns['lat'] = col
        elif 'longitude' in col_lower or 'lon' == col_lower or 'long' == col_lower:
            coordinate_columns['lon'] = col
        elif any(term in col_lower for term in ['location', 'county', 'region', 'area', 'place']):
            location_columns.append(col)
    
    # Check if we have explicit coordinates
    if coordinate_columns['lat'] and coordinate_columns['lon']:
        # Convert coordinates to numeric
        map_df[coordinate_columns['lat']] = pd.to_numeric(map_df[coordinate_columns['lat']], errors='coerce')
        map_df[coordinate_columns['lon']] = pd.to_numeric(map_df[coordinate_columns['lon']], errors='coerce')
        
        # Check if we have valid coordinates
        valid_coords = (
            map_df[coordinate_columns['lat']].notna() & 
            map_df[coordinate_columns['lon']].notna()
        ).sum() > 0
        
        if valid_coords:
            return True, map_df, coordinate_columns['lat'], coordinate_columns['lon']
    
    # If no explicit coordinates but we have location names, we can try to use California county centroids
    if location_columns:
        # Create new lat/lon columns
        map_df['latitude'] = None
        map_df['longitude'] = None
        
        # Try to match location names to known coordinates
        for idx, row in map_df.iterrows():
            for col in location_columns:
                location = str(row[col]).lower() if pd.notna(row[col]) else ""
                
                # Check for direct match
                if location in ca_counties:
                    map_df.at[idx, 'latitude'] = ca_counties[location][0]
                    map_df.at[idx, 'longitude'] = ca_counties[location][1]
                    break
                
                # Check for partial matches (e.g., "San Diego County" should match "san diego")
                for county, coords in ca_counties.items():
                    if county in location:
                        map_df.at[idx, 'latitude'] = coords[0]
                        map_df.at[idx, 'longitude'] = coords[1]
                        break
        
        # Convert to numeric
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        
        # Check if we found any valid coordinates
        valid_coords = (map_df['latitude'].notna() & map_df['longitude'].notna()).sum() > 0
        
        if valid_coords:
            return True, map_df, 'latitude', 'longitude'
    
    # If no location data found, check if the dataframe looks like it contains wildfire predictions
    # and use a default California map with random points
    wildfire_related = any(
        any(term in col.lower() for term in ['fire', 'wildfire', 'risk', 'hazard', 'burn', 'acre'])
        for col in df.columns
    )
    
    if wildfire_related and len(df) > 0:
        # Create synthetic coordinates for demonstration
        map_df['latitude'] = np.random.uniform(32.5, 42.0, size=len(map_df))  # California latitude range
        map_df['longitude'] = np.random.uniform(-124.4, -114.1, size=len(map_df))  # California longitude range
        return True, map_df, 'latitude', 'longitude'
    
    # No location data could be extracted
    return False, None, None, None

def create_wildfire_map(map_df, lat_col, lon_col):
    """
    Create a Folium map with wildfire prediction data.
    """
    # Create base map centered on California
    m = folium.Map(location=[37.8, -119.4], zoom_start=6, tiles="CartoDB positron")
    
    # Try to determine if we have a risk or severity column
    risk_cols = [col for col in map_df.columns if any(
        term in col.lower() for term in ['risk', 'hazard', 'severity', 'danger', 'probability', 'chance']
    )]
    
    # Add markers for each location
    for idx, row in map_df.iterrows():
        # Skip if coordinates are missing
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
            continue
            
        # Prepare popup content
        popup_content = "<b>Location Information</b><br>"
        
        # Add all non-coordinate columns to popup
        for col in map_df.columns:
            if col != lat_col and col != lon_col and pd.notna(row[col]):
                popup_content += f"<b>{col}:</b> {row[col]}<br>"
        
        # Determine marker color based on risk/severity if available
        marker_color = 'orange'  # Default
        
        if risk_cols:
            risk_col = risk_cols[0]
            try:
                risk_value = pd.to_numeric(row[risk_col], errors='coerce')
                if not pd.isna(risk_value):
                    # Scale to 0-100 if not already
                    if 0 <= risk_value <= 1:
                        risk_pct = risk_value * 100
                    else:
                        risk_pct = min(100, max(0, risk_value))
                    
                    # Set color based on risk percentage
                    if risk_pct < 25:
                        marker_color = 'green'
                    elif risk_pct < 50:
                        marker_color = 'blue'
                    elif risk_pct < 75:
                        marker_color = 'orange'
                    else:
                        marker_color = 'red'
            except:
                # If conversion fails, use default color
                pass
        
        # Create popup and add marker
        popup = folium.Popup(popup_content, max_width=300)
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup,
            icon=folium.Icon(color=marker_color, icon="fire", prefix="fa")
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 160px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; padding: 10px;
                border-radius: 5px;">
      <b>Wildfire Risk Level</b><br>
      <div style="display: flex; align-items: center; margin-top: 5px;">
        <div style="background-color: red; width: 20px; height: 20px; border-radius: 50%; margin-right: 5px;"></div>
        <span>High Risk</span>
      </div>
      <div style="display: flex; align-items: center; margin-top: 5px;">
        <div style="background-color: orange; width: 20px; height: 20px; border-radius: 50%; margin-right: 5px;"></div>
        <span>Moderate-High</span>
      </div>
      <div style="display: flex; align-items: center; margin-top: 5px;">
        <div style="background-color: blue; width: 20px; height: 20px; border-radius: 50%; margin-right: 5px;"></div>
        <span>Moderate-Low</span>
      </div>
      <div style="display: flex; align-items: center; margin-top: 5px;">
        <div style="background-color: green; width: 20px; height: 20px; border-radius: 50%; margin-right: 5px;"></div>
        <span>Low Risk</span>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add state boundary for context
    folium.GeoJson(
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json",
        name="California",
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        },
        highlight_function=lambda x: {'weight': 3},
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'])
    ).add_to(m)
    
    return m

def format_time_info(time_info):
    """
    Format time information for display.
    """
    if not time_info:
        return "Future"
    
    # Month names
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    # Get month and year
    month = time_info.get('month')
    year = time_info.get('year')
    
    if month and year:
        return f"{month_names[month]} {year}"
    elif month:
        return month_names[month]
    elif year:
        return str(year)
    elif time_info.get('season'):
        return f"{time_info['season'].capitalize()} {year if year else datetime.now().year + 1}"
    
    return "Future"

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371

# California counties dictionary globally available
ca_counties = {
    'alameda': (37.6017, -121.7195),
    'alpine': (38.5974, -119.8224),
    'amador': (38.4483, -120.6547),
    'butte': (39.6254, -121.6004),
    'calaveras': (38.1957, -120.6805),
    'colusa': (39.1789, -122.2339),
    'contra costa': (37.9323, -121.9018),
    'del norte': (41.7076, -123.9654),
    'el dorado': (38.7798, -120.5228),
    'fresno': (36.7378, -119.7871),
    'glenn': (39.5989, -122.3935),
    'humboldt': (40.7450, -123.8695),
    'imperial': (33.0114, -115.4734),
    'inyo': (36.5616, -117.4119),
    'kern': (35.3426, -118.7271),
    'kings': (36.0754, -119.8156),
    'lake': (39.1020, -122.7536),
    'lassen': (40.6195, -120.5910),
    'los angeles': (34.0522, -118.2437),
    'madera': (37.2220, -119.7681),
    'marin': (38.0834, -122.7633),
    'mariposa': (37.5791, -119.9132),
    'mendocino': (39.4381, -123.3911),
    'merced': (37.2010, -120.7120),
    'modoc': (41.5875, -120.7244),
    'mono': (37.9375, -118.8574),
    'monterey': (36.2400, -121.3100),
    'napa': (38.5025, -122.2654),
    'nevada': (39.2999, -120.7742),
    'orange': (33.7175, -117.8311),
    'placer': (39.0598, -120.8039),
    'plumas': (40.0024, -120.8039),
    'riverside': (33.9533, -117.3961),
    'sacramento': (38.5816, -121.4944),
    'san benito': (36.6108, -121.0829),
    'san bernardino': (34.1083, -117.2898),
    'san diego': (32.7157, -117.1611),
    'san francisco': (37.7749, -122.4194),
    'san joaquin': (37.9349, -121.2719),
    'san luis obispo': (35.2828, -120.6596),
    'san mateo': (37.5630, -122.3255),
    'santa barbara': (34.4208, -119.6982),
    'santa clara': (37.3541, -121.9552),
    'santa cruz': (36.9741, -122.0308),
    'shasta': (40.7909, -122.0388),
    'sierra': (39.5769, -120.5221),
    'siskiyou': (41.5906, -122.5403),
    'solano': (38.3105, -121.9018),
    'sonoma': (38.5405, -122.9961),
    'stanislaus': (37.5091, -120.9876),
    'sutter': (39.0342, -121.6941),
    'tehama': (40.1262, -122.2381),
    'trinity': (40.6487, -123.1144),
    'tulare': (36.2311, -118.8009),
    'tuolumne': (38.0278, -119.9647),
    'ventura': (34.3705, -119.1391),
    'yolo': (38.7318, -121.8072),
    'yuba': (39.2638, -121.3489),
    'modoc forest': (41.3835, -121.0460),
    'yosemite': (37.8651, -119.5383)
}

# Display the selected page based on navigation
if page == "🔥 Wildfire Prediction System":
    wildfire_prediction_system()
elif page == "🖼️ Fire/Smoke Image Detection":
    fire_detection_module()
