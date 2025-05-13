import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import re

# OpenWeatherMap API configuration
OWM_API_KEY = "bf9f708142dc53c972229cd59ca86846"  # You'll need to sign up for a free API key
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')

# Ensure predictions directory exists
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Ollama setup
OLLAMA_SERVER_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Your local model name

# Global variables to hold models - only loaded when needed
_embedding_model = None
_faiss_index = None
_metadata = None
_prediction_model = None
_feature_columns = None

def get_weather_forecast(location, days_ahead=30):
    """Get weather forecast for a location."""
    params = {
        "q": f"{location},US",
        "appid": OWM_API_KEY,
        "units": "imperial"
    }

    try:
        response = requests.get(OWM_BASE_URL, params=params)
        data = response.json()

        forecast_data = {
            "temperature": [],
            "humidity": [],
            "wind": [],
            "precipitation": []
        }

        for item in data['list']:
            forecast_data["temperature"].append(item['main']['temp'])
            forecast_data["humidity"].append(item['main']['humidity'])
            forecast_data["wind"].append(item['wind']['speed'])
            if 'rain' in item and '3h' in item['rain']:
                forecast_data["precipitation"].append(item['rain']['3h'])
            else:
                forecast_data["precipitation"].append(0)

        avg_forecast = {
            "avg_temp": sum(forecast_data["temperature"])/len(forecast_data["temperature"]),
            "avg_humidity": sum(forecast_data["humidity"])/len(forecast_data["humidity"]),
            "avg_wind": sum(forecast_data["wind"])/len(forecast_data["wind"]),
            "total_precip": sum(forecast_data["precipitation"])
        }

        return avg_forecast
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_drought_data(location):
    """Get current drought data for California regions."""
    drought_levels = {
        "los angeles": 3,
        "san diego": 3,
        "san francisco": 2,
        "sacramento": 2,
        "northern california": 2,
        "southern california": 3,
        "central california": 3
    }

    default_level = 2

    for region, level in drought_levels.items():
        if region in location.lower():
            return {
                "drought_level": level,
                "description": ["None", "Abnormally Dry", "Moderate", "Severe", "Extreme", "Exceptional"][level],
                "impact_factor": 1 + (level * 0.15)
            }

    return {
        "drought_level": default_level,
        "description": ["None", "Abnormally Dry", "Moderate", "Severe", "Extreme", "Exceptional"][default_level],
        "impact_factor": 1 + (default_level * 0.15)
    }

def is_prediction_query(query):
    """
    Determine if the query is asking for a prediction.
    
    Args:
        query (str): User query
    
    Returns:
        bool: True if prediction query, False otherwise
    """
    prediction_keywords = [
        'predict', 'forecast', 'future', 'upcoming', 'next year', 
        'next month', 'next season', 'coming', 'soon', 'will there be',
        'how likely', 'probability', 'chance', 'risk', 'expect', 'anticipate'
    ]
    
    wildfire_keywords = [
        'fire', 'wildfire', 'blaze', 'burn', 'flame', 'heat'
    ]
    
    # Check if query contains prediction keywords
    has_prediction_term = any(term in query.lower() for term in prediction_keywords)
    has_wildfire_term = any(term in query.lower() for term in wildfire_keywords)
    
    return has_prediction_term and has_wildfire_term

def extract_location_from_query(query):
    """
    Extract location mentions from a query.
    
    Args:
        query (str): User query
    
    Returns:
        str: Extracted location or None
    """
    # List of known California locations (cities, counties, regions)
    california_locations = [
        'los angeles', 'san francisco', 'san diego', 'sacramento', 'fresno',
        'long beach', 'oakland', 'bakersfield', 'anaheim', 'santa ana',
        'riverside', 'stockton', 'irvine', 'chula vista', 'fremont',
        'san jose', 'modesto', 'fontana', 'moreno valley', 'santa clarita',
        'alameda', 'alpine', 'amador', 'butte', 'calaveras', 'colusa',
        'contra costa', 'del norte', 'el dorado', 'fresno', 'glenn',
        'humboldt', 'imperial', 'inyo', 'kern', 'kings', 'lake', 'lassen',
        'madera', 'marin', 'mariposa', 'mendocino', 'merced', 'modoc',
        'mono', 'monterey', 'napa', 'nevada', 'orange', 'placer', 'plumas',
        'riverside', 'sacramento', 'san benito', 'san bernardino', 'san diego',
        'san francisco', 'san joaquin', 'san luis obispo', 'san mateo',
        'santa barbara', 'santa clara', 'santa cruz', 'shasta', 'sierra',
        'siskiyou', 'solano', 'sonoma', 'stanislaus', 'sutter', 'tehama',
        'trinity', 'tulare', 'tuolumne', 'ventura', 'yolo', 'yuba',
        'northern california', 'southern california', 'central california',
        'bay area', 'central valley', 'sierra nevada', 'central coast'
    ]
    
    # Check for location mentions
    query_lower = query.lower()
    
    for location in california_locations:
        if location in query_lower:
            return location
    
    # If no specific location is found but 'california' is mentioned, return the state
    if 'california' in query_lower:
        return 'california'
    
    # Default to None if no location found
    return None

def extract_time_from_query(query):
    """
    Extract time information (month, season, year) from query.
    
    Args:
        query (str): User query
    
    Returns:
        dict: Time information including month, year, and season
    """
    query_lower = query.lower()
    time_info = {
        'month': None,
        'year': None,
        'season': None
    }
    
    # Extract year
    year_pattern = r'\b(20\d{2})\b'  # Match years like 2023, 2024, 2025, etc.
    year_match = re.search(year_pattern, query_lower)
    if year_match:
        time_info['year'] = int(year_match.group(1))
    
    # Extract month
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 
        'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for month_name, month_num in month_names.items():
        if month_name in query_lower:
            time_info['month'] = month_num
            break
    
    # Extract season
    seasons = {
        'winter': {'months': [12, 1, 2], 'peak_month': 1},
        'spring': {'months': [3, 4, 5], 'peak_month': 4},
        'summer': {'months': [6, 7, 8], 'peak_month': 7},
        'fall': {'months': [9, 10, 11], 'peak_month': 10},
        'autumn': {'months': [9, 10, 11], 'peak_month': 10}
    }
    
    for season_name, season_data in seasons.items():
        if season_name in query_lower:
            time_info['season'] = season_name
            # If no specific month mentioned, use the peak month of the season
            if time_info['month'] is None:
                time_info['month'] = season_data['peak_month']
            break
    
    # Handle relative time references
    if 'next month' in query_lower:
        next_month = datetime.now().month + 1
        if next_month > 12:
            next_month = 1
            if time_info['year'] is None:
                time_info['year'] = datetime.now().year + 1
        time_info['month'] = next_month
    
    elif 'next year' in query_lower:
        time_info['year'] = datetime.now().year + 1
    
    # If no year specified, use next year for future predictions
    if time_info['year'] is None:
        current_month = datetime.now().month
        if time_info['month'] is not None:
            if time_info['month'] < current_month:
                time_info['year'] = datetime.now().year + 1
            else:
                time_info['year'] = datetime.now().year
        else:
            time_info['year'] = datetime.now().year + 1
    
    # If still no month specified, use a reasonable default
    if time_info['month'] is None:
        # Default to summer fire season peak (July)
        time_info['month'] = 7
    
    return time_info

def get_predictions_for_location(location, time_info=None, months_ahead=6):
    """
    Get wildfire predictions for a specific location and time.
    
    Args:
        location (str): Location name (city, county, region)
        time_info (dict): Dictionary containing month, year, season information
        months_ahead (int): How many months ahead to predict (if time_info not provided)
    
    Returns:
        dict: Prediction results or None if predictions not available
    """
    # Use provided time_info or create default
    if time_info is None:
        time_info = {
            'month': None,
            'year': None,
            'season': None
        }
        # Default to X months ahead
        future_date = datetime.now() + timedelta(days=30*months_ahead)
        time_info['month'] = future_date.month
        time_info['year'] = future_date.year
    
    # Create the target date based on time_info
    target_year = time_info['year'] if time_info['year'] is not None else datetime.now().year + 1
    target_month = time_info['month'] if time_info['month'] is not None else 7  # Default July
    target_day = 15

    target_date = datetime(target_year, target_month, target_day)

    print(f"Generating prediction for {location} on {target_date.strftime('%Y-%m-%d')}")

    # Initialize prediction with base values
    month = target_date.month
    prediction = {
        'location': location,
        'date': target_date.strftime('%Y-%m-%d'),
        'risk_distribution': {'Low': 0, 'Moderate': 0, 'High': 0, 'Extreme': 0}
    }
    
    # Set base probability based on month (seasonal fire risk)
    if month in [7, 8, 9]:  # Peak fire season
        base_probability = 0.55
        prediction['risk_distribution'] = {'Low': 1, 'Moderate': 3, 'High': 5, 'Extreme': 1}
    elif month in [5, 6, 10]:  # Shoulder season
        base_probability = 0.35
        prediction['risk_distribution'] = {'Low': 3, 'Moderate': 5, 'High': 2, 'Extreme': 0}
    elif month in [11, 4]:  # Transition season
        base_probability = 0.20
        prediction['risk_distribution'] = {'Low': 5, 'Moderate': 4, 'High': 1, 'Extreme': 0}
    else:  # Winter (low fire season)
        base_probability = 0.10
        prediction['risk_distribution'] = {'Low': 8, 'Moderate': 2, 'High': 0, 'Extreme': 0}
    
    # Initialize risk modifiers
    risk_modifiers = 0.0
    
    # First determine region category for consistent treatment
    location_lower = location.lower()
    
    # Define region classifications
    is_southern_coastal = any(loc in location_lower for loc in ['los angeles', 'san diego', 'orange', 'ventura', 'santa barbara', 'long beach', 'santa ana', 'irvine', 'anaheim', 'chula vista'])
    is_southern_inland = any(loc in location_lower for loc in ['riverside', 'san bernardino', 'imperial', 'fontana', 'moreno valley', 'bakersfield', 'kern'])
    is_central_valley = any(loc in location_lower for loc in ['fresno', 'sacramento', 'stockton', 'modesto', 'merced', 'stanislaus', 'san joaquin', 'madera', 'kings', 'tulare', 'yolo', 'sutter', 'yuba', 'placer', 'colusa', 'glenn', 'butte', 'tehama', 'central valley'])
    is_central_coast = any(loc in location_lower for loc in ['monterey', 'san luis obispo', 'santa cruz', 'san benito', 'central coast'])
    is_bay_area = any(loc in location_lower for loc in ['san francisco', 'oakland', 'fremont', 'san jose', 'san mateo', 'santa clara', 'alameda', 'contra costa', 'marin', 'napa', 'solano', 'sonoma', 'bay area'])
    is_northern_coastal = any(loc in location_lower for loc in ['mendocino', 'humboldt', 'del norte', 'lake', 'sonoma', 'trinity'])
    is_northern_inland = any(loc in location_lower for loc in ['shasta', 'siskiyou', 'lassen', 'plumas', 'modoc', 'tehama'])
    is_mountain_sierra = any(loc in location_lower for loc in ['sierra', 'tahoe', 'nevada', 'placer', 'el dorado', 'alpine', 'amador', 'calaveras', 'tuolumne', 'mariposa', 'sierra nevada'])
    is_eastern_sierra = any(loc in location_lower for loc in ['mono', 'inyo', 'death valley'])
    
    # Generic region fallbacks
    if 'southern california' in location_lower:
        is_southern_coastal = True
        is_southern_inland = True
    if 'northern california' in location_lower:
        is_northern_coastal = True
        is_northern_inland = True
    if 'central california' in location_lower:
        is_central_valley = True
        is_central_coast = True
    
    # Apply regional base adjustments based on the more granular classification
    
    # Southern California Coastal (LA, San Diego, etc.)
    if is_southern_coastal:
        if month in [9, 10, 11]:  # Santa Ana wind season
            risk_modifiers += 0.12
            if 'los angeles' in location_lower or 'ventura' in location_lower:
                risk_modifiers += 0.03  # LA and Ventura have extra risk during Santa Ana
        elif month in [5, 6, 7, 8]:
            risk_modifiers += 0.06
        else:
            risk_modifiers += 0.02
    
    # Southern California Inland (Riverside, San Bernardino, etc.)
    if is_southern_inland:
        if month in [6, 7, 8, 9, 10]:  # Extended fire season in hot inland areas
            risk_modifiers += 0.15
        elif month in [11, 5]:  # Shoulder season
            risk_modifiers += 0.08
        else:
            risk_modifiers += 0.03
    
    # Central Valley (Sacramento, Fresno, etc.)
    if is_central_valley:
        if month in [6, 7, 8, 9]:  # Hot, dry summers
            risk_modifiers += 0.08
        elif month in [10, 5]:  # Shoulder season
            risk_modifiers += 0.04
        else:
            risk_modifiers -= 0.02  # Winter tends to be foggy with limited fire risk
    
    # Central Coast (Monterey, SLO, etc.)
    if is_central_coast:
        if month in [8, 9, 10]:  # Late summer/fall when coastal influence weakens
            risk_modifiers += 0.05
        else:
            risk_modifiers -= 0.08  # Strong marine influence most of the year
    
    # Bay Area (SF, Oakland, San Jose, etc.)
    if is_bay_area:
        if month in [9, 10]:  # Diablo winds season
            risk_modifiers += 0.10
            if 'oakland' in location_lower or 'berkeley' in location_lower:
                risk_modifiers += 0.02  # Extra risk in East Bay hills
        elif month in [7, 8]:  # Summer
            risk_modifiers += 0.02
        else:
            risk_modifiers -= 0.07  # Marine influence reduces risk most of the year
    
    # Northern Coastal (Mendocino, Humboldt, etc.)
    if is_northern_coastal:
        if month in [7, 8, 9]:  # Late summer
            risk_modifiers += 0.03
        else:
            risk_modifiers -= 0.10  # Heavy rainfall most of the year
    
    # Northern Inland (Shasta, Modoc, etc.)
    if is_northern_inland:
        if month in [6, 7, 8, 9]:  # Summer fire season
            risk_modifiers += 0.12
        elif month in [5, 10]:  # Shoulder season
            risk_modifiers += 0.05
        else:
            risk_modifiers -= 0.05  # Winter conditions reduce risk
    
    # Mountain/Sierra (Tahoe, Sierra, etc.)
    if is_mountain_sierra:
        if month in [7, 8, 9]:  # Mountain fire season
            risk_modifiers += 0.15
        elif month in [6, 10]:  # Shoulder season
            risk_modifiers += 0.07
        elif month in [12, 1, 2, 3]:  # Snow cover
            risk_modifiers -= 0.15
        else:
            risk_modifiers -= 0.05
    
    # Eastern Sierra (Mono, Inyo, etc.)
    if is_eastern_sierra:
        if month in [6, 7, 8, 9]:  # Dry season
            risk_modifiers += 0.10
        elif month in [5, 10]:  # Shoulder season
            risk_modifiers += 0.03
        else:
            risk_modifiers -= 0.07  # Winter conditions
    
    # If no specific region matched, use a default California model
    if not any([is_southern_coastal, is_southern_inland, is_central_valley, 
                is_central_coast, is_bay_area, is_northern_coastal, 
                is_northern_inland, is_mountain_sierra, is_eastern_sierra]):
        # Generic California seasonal pattern
        if month in [7, 8, 9]:  # Peak fire season 
            risk_modifiers += 0.10
        elif month in [5, 6, 10, 11]:  # Shoulder season
            risk_modifiers += 0.05
        else:  # Winter
            risk_modifiers -= 0.05
    
    # Get weather and drought data
    weather_data = None
    drought_data = None
    
    days_until_target = (target_date - datetime.now()).days
    if days_until_target <= 5:
        try:
            weather_data = get_weather_forecast(location)
        except Exception as e:
            print(f"Weather forecast error: {e}")
    
    try:
        drought_data = get_drought_data(location)
    except Exception as e:
        print(f"Drought data error: {e}")
    
    # Apply weather modifiers (additive)
    if weather_data:
        # High temperature increases risk
        if weather_data["avg_temp"] > 95:
            risk_modifiers += 0.15
        elif weather_data["avg_temp"] > 90:
            risk_modifiers += 0.10
        elif weather_data["avg_temp"] > 85:
            risk_modifiers += 0.05
        
        # Low humidity increases risk
        if weather_data["avg_humidity"] < 20:
            risk_modifiers += 0.15
        elif weather_data["avg_humidity"] < 30:
            risk_modifiers += 0.08
        elif weather_data["avg_humidity"] < 40:
            risk_modifiers += 0.04
        
        # High wind increases risk
        if weather_data["avg_wind"] > 15:
            risk_modifiers += 0.10
        elif weather_data["avg_wind"] > 10:
            risk_modifiers += 0.05
        
        # Recent precipitation decreases risk
        if weather_data["total_precip"] > 1.0:
            risk_modifiers -= 0.15
        elif weather_data["total_precip"] > 0.5:
            risk_modifiers -= 0.10
        elif weather_data["total_precip"] > 0.1:
            risk_modifiers -= 0.05
        
        prediction["weather_data"] = {
            "temperature": f"{weather_data['avg_temp']:.1f}°F",
            "humidity": f"{weather_data['avg_humidity']:.1f}%",
            "wind": f"{weather_data['avg_wind']:.1f} mph"
        }
    
    # Apply drought modifiers (additive)
    if drought_data:
        # Calculate drought impact as percentage points rather than multiplier
        drought_level = drought_data["drought_level"]
        
        if drought_level == 0:  # No drought
            drought_impact = 0.0
        elif drought_level == 1:  # Abnormally Dry
            drought_impact = 0.05
        elif drought_level == 2:  # Moderate Drought
            drought_impact = 0.10
        elif drought_level == 3:  # Severe Drought
            drought_impact = 0.15
        elif drought_level == 4:  # Extreme Drought
            drought_impact = 0.20
        else:  # Exceptional Drought
            drought_impact = 0.25
        
        risk_modifiers += drought_impact
        
        prediction["drought_data"] = {
            "level": drought_data["description"],
            "impact": f"+{drought_impact*100:.0f}% increased risk"
        }
    
    # Calculate final probability with a soft cap approach
    # As we approach 90%, additional modifiers have diminishing returns
    final_probability = base_probability + risk_modifiers
    
    # Soft cap function: as probability approaches 1, it gets harder to increase further
    if final_probability > 0.7:
        # Apply diminishing returns using a dampening function
        excess = final_probability - 0.7
        dampened_excess = excess * (1 - excess/3)  # Diminishing returns
        final_probability = 0.7 + dampened_excess
    
    # Ensure probability stays within reasonable bounds
    final_probability = max(0.05, min(0.95, final_probability))
    
    # Update prediction with final values
    prediction['avg_probability'] = final_probability
    prediction['probability'] = final_probability  # Alternate key for compatibility
    
    # Update high risk percentage (correlated with probability)
    prediction['high_risk_percentage'] = final_probability * 100
    
    # Set risk category based on final probability
    if final_probability >= 0.7:
        prediction['risk_category'] = 'High'
    elif final_probability >= 0.4:
        prediction['risk_category'] = 'Moderate'
    elif final_probability >= 0.2:
        prediction['risk_category'] = 'Low-Moderate'
    else:
        prediction['risk_category'] = 'Low'
    
    # Update risk distribution based on final category
    # Shift distribution weights toward the final category
    if prediction['risk_category'] == 'High':
        prediction['risk_distribution']['Low'] = max(0, prediction['risk_distribution']['Low'] - 1)
        prediction['risk_distribution']['High'] += 1
        if final_probability > 0.8:
            prediction['risk_distribution']['Extreme'] += 1
    elif prediction['risk_category'] == 'Low':
        prediction['risk_distribution']['High'] = max(0, prediction['risk_distribution']['High'] - 1)
        prediction['risk_distribution']['Low'] += 1
    
    return prediction

def call_llm(context, user_query, include_prediction=False, prediction_data=None):
    """
    Call the LLM with context and user query.
    
    Args:
        context (str): Retrieved context
        user_query (str): User query
        include_prediction (bool): Whether to include prediction data
        prediction_data (dict): Prediction data if available
    
    Returns:
        str: LLM response
    """
    prediction_section = ""
    
    if include_prediction and prediction_data:
        # Format the prediction data for inclusion in the prompt
        prediction_section = f"""
FUTURE WILDFIRE PREDICTION:
- Location: {prediction_data.get('location', prediction_data.get('region', 'California'))}
- Date: {prediction_data.get('date', 'Future')}
- Wildfire Risk Probability: {prediction_data.get('probability', prediction_data.get('avg_probability', 0))*100:.1f}%
- Risk Category: {prediction_data.get('risk_category', 'Unknown')}
- High Risk Areas: {prediction_data.get('high_risk_percentage', 0):.1f}% of the region

This is an AI-generated prediction based on historical patterns and environmental factors.
"""
    
    prompt = f"""
You are a specialized AI assistant for predicting and analyzing California wildfires.

Use the following context to answer the user's question.

Context:
{context}

{prediction_section}

Instructions:
- If the context has enough information, answer normally based on the data.
- If the user is asking about future predictions and prediction data is provided, emphasize that information.
- If the context and prediction data does not fully answer, you must predict or estimate based on your general knowledge.
- Always indicate clearly when you are predicting ("This is a prediction based on AI estimation.")
- Summarize any predictions in a structured table format with THESE EXACT COLUMNS:

| Location | Latitude | Longitude | Predicted Start Date | Risk Level | Estimated Probability |

IMPORTANT: The Latitude and Longitude columns MUST contain numerical coordinates, NOT text descriptions.

For example, this is the CORRECT format:
| San Jose | 37.3382 | -121.8863 | 2026-07-15 | Moderate | 67.0% |

This is INCORRECT and will break the map functionality:
| San Jose | 37.3382 | San Jose | 2026-07-15 | Moderate | 67.0% |

Here are ACCURATE coordinates for key California locations:
- San Francisco: 37.7749, -122.4194
- Los Angeles: 34.0522, -118.2437
- San Diego: 32.7157, -117.1611
- Sacramento: 38.5816, -121.4944
- San Jose: 37.3382, -121.8863
- Fresno: 36.7378, -119.7871
- Oakland: 37.8044, -122.2711
- San Mateo: 37.5630, -122.3255
- Redwood City: 37.4852, -122.2364
- Palo Alto: 37.4419, -122.1430
- Mountain View: 37.3861, -122.0839
- Santa Cruz: 36.9741, -122.0308
- Monterey: 36.2400, -121.3100
- Bakersfield: 35.3733, -119.0187
- Santa Barbara: 34.4208, -119.6982
- San Luis Obispo: 35.2828, -120.6596
- Riverside: 33.9533, -117.3961
- Anaheim: 33.8366, -117.9143
- Irvine: 33.6846, -117.8265
- Bay Area: 37.8272, -122.2913
- Central Valley: 36.7478, -119.7726
- Sierra Nevada: 38.9826, -120.5826
- Southern California: 34.0522, -118.2437
- Northern California: 39.8283, -121.4221
- Central California: 36.7396, -119.7844

- The Risk Level should be one of: Low, Low-Moderate, Moderate, High, or Extreme
- The Estimated Probability should be a percentage value
- The Predicted Start Date should be in YYYY-MM-DD format
- Do not invent external links or make up fake official sources.
- Be clear, transparent, and ethical.
- If providing a prediction for a specific time period (month/season), clearly state this is for that time period.

User Question:
{user_query}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_SERVER_URL, json=payload)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error calling local model: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception calling local model: {e}"

def try_load_faiss():
    """Try to load FAISS and metadata, return empty string if it fails."""
    global _faiss_index, _metadata, _embedding_model
    
    try:
        import faiss
        import joblib
        from sentence_transformers import SentenceTransformer
        
        MODEL_DIR = os.path.join(BASE_DIR, 'models')
        FAISS_STORE_PATH = os.path.join(BASE_DIR, 'data', 'faiss_store')
        FAISS_INDEX_FILE = os.path.join(FAISS_STORE_PATH, 'faiss_index.idx')
        METADATA_FILE = os.path.join(FAISS_STORE_PATH, 'metadata.csv')
        
        # Only load once
        if _faiss_index is None:
            _faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            _metadata = pd.read_csv(METADATA_FILE)
            _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Loaded FAISS and SentenceTransformer successfully")
            
        return True
    except Exception as e:
        print(f"Error loading FAISS components: {e}")
        return False

def perform_retrieval(query, top_k=5):
    """Perform retrieval using FAISS if available."""
    # Try to load FAISS components
    if not try_load_faiss():
        # If FAISS failed to load, return empty context
        return ""
    
    try:
        # Embed query
        query_vector = _embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = _faiss_index.search(query_vector, top_k)
        indices = indices[0]
        
        # Prepare context
        texts = []
        for idx in indices:
            if idx < len(_metadata):
                texts.append(str(_metadata.iloc[idx]['text']))
        
        return "\n\n".join(texts)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return ""

def handle_user_query(user_query, top_k=5):
    """
    Process user query and generate response.
    
    Args:
        user_query (str): User question
        top_k (int): Number of documents to retrieve
    
    Returns:
        str: Response to user
    """
    # Check if it's a prediction query
    is_prediction = is_prediction_query(user_query)
    
    # Extract location and time information if it's a prediction query
    location = None
    time_info = None
    prediction_data = None
    
    if is_prediction:
        # Extract location
        location = extract_location_from_query(user_query)
        
        # Extract time information
        time_info = extract_time_from_query(user_query)
        
        if location:
            print(f"Getting predictions for {location} at time {time_info}...")
            prediction_data = get_predictions_for_location(location, time_info)
    
    # Perform retrieval if possible, otherwise use empty context
    context = perform_retrieval(user_query, top_k)
    
    # If no context was retrieved, provide a fallback message
    if not context:
        context = "I have information about California wildfires including historical data and can make predictions based on seasonal patterns and environmental factors."
    
    # Call the LLM with or without prediction data
    response = call_llm(
        context, 
        user_query, 
        include_prediction=is_prediction,
        prediction_data=prediction_data
    )
    
    return response

if __name__ == "__main__":
    # Test the function
    test_query = "What's the wildfire risk for Los Angeles in July 2025?"
    print(f"Test query: {test_query}")
    response = handle_user_query(test_query)
    print(response)