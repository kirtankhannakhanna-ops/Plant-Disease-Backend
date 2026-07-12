import streamlit as st
import os
import io
import requests
import numpy as np
import pandas as pd
import datetime
from gtts import gTTS
from PIL import Image, ImageOps
from streamlit_js_eval import streamlit_js_eval
import tensorflow as tf
tflite = tf.lite
    

# 1. Absolute Mobile Viewport Dynamic Configuration
st.set_page_config(
    page_title="Kissan Dost AI", 
    page_icon="🌱", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Responsive Layout CSS Overrides
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f9fbf9 !important;
        color: #111111 !important;
    }
    .block-container {
        padding-top: 15px !important;
        padding-bottom: 15px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
        max-width: 100% !important;
    }
    .header-container {
        background: linear-gradient(135deg, #1e5128, #4e9f3d);
        padding: 18px 10px;
        border-radius: 12px;
        color: white !important;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .header-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 !important;
    }
    .header-subtitle {
        font-size: 13px !important;
        color: #e8f5e9 !important;
        margin-top: 4px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f4f1 !important;
        padding: 4px;
        border-radius: 15px;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px !important;
        background-color: transparent !important;
        border-radius: 12px;
        font-weight: 700;
        color: #1e5128 !important;
        font-size: 12px !important;
        flex-grow: 1;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e5128 !important;
        color: white !important;
        box-shadow: 0 2px 6px rgba(30,81,40,0.15);
    }
    .result-card {
        background-color: #ffffff !important;
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #1e5128;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-top: 10px;
        color: #111111 !important;
    }
    .result-card p, .result-card h3, .result-card h4, .result-card b {
        color: #111111 !important;
    }
    .stButton>button {
        background: linear-gradient(180deg, #4e9f3d, #1e5128) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 0 #113318 !important;
        width: 100%;
        font-size: 15px !important;
        margin-top: 12px;
    }
    .stButton>button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 0 0 transparent !important;
    }
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
        color: #111111 !important;
        background-color: #ffffff !important;
    }
    label p {
        color: #111111 !important;
        font-weight: 700;
        font-size: 14px !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #111111 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Device-Specific Session Memory
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
    
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False

def log_scan(plant, disease, severity):
    st.session_state.scan_history.insert(0, {
        "Plant": plant,
        "Disease Identified": disease,
        "Severity Level": severity,
        "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# Naya TFLite Model Loading Function
@st.cache_resource
def load_plant_model():
    model_path = "./models/model.tflite"
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    model = load_plant_model()
except Exception as e:
    pass

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

translations = {
    "English": {
        "title": "Kissan Dost AI", "sub": "Smart Multilingual Agri-Smart Suite",
        "select_lang": "Choose Language / زبان منتخب کریں / ٻولي چونڊيو / ਭਾਸ਼ਾ ਚੁਣੋ",
        "upload_lbl": "Upload plant leaf image from Gallery:", "camera_lbl": "Capture Live Photo using Camera:",
        "btn_predict": "Analyze Plant Image", "mode_lbl": "Select Input Method:",
        "cam_opt": "📸 Use Live Camera", "gal_opt": "📁 Upload from Gallery",
        "leaf_lbl": "🌿 Plant/Leaf Type:", "dis_lbl": "🔍 Detected Disease:",
        "rem_lbl": "📋 Recommended Remedy:", "sev_lbl": "📊 Disease Severity Status:",
        "weather_tab": "🌦️ Weather", "npk_tab": "🧪 Fertilizer", "ai_tab": "🔬 AI Detector", "dash_tab": "📊 Analytics",
        "acre_lbl": "Enter Land Area (in Acres):", "crop_lbl": "Select Crop Type:", "calc_btn": "Calculate Fertilizer Needed",
        "open_cam_btn": "📸 Tap to Open Camera", "close_cam_btn": "❌ Close Camera"
    },
    "Urdu": {
        "title": "کسان دوست AI", "sub": "پودوں کی بیماریوں کا شناختی نظام اور اسمارٹ زرعی ایپ",
        "select_lang": "Choose Language / زبان منتخب کریں / ٻولي چونڊيو / ਭਾਸ਼ਾ ਚੁਣੋ",
        "upload_lbl": "گیلری سے پودے کے پتے کی تصویر اپلوڈ کریں:", "camera_lbl": "کیمرے سے لائیو تصویر کھینچیں:",
        "btn_predict": "بیماری کی تشخیص کریں", "mode_lbl": "طریقہ منتخب کریں:",
        "cam_opt": "📸 لائیو کیمرہ استعمال کریں", "gal_opt": "📁 گیلری سے اپلوڈ کریں",
        "leaf_lbl": "🌿 پودے / پتے کی قسم:", "dis_lbl": "🔍 تشخیص شدہ بیماری:",
        "rem_lbl": "📋 تجویز کردہ علاج:", "sev_lbl": "📊 بیماری کی شدت کی صورتحال:",
        "weather_tab": "🌦️ موسم", "npk_tab": "🧪 کھاد", "ai_tab": "🔬 تشخیص", "dash_tab": "📊 اینالیٹکس",
        "acre_lbl": "زمین کا رقبہ (ایکڑ میں) لکھیں:", "crop_lbl": "فصل کی قسم منتخب کریں:", "calc_btn": "کھاد کی مقدار معلوم کریں",
        "open_cam_btn": "📸 کیمرہ آن کرنے کے لیے کلک کریں", "close_cam_btn": "❌ کیمرہ بند کریں"
    },
    "Sindhi": {
        "title": "ڪسان دوست AI", "sub": "ٻوٽن جي بيمارين جي سڃاڻپ ۽ سمارٽ زرعي نظام",
        "select_lang": "Choose Language / زبان منتخب کریں / ٻولي چونڊيو / ਭਾਸ਼ਾ ਚੁਣੋ",
        "upload_lbl": "گيلري مان ٻوٽي جي پن جي تصوير اپلوڊ ڪريو:", "camera_lbl": "ڪيمرا سان لائيو تصوير ڪڍو:",
        "btn_predict": "بيماري جي تشخيص ڪريو", "mode_lbl": "طريقو چونڊيو:",
        "cam_opt": "📸 لائيو ڪيمرا استعمال ڪريو", "gal_opt": "📁 گيلري مان اپلوڊ ڪريو",
        "leaf_lbl": "🌿 ٻوٽي / پن جو قسم:", "dis_lbl": "🔍 تشخيص ٿيل بيماري:",
        "rem_lbl": "📋 تجويز ڪيل علاج:", "sev_lbl": "📊 بيماري جي شدت جي صورتحال:",
        "weather_tab": "🌦️ موسم", "npk_tab": "🧪 ڀاڻ", "ai_tab": "🔬 تشخيص", "dash_tab": "📊 ايناليٽڪس",
        "acre_lbl": "زمين جو رقبو (ايڪڙ ۾) لکو:", "crop_lbl": "فصل جو قسم چونڊيو:", "calc_btn": "ڀاڻ جي مقدار معلوم ڪريو",
        "open_cam_btn": "📸 ڪيمرا آن ڪرڻ لاءِ ڪلڪ ڪريو", "close_cam_btn": "❌ ڪيمرا بند ڪريو"
    },
    "Punjabi": {
        "title": "ਕਿਸਾਨ ਦੋਸਤ AI", "sub": "ਪੌਦਿਆਂ ਦੀਆਂ ਬਿਮਾਰੀਆਂ ਦੀ ਪਛਾਣ ਅਤੇ ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ ਐਪ",
        "select_lang": "Choose Language / زبان منتخب کریں / ٻولي چونڊيو / ਭਾਸ਼ਾ ਚੁਣੋ",
        "upload_lbl": "ਗੈਲਰੀ ਤੋਂ ਪੌਦੇ ਦੇ ਪੱਤੇ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ:", "camera_lbl": "ਕੈਮਰੇ ਤੋਂ ਲਾਈਵ ਫੋਟੋ ਖਿੱਚੋ:",
        "btn_predict": "ਬਿਮਾਰੀ ਦੀ ਜਾਂਚ ਕਰੋ", "mode_lbl": "ਤਰੀਕਾ ਚੁਣੋ:",
        "cam_opt": "📸 ਲਾਈਵ ਕੈਮਰਾ ਵਰਤੋ", "gal_opt": "📁 ਗੈਲਰੀ ਤੋਂ ਅਪਲੋਡ ਕਰੋ",
        "leaf_lbl": "🌿 ਪੌਦੇ / ਪੱਤੇ ਦੀ ਕਿਸਮ:", "dis_lbl": "🔍 ਪਛਾਣ ਕੀਤੀ ਬਿਮਾਰੀ:",
        "rem_lbl": "📋 ਸੁਝਾਇਆ ਗਿਆ ਇਲਾਜ:", "sev_lbl": "📊 ਬਿਮਾਰੀ ਦੀ ਗੰਭੀਰਤਾ ਦੀ ਸਥਿਤੀ:",
        "weather_tab": "🌦️ ਮੌਸਮ", "npk_tab": "🧪 ਖਾਦ", "ai_tab": "🔬 ਜਾਂਚ", "dash_tab": "📊 ਐਨਾਲਿਟਿਕਸ",
        "acre_lbl": "ਜ਼ਮੀਨ ਦਾ ਰਕਬਾ (ਏਕੜ ਵਿੱਚ) ਲਿਖੋ:", "crop_lbl": "ਫਸਲ ਦੀ ਕਿਸਮ ਚੁਣੋ:", "calc_btn": "ਖਾਦ ਦੀ ਮਾਤਰਾ ਪਤਾ ਕਰੋ",
        "open_cam_btn": "📸 ਕੈਮਰਾ ਚਾਲੂ ਕਰਨ ਲਈ ਕਲਿੱਕ ਕਰੋ", "close_cam_btn": "❌ ਕੈਮਰਾ ਬੰਦ ਕਰੋ"
    }
}

lang = st.selectbox(translations["English"]["select_lang"], ["English", "Urdu", "Sindhi", "Punjabi"])
t = translations[lang]

# Premium Styled Colorful Brand Banner
st.markdown(f"""
    <div class="header-container">
        <div class="header-title">{t['title']}</div>
        <div class="header-subtitle">{t['sub']}</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([t["ai_tab"], t["weather_tab"], t["npk_tab"], t["dash_tab"]])

def split_raw_name(raw_name):
    if "___" in raw_name:
        parts = raw_name.split("___")
        plant = parts[0].replace("_", " ").replace(" (maize)", "")
        disease = parts[1].replace("_", " ")
    else:
        plant = "Unknown Plant"
        disease = raw_name.replace("_", " ")
    return plant, disease

def estimate_severity(img_source):
    img = Image.open(img_source).convert("RGB")
    gray = ImageOps.grayscale(img)
    gray_np = np.array(gray)
    damaged_pixels = np.sum(gray_np < 100)
    total_pixels = gray_np.size
    severity_pct = int((damaged_pixels / total_pixels) * 100)
    if severity_pct < 10: severity_pct = 15
    if severity_pct > 90: severity_pct = 85
    return severity_pct

def get_disease_details(disease_name, language):
    plant_en, disease_en = split_raw_name(disease_name)
    if "Apple_scab" in disease_name:
        if language == "Urdu": return {"name": "ایپل اسکیب (پپڑی روگ)", "remedy": "فصل کے آغاز میں تانبے پر مبنی فنگسائڈز کا چھڑکاؤ کریں۔"}
        if language == "Sindhi": return {"name": "ایپل اسڪيب", "remedy": "فصل جي شروعات ۾ تانبي واري فنگسائڊ جو ڇڙڪاءُ ڪريو."}
        if language == "Punjabi": return {"name": "ਐਪਲ ਸਕੈਬ", "remedy": "ਫਸਲ ਦੇ ਸ਼ੁਰੂ ਵਿੱਚ ਤਾਂਬੇ ਦੇ ਫੰਗਸਾਈਡ ਦਾ ਛਿੜਕਾਅ ਕਰੋ।"}
        return {"name": "Apple Scab", "remedy": "Apply copper-based fungicides early. Remove fallen leaves."}
    
    if language == "Urdu": return {"name": f"{disease_en}", "remedy": f"متاثرہ حصے کو {plant_en} سے الگ کریں اور اسپرے کریں۔"}
    if language == "Sindhi": return {"name": f"{disease_en}", "remedy": f"متاثر ٿيل حصي کي {plant_en} کان الڳ ڪريو ۽ اسپري ڪريو."}
    if language == "Punjabi": return {"name": f"{disease_en}", "remedy": f"ਪ੍ਰਭਾਵਿਤ ਹਿੱਸੇ ਨੂੰ {plant_en} ਤੋਂ ਵੱਖ ਕਰੋ اور ਸਪ੍ਰੇ ਕਰੋ।"}
    return {"name": f"{disease_en}", "remedy": f"Remove infected parts of {plant_en} and apply standard crop care."}

def generate_voice_output(plant_text, disease_text, language):
    code = "en"
    if language in ["Urdu", "Sindhi", "Punjabi"]: code = "ur"
    full_speech = f"Plant type is {plant_text}. Detected condition is {disease_text}."
    try:
        tts = gTTS(text=full_speech, lang=code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp.getvalue(), format="audio/mp3")
    except:
        pass

# ==========================================
# TAB 1: AI DETECTOR
# ==========================================
with tab1:
    input_mode = st.radio(t["mode_lbl"], [t["cam_opt"], t["gal_opt"]], horizontal=True)
    final_image_source = None

    if input_mode == t["cam_opt"]:
        if not st.session_state.camera_enabled:
            if st.button(t["open_cam_btn"]):
                st.session_state.camera_enabled = True
                st.rerun()
        else:
            final_image_source = st.camera_input(t["camera_lbl"])
            if st.button(t["close_cam_btn"]):
                st.session_state.camera_enabled = False
                st.rerun()
    else:
        st.session_state.camera_enabled = False
        final_image_source = st.file_uploader(t["upload_lbl"], type=["jpg", "jpeg", "png"])

    if final_image_source is not None:
        st.image(final_image_source, use_container_width=True)
        
        if st.button(t["btn_predict"]):
            with st.spinner("Processing Pattern Engine..."):
                image = Image.open(final_image_source).convert("RGB")
                image = image.resize((128, 128))
                
                # Naya TFLite Prediction Logic
                img_array = np.array(image) / 255.0  
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32) 
                
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                
                model.set_tensor(input_details[0]['index'], img_array)
                model.invoke()
                
                predictions = model.get_tensor(output_details[0]['index'])
                predicted_class_idx = np.argmax(predictions[0])
                detected_raw_name = CLASS_NAMES[predicted_class_idx]
                
                plant_en, disease_split_en = split_raw_name(detected_raw_name)
                details = get_disease_details(detected_raw_name, lang)
                pct = estimate_severity(final_image_source)
                if "healthy" in detected_raw_name.lower(): pct = 0
                
                log_scan(plant_en, details['name'], pct)
                    
                st.markdown(f"""
                <div class="result-card">
                    <h3 style='color:#1e5128; margin-top:0; font-weight:700;'>📊 Diagnostic Report</h3>
                    <p><b>{t['leaf_lbl']}</b> {plant_en}</p>
                    <p><b>{t['dis_lbl']}</b> {details['name']}</p>
                    <p><b>{t['sev_lbl']}</b> {pct}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(pct / 100)
                    
                st.markdown(f"""
                <div class="result-card">
                    <h4 style='color:#1e5128; margin-top:0; font-weight:700;'>{t['rem_lbl']}</h4>
                    <p>{details['remedy']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                generate_voice_output(plant_en, details['name'], lang)

# ==========================================
# TAB 2: LIVE WEATHER (100% AUTOMATIC EXACT LOCATION)
# ==========================================
with tab2:
    st.subheader("🌦️ Live Device Weather Tracking")
    
    # Optional Manual Search fallback (hidden unless needed)
    city_input = st.text_input("🌍 Search City (Optional):", placeholder="App will auto-detect, but you can type here if needed...")
    
    lat, lon, final_city_name = None, None, None
    
    if city_input:
        # Manual Mode
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_input}&count=1&language=en&format=json"
        try:
            geo_data = requests.get(geo_url, timeout=5).json()
            if "results" in geo_data and len(geo_data["results"]) > 0:
                lat = geo_data["results"][0]["latitude"]
                lon = geo_data["results"][0]["longitude"]
                final_city_name = geo_data["results"][0]["name"]
            else:
                st.error("City not found. Please check spelling.")
        except:
            st.error("Geocoding service unavailable.")
    else:
        # 100% AUTOMATIC MODE (Starts Automatically)
        location = streamlit_js_eval(data_of='getLocation', key='get_user_gps_coords')
        
        if location:
            lat = location['coords']['latitude']
            lon = location['coords']['longitude']
            
            # REVERSE GEOCODING AI: Convert Coordinates to exact town/village name!
            try:
                rev_url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
                rev_data = requests.get(rev_url, timeout=5).json()
                # Fetches exact local town name
                final_city_name = rev_data.get("locality", rev_data.get("city", "Auto-Detected Exact GPS Location"))
            except:
                final_city_name = "Auto-Detected Exact GPS Location"
        else:
            # High-Performance Automatic Network Backup
            try:
                ip_data = requests.get("https://ipapi.co/json/", timeout=3).json()
                lat = ip_data.get("latitude")
                lon = ip_data.get("longitude")
                final_city_name = ip_data.get("city", "Auto-Detected Network Location")
            except:
                pass

    # Fetch and Display Weather Data
    if lat and lon:
        try:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            response = requests.get(weather_url, timeout=5).json()
            
            if "current_weather" in response:
                current = response["current_weather"]
                temp = current["temperature"]
                humidity_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m"
                hum_resp = requests.get(humidity_url, timeout=5).json()
                humidity = hum_resp.get("current", {}).get("relative_humidity_2m", 65)
                
                st.markdown(f"""
                <div class="result-card">
                    📍 Live Auto-Location Synchronized:<br>
                    <b style="font-size: 20px;">{final_city_name}</b>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1: st.metric(label="Temperature", value=f"{int(temp)}°C")
                with col2: st.metric(label="Humidity", value=f"{humidity}%")
            else:
                st.error("Meteo engine connection failed.")
        except Exception as e:
            st.error("Network synchronization timeout.")

# ==========================================
# TAB 3: NPK FERTILIZER CALCULATOR
# ==========================================
with tab3:
    st.subheader(t["npk_tab"])
    acres = st.number_input(t["acre_lbl"], min_value=1.0, max_value=500.0, value=1.0)
    crop = st.selectbox(t["crop_lbl"], ["Wheat (ਗੰਮ / ਗੰਦਮ)", "Cotton (ਕਪਾਸ / کپاس)", "Rice (ਚੌਲ / چاول)", "Potato (ਆਲੂ / ਆਲੂ)"])
    if st.button(t["calc_btn"]):
        crop_factors = {"Wheat (ਗੰਮ / ਗੰਦਮ)": (50, 30, 20), "Cotton (ਕਪਾਸ / کپاس)": (60, 40, 30), "Rice (ਚੌਲ / چاول)": (45, 25, 25), "Potato (ਆਲੂ / ਆਲੂ)": (70, 50, 60)}
        n, p, k = crop_factors[crop]
        
        st.markdown(f"""
        <div class="result-card">
            <h4 style='color:#1e5128; margin-top:0; font-weight:700;'>📋 Fertilizer Bags Plan Required:</h4>
            <p>🧪 <b>Urea (Nitrogen):</b> {round((n * acres) / 46, 1)} Bags</p>
            <p>🧪 <b>DAP (Phosphorus):</b> {round((p * acres) / 46, 1)} Bags</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 4: REALTIME ANALYTICS DASHBOARD
# ==========================================
with tab4:
    st.subheader("📊 Farm Health Analytics Dashboard")
    
    if st.button("🗑️ Clear My History", key="del_btn"):
        st.session_state.scan_history = []
        st.rerun()

    if st.session_state.scan_history:
        df = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(df, use_container_width=True)
        
        diseased_df = df[df["Severity Level"] > 0]
        most_diseased = diseased_df["Plant"].mode()[0] if not diseased_df.empty else "None"
            
        healthy_df = df[df["Severity Level"] == 0]
        healthiest = healthy_df["Plant"].mode()[0] if not healthy_df.empty else "None"
            
        valid_severities = [s for s in df["Severity Level"].tolist() if s > 0]
        avg_sev = int(np.mean(valid_severities)) if valid_severities else 0
        
        st.markdown(f"""
        <div class="result-card" style="margin-top: 20px;">
            <h4 style='color:#1e5128; margin-top:0; font-weight:700;'>📈 Plant Health Summary</h4>
            <p>🦠 <b>Most Diseased Plant:</b> {most_diseased}</p>
            <p>🌿 <b>Healthiest Plant:</b> {healthiest}</p>
            <p>📊 <b>Average Infection Severity:</b> {avg_sev}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No scans logged yet. Use the AI Detector to start your analysis!")