import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="VisionFlow | Advanced Color Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Elegant CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #30363d;
    }

    /* Elegant Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
    }

    /* Soft Title Gradient */
    .main-title {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(#fff, #888);
        -webkit-background-clip: text;
        font-weight: 500;
        font-size: 3rem !important;
        margin-bottom: 0px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        transition: all 0.3s;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function ---
def hex_to_hsv(hex_color):
    rgb = mcolors.hex2color(hex_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    return int(hsv[0] * 179), int(hsv[1] * 255), int(hsv[2] * 255)

# --- Header Section ---
st.markdown('<h1 class="main-title">VisionFlow</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#00E613; font-size:1.1rem; margin-bottom:2rem;">High-precision real-time color tracking & analytics</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1213/1213032.png", width=80)
    st.header("Configurations")
    
    picked_color = st.color_picker("🎯 Select Target Color", "#00FF00")
    ph, ps, pv = hex_to_hsv(picked_color)

    # Auto-logic for achromatic detection (Black/White/Gray)
    is_achromatic = ps < 30 or pv < 80 
    
    with st.expander("🛠 Advanced Tuning", expanded=True):
        h_min = st.slider("Hue Range Low", 0, 179, 0 if is_achromatic else max(0, ph - 12))
        h_max = st.slider("Hue Range High", 0, 179, 179 if is_achromatic else min(179, ph + 12))
        s_min = st.slider("Saturation Min", 0, 255, 0 if is_achromatic else max(0, ps - 60))
        s_max = st.slider("Saturation Max", 0, 255, 255)
        v_min = st.slider("Value Min", 0, 255, 0 if pv < 128 else max(0, pv - 80))
        v_max = st.slider("Value Max", 0, 255, min(255, pv + 80) if pv < 128 else 255)

    lower_limit = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_limit = np.array([h_max, s_max, v_max], dtype=np.uint8)

# --- Main Interaction Logic ---
if "run" not in st.session_state: st.session_state.run = False
if "log" not in st.session_state: st.session_state.log = []

col_btn1, col_btn2, _ = st.columns([1, 1, 2])
with col_btn1:
    start_click = st.button("▶ Start Engine", use_container_width=True, type="primary")
with col_btn2:
    stop_click = st.button("🛑 Stop Engine", use_container_width=True)

if start_click: st.session_state.run = True
if stop_click: st.session_state.run = False

# Statistics Area
stat_col1, stat_col2, stat_col3 = st.columns(3)
count_metric = stat_col1.empty()
fps_metric = stat_col2.empty()
status_metric = stat_col3.empty()

# Display Area
view_container = st.container()
frame_placeholder = view_container.empty()

# --- Processing Loop ---
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = time.time()
    frame_count = 0

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        
        # Elegant Denoising
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1500:
                object_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                # Drawing elegant soft boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (76, 175, 80), 2)
                cv2.putText(frame, f"ID:{object_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Update Metrics with HTML cards
        count_metric.markdown(f'<div class="metric-card"><p style="color:#FFFFFF; margin:0;">ACTIVE OBJECTS</p><h2 style="color:#4CAF50; margin:0;">{object_count}</h2></div>', unsafe_allow_html=True)
        fps_metric.markdown(f'<div class="metric-card"><p style="color:#8b949e; margin:0;">ENGINE SPEED</p><h2 style="color:#2196F3; margin:0;">{int(fps)} FPS</h2></div>', unsafe_allow_html=True)
        status_metric.markdown(f'<div class="metric-card"><p style="color:#8b949e; margin:0;">SYSTEM STATUS</p><h2 style="color:#FF9800; margin:0;">RUNNING</h2></div>', unsafe_allow_html=True)

        # Display Frame side-by-side
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        combined = np.hstack((frame_rgb, mask_rgb))
        frame_placeholder.image(combined, channels="RGB", use_container_width=True)

        st.session_state.log.append({"frame": frame_count, "count": object_count})
        time.sleep(0.01)

    cap.release()
    status_metric.markdown('<div class="metric-card"><p style="color:#8b949e; margin:0;">SYSTEM STATUS</p><h2 style="color:#f44336; margin:0;">IDLE</h2></div>', unsafe_allow_html=True)

# --- Footer Log ---
if not st.session_state.run and st.session_state.log:
    with st.expander("📊 Session Analytics"):
        df = pd.DataFrame(st.session_state.log)
        st.line_chart(df.set_index('frame')['count'])
        st.download_button("Export Session Data", df.to_csv(index=False), "vision_log.csv")