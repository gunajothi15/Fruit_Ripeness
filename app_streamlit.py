# app_streamlit.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import plotly.express as px
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\Gunajothi\runs\detect\train4\weights\best.pt"
CONFIDENCE = 0.4

CLASS_NAMES = ['raw_banana', 'raw_mango', 'ripe_banana', 'ripe_mango']

# Nutrition per 100g
NUTRITION_INFO = {
    'raw_banana': {'Calories': 89,'Carbs': 22.8,'Protein': 1.1,'Fat': 0.3,'Fiber': 2.6,'Sugar': 12},
    'ripe_banana': {'Calories': 89,'Carbs': 23,'Protein': 1.1,'Fat': 0.3,'Fiber': 2.6,'Sugar': 17},
    'raw_mango': {'Calories': 60,'Carbs': 15,'Protein': 0.8,'Fat': 0.4,'Fiber': 1.6,'Sugar': 14},
    'ripe_mango': {'Calories': 70,'Carbs': 17,'Protein': 0.8,'Fat': 0.6,'Fiber': 2,'Sugar': 15}
}

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

def draw_boxes(image_bgr, results):
    img = image_bgr.copy()
    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)
    return img

def get_all_fruits(results):
    out = []
    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                out.append((CLASS_NAMES[cls_id], conf))
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="Fruit Scanner", layout="wide")
st.title("🍌🥭 Fruit Ripeness Scanner (Per 100g Nutrition)")

with st.spinner("Loading YOLO model..."):
    model = load_model(MODEL_PATH)

# Sidebar
st.sidebar.header("Input Settings")
mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Webcam"])
conf_value = st.sidebar.slider("Detection Confidence", 0.0, 1.0, CONFIDENCE, 0.01)

# ---------------- MAIN PROCESS FUNCTION ----------------
def process_image(pil_img):

    # YOLO Prediction
    bgr = pil_to_bgr(pil_img)
    results = model.predict(source=bgr, save=False, conf=conf_value, verbose=False)

    # Annotated Image
    annotated_img = draw_boxes(bgr, results)
    annotated_pil = bgr_to_pil(annotated_img)
    st.image(annotated_pil, use_container_width=True)

    # Extract fruits
    detections = get_all_fruits(results)

    # ---------------- FRUIT COUNT ----------------
    st.subheader("🍉 Fruit Count")
    fruit_counts = {}
    for fruit, _ in detections:
        fruit_counts[fruit] = fruit_counts.get(fruit, 0) + 1

    if len(fruit_counts) == 0:
        st.warning("No fruits detected.")
        return

    # Badges
    badge_html = ""
    for fruit, count in fruit_counts.items():
        color = "#E6D227" if "ripe" in fruit else "#0DB616"
        badge_html += f"""
        <span style='
            display:inline-block;
            background:{color};
            padding:8px 14px;
            margin:4px;
            border-radius:10px;
            font-weight:600;
            color:black;
        '>{count}× {fruit.replace('_',' ').title()}</span>
        """
    st.markdown(badge_html, unsafe_allow_html=True)

    # ---------------- NUTRITION TABLE ----------------
    st.subheader("🥗 Nutrition Table (per 100g)")

    table_rows = []
    for fruit, count in fruit_counts.items():
        info = NUTRITION_INFO[fruit]
        table_rows.append({
            "Fruit": fruit.replace("_"," ").title(),
            "Calories (per 100g)": info["Calories"],
            "Carbs (g)": info["Carbs"],
            "Protein (g)": info["Protein"],
            "Fat (g)": info["Fat"],
            "Fiber (g)": info["Fiber"],
            "Sugar (g)": info["Sugar"],
            "Count": count
        })

    df = pd.DataFrame(table_rows)
    st.table(df)

    # ---------------- CHART ----------------
    st.subheader("📊 Nutrition Comparison Chart (per 100g)")
    fig = px.bar(
        df,
        x="Fruit",
        y=["Calories (per 100g)", "Carbs (g)", "Sugar (g)"],
        barmode="group",
        title="Nutritional Comparison per 100g",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MAIN APP ----------------
if mode == "Upload Image":
    file = st.file_uploader("Upload a fruit image", type=["png","jpg","jpeg"])
    if file:
        pil_img = Image.open(file).convert("RGB")
        process_image(pil_img)

else:
    cam = st.camera_input("Webcam")
    if cam:
        pil_img = Image.open(cam).convert("RGB")
        process_image(pil_img)
