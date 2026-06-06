import streamlit as st
from bootstrap import add_project_src_to_path

add_project_src_to_path()
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from retail_shelf_ai.streamlit_utils import load_detection_model, load_classification_model, run_fast_pipeline

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="RetailVision AI | Enterprise",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {background-color: #0E1117;}
    
    /* تحسين العناوين */
    h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #FFFFFF;}
    h3 {color: #A0A0A0; font-weight: 400;}
    
    /* بطاقات الإحصائيات */
    .metric-card {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-value {font-size: 28px; font-weight: bold; color: #F3F4F6;}
    .metric-label {font-size: 14px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;}
    
    /* زر التحميل */
    div.stDownloadButton > button {
        width: 100%;
        background-color: #10B981;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "bottle_drink", "Hand_Soap_Pump_Bottles", "Dark_OTC_and_Hygiene_Mix", "Diaper_Packs", 
    "Refill_and_Soft_Pouches", "shampo", "cream", "feet_care", "Men_Grooming_Dark", 
    "Medicine_Soap_Detergent_Packs", "Powder_and_Supplement_Canisters", 
    "Coffee_and_Breakfast_Packs", "hand_care", "Red_Box_Mix", "vitamine", 
    "Manual_Dishwashing_Liquid", "Red_Orange_Grooming", "perfumes", "Beechams_Cold_Flu", 
    "Green_Pharmacy_and_Herbal", "shampo_hair", "tomato paste", 
    "Mixed_Household_Grocery_Packs", "Feminine_Hygiene_Baskets", "Detergent_and_Cleaning", 
    "Feminine_Hygiene_Dark", "Solid_Bar_Soap", "Scented_Candles_and_Diffusers", 
    "CocaCola_Cans", "Bottles_Jars_Mix", "Pharmacy_Box", "Household_Pharmacy_Mix", 
    "Soda_Cartons_Mixed", "Coffee_Canisters_Mix", "energy_drink", "Body_Care_General", 
    "Large_Plastic_Soda_Bottles", "Body_Care_Mix", "Shampoo_Body_Wash_Mix", 
    "Body_Lotions_and_Creams", "Dove_and_White_Body_Wash", "Body_Care_Light_Bottles", 
    "Colorful_Household_Mix", "Food_Condiment_Jars", "Liquid_Soap_and_Body_Wash", 
    "pepsi", "Household_Misc_Containers", "Folgers_Coffee_Canisters", 
    "Hair_Dye_and_Cosmetics", "girl_game", "Green_Soda_Packs", "OTC_Medicine_General", 
    "Coffee_Creamer_and_Spreads", "Soft_Hygiene_Packs", "OTC_Pain_and_Digestive_Relief", 
    "Energy_Drink_Cans", "Tubs_and_Jars", "Dark_Personal_Care_Bottles", "Blue_Green_Boxes", 
    "Deodorant_Sticks", "Lotion_and_Cream_Tubes", "Personal_Care_Mix", 
    "White_OTC_Medicine_Boxes", "Foot_Care_and_Gadgets", "Dark_Soda_Bottles", 
    "General_Pharmacy", "coffee_box", "Pantry_Condiment_Jars", "Beverage_Bottles", 
    "creem", "Facial_Skincare_Tubes", "Soda_Bottles_Mixed", "Dish_and_Hand_Soap", 
    "CocaCola_Cartons", "Green_Packs_Mix", "nova", "dove_shampo", "Aerosol_Deodorants", 
    "cola_dark", "Personal_Care_Boxes", "Soda_Cartons_Colorful", "Mouthwash_and_Liquid_Soap", 
    "Blue_Box_Mix", "Cylindrical_Tubs", "Liquid_Detergent_and_Bleach", 
    "Feminine_Hygiene_Packs", "Oral_Care_Mix", "Bar_Soap_Packs", "Bagged_Coffee_Mix", 
    "Colorful_Body_Wash_and_Deo", "CocaCola_DrPepper_Cartons"
]

# ==========================================
# 2. Sidebar & Model Loading
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🎛️ Control Center</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("1. AI Configuration")
    model_choice = st.radio("Detection Engine:", ["YOLO", "RT-DETR"], horizontal=True, help="Select the architecture for dense product localization.")
    conf_val = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.45, help="Higher values reduce false positives.")
    
    st.subheader("2. Input Source")
    input_source = st.selectbox("Data Feed:", ["🖼️ Upload Image", "📷 Camera Snapshot"])
    
    st.markdown("---")
    st.caption("System Status: 🟢 Demo Ready")
    st.caption("v2.4.0")

# تحميل النماذج
det_model = load_detection_model(model_choice)
clf_model, device = load_classification_model('weights/resnet50_best.pth', 91)

# ==========================================
# 3. Header Area
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📦 RetailVision AI")
    st.markdown("### Dense Product Detection & Visual Product-Group Auditing")
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 20px;">
        <span style="background-color: #3B82F6; color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px;">Dense CV</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# 4. Main Processing Logic
# ==========================================
image = None

if input_source == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload Shelf Image (High Resolution Recommended)", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif input_source == "📷 Camera Snapshot":
    cam_img = st.camera_input("Capture Shelf Area")
    if cam_img:
        image = Image.open(cam_img)

if image:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📍 Source Feed")
        st.image(image, use_container_width=True)
    
    # زر التشغيل
    analyze_btn = st.button("🚀 Initialize Audit Scan", type="primary", use_container_width=True)

    if analyze_btn:
        with st.status("🔄 Processing Neural Pipeline...", expanded=True) as status:
            st.write(f"1. Object Localization ({model_choice})...")
            det_img_plotted, results = run_fast_pipeline(det_model, clf_model, device, image, conf_val, CLASS_NAMES)
            
            with c2:
                st.markdown(f"##### 🔍 AI Vision Overlay")
                st.image(det_img_plotted, use_container_width=True, channels="BGR")
            
            st.write("2. Optional Visual Group Classification (ResNet50)...")
            st.write("3. Aggregating Inventory Data...")
            status.update(label="Audit Complete!", state="complete", expanded=False)

        # ==========================================
        # 5. Dashboard & Analytics
        # ==========================================
        if results:
            st.divider()
            st.markdown("## 📊 Audit Report Dashboard")
            
            df = pd.DataFrame(results)
            total_items = len(df)
            unique_groups = df['label'].nunique()
            avg_conf = df['confidence'].mean()
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.markdown(f'<div class="metric-card"><div class="metric-value">{total_items}</div><div class="metric-label">Total Stock</div></div>', unsafe_allow_html=True)
            kpi2.markdown(f'<div class="metric-card"><div class="metric-value">{unique_groups}</div><div class="metric-label">Visual Groups</div></div>', unsafe_allow_html=True)
            kpi3.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">AI Confidence</div></div>', unsafe_allow_html=True)
            kpi4.markdown(f'<div class="metric-card"><div class="metric-value">{model_choice}</div><div class="metric-label">Model Used</div></div>', unsafe_allow_html=True)
            
            st.write("") 
            
            st.subheader("📍 Interactive Shelf Map")
            st.info("💡 Pro Tip: Hover over boxes for visual group details. Use mouse to zoom in/out.")
            
            fig = px.imshow(image)
            
            hover_texts = [f"<b>Group:</b> {r['label']}<br><b>Conf:</b> {r['confidence']:.1f}%" for r in results]
            
            for i, r in enumerate(results):
                box = r['bbox']
                fig.add_trace(go.Scatter(
                    x=[box[0], box[2], box[2], box[0], box[0]],
                    y=[box[1], box[1], box[3], box[3], box[1]],
                    mode='lines', 
                    fill="toself",
                    fillcolor='rgba(0,0,0,0)', 
                    line=dict(color='#00FF00', width=2),
                    name=r['label'],
                    text=hover_texts[i],
                    hoverinfo="text", 
                    hoveron="fills", 
                    showlegend=False
                ))

            fig.update_layout(
                height=650, 
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#1F2937",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            col_chart, col_data = st.columns([1.5, 1])
            
            with col_chart:
                st.subheader("📈 Stock Distribution by Visual Product Group")
                counts = df['label'].value_counts().reset_index()
                counts.columns = ['Visual Product Group', 'Quantity']
                fig_bar = px.bar(
                    counts.head(15), 
                    x='Quantity', 
                    y='Visual Product Group', 
                    orientation='h',
                    color='Quantity',
                    color_continuous_scale='Tealgrn'
                )
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_data:
                st.subheader("📋 Inventory List")
                st.dataframe(
                    df[['label', 'confidence', 'det_conf']].rename(columns={'label': 'Visual Group', 'confidence': 'Group Conf.', 'det_conf': 'Det Conf.'}),
                    use_container_width=True,
                    height=400
                )
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Audit Report (CSV)",
                    data=csv_data,
                    file_name='shelf_audit_report.csv',
                    mime='text/csv'
                )

        else:
            st.warning("⚠️ No products detected. Please adjust the 'Sensitivity Threshold' in the sidebar.")


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 12px;">
    RetailVision AI System © 2025 | Powered by PyTorch, YOLOv8m, RT-DETR & ResNet-50 | Portfolio Demo
</div>
""", unsafe_allow_html=True)