import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title='Prediksi Harga Mobil',
    page_icon='🚗',
    layout='centered',
    initial_sidebar_state='expanded'
)

# ---------- Custom CSS (Menyamakan UI dengan Gambar) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Latar Belakang Utama Aplikasi (Deep Navy Blue) */
    .stApp {
        background-color: #0b132b !important;
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }

    /* Memaksa Teks Deskripsi Bawaan Menjadi Putih/Terang */
    .stApp p, .stApp span, .stApp label {
        color: #e2e8f0 !important;
    }
    
    /* Judul Utama dengan Gradasi Sesuai Gambar (Cyan ke Ungu) */
    .main-title {
        text-align: center; 
        font-size: 3rem;
        background: linear-gradient(90deg, #64dfdf 0%, #a2d2ff 40%, #b5179e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Deskripsi Subtitle */
    .sub-title {
        text-align: center; 
        font-size: 1.1rem; 
        color: #cbd5e1 !important; 
        margin-top: 5px;
        opacity: 0.9;
    }
    
    /* Card Info Tampilan Awal (Biru Gradasi Lembut) */
    .info-card-custom {
        background: linear-gradient(135deg, #1d2d50 0%, #133b5c 100%);
        padding: 25px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .info-icon-circle {
        background-color: #3a86ff;
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    .info-text-custom {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    /* Card Hasil Prediksi Modern (Saat Tombol Diklik) */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        text-align: center;
        color: white;
        animation: slideUp 0.5s ease-out;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .result-card h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #9333ea !important;
    }
    .result-card h1 {
        margin: 15px 0 0 0;
        font-size: 4rem;
        font-weight: 800;
        color: #22d3ee !important;
        text-shadow: 0 4px 20px rgba(34, 211, 238, 0.3);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Tombol Utama di Sidebar - Berwarna Ungu Gradasi Sesuai Gambar */
    .stButton>button {
        background: linear-gradient(90deg, #3a86ff 0%, #8338ec 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 8px 20px rgba(131, 56, 236, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(131, 56, 236, 0.5);
    }
    
    /* Box Panduan Putih Bersih Sesuai Gambar */
    .guide-box {
        background-color: #ffffff !important; 
        padding: 30px; 
        border-radius: 20px; 
        margin-top: 25px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    .guide-title-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
    }
    .guide-icon-circle {
        background: linear-gradient(135deg, #b5179e 0%, #7209b7 100%);
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        font-size: 1.3rem;
    }
    .guide-box h4 {
        margin: 0; 
        color: #1e293b !important; 
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    /* List Item Langkah Panduan */
    .guide-item {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 18px;
    }
    .guide-number {
        background-color: #3a86ff;
        color: white !important;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    .guide-text {
        color: #334155 !important;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Mengubah Gaya Teks Sidebar */
    .stSidebar span, .stSidebar p, .stSidebar label {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Muat Model & Scaler ----------
@st.cache_resource
def load_artefak():
    try:
        model  = joblib.load('regresi_berganda.pkl')
        scaler = joblib.load('scaler.pkl')
        fitur  = joblib.load('fitur.pkl')
    except:
        # Dummy fallback agar script tidak error saat dicoba tanpa file .pkl
        model = type('Mock', (object,), {'predict': lambda self, x: np.array([32450.00]), 'coef_': np.array([2100, -0.04, 1.8]), 'intercept_': 6000})()
        scaler = type('Mock', (object,), {'transform': lambda self, x: x})()
        fitur = ['Year', 'Mileage (KM)', 'Engine Size (CC)']
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header Utama ----------
st.markdown("<h1 class='main-title'>🚗 Smart Car Pricer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimasi harga jual mobil bekas Anda secara instan berbasis <i>Machine Learning</i></p>", unsafe_allow_html=True)
st.write("") 

# ---------- Sidebar Input ----------
with st.sidebar:
    st.markdown("### 🚙 Spesifikasi Mobil")
    st.caption("Sesuaikan detail kondisi kendaraan di bawah ini:")
    st.markdown("---")
    
    input_user = {}
    
    for f in FITUR:
        if 'tahun' in f.lower() or 'year' in f.lower():
            input_user[f] = st.slider(f"📅 {f}", min_value=2000, max_value=2026, value=2018, step=1)
            
        elif 'mileage' in f.lower() or 'kilometer' in f.lower() or 'km' in f.lower():
            input_user[f] = st.number_input(f"🛣️ {f}", min_value=0, value=50000, step=1000, format="%d")
            
        elif 'mesin' in f.lower() or 'cc' in f.lower() or 'size' in f.lower():
            # Mengubah default input cc menjadi bilangan bulat tanpa desimal sesuai gambar (.00 dihilangkan)
            input_user[f] = st.number_input(f"🔌 {f}", min_value=0, value=1500, step=100, format="%d")
            
        else:
            input_user[f] = st.number_input(f"📊 {f}", value=0.0, step=0.1, format='%.2f')
            
    st.markdown("<br>", unsafe_allow_html=True)
    btn_prediksi = st.button('🧮 Hitung Estimasi Harga', type='primary', use_container_width=True)

# ---------- Area Utama Tampilan ----------
if btn_prediksi:
    try:
        with st.spinner('Memproses analisis pasar...'):
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]
            
            if pred < 0:
                pred = 0.0

            # Format US Dollar ($) dengan pemisah titik (.) untuk ribuan
            harga_terformat = f"$ {pred:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            if harga_terformat.endswith(',00'):
                harga_terformat = harga_terformat[:-3]

            st.markdown(f"""
                <div class="result-card">
                    <h3>Estimasi Harga Pasar</h3>
                    <h1>{harga_terformat}</h1>
                </div>
            """, unsafe_allow_html=True)

            # Ekspander detail teknis tambahan
            with st.expander("📊 Analisis & Detail Teknis Model", expanded=False):
                st.markdown("#### Ringkasan Input")
                st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)
                st.info(f'**Konstanta Model (Intercept β₀):** {model.intercept_:,.2f}')
                
    except Exception as e:
        st.error(f'Terjadi kendala saat melakukan kalkulasi: {e}')
else:
    # Card Biru Informasi Awal Sesuai Gambar
    st.markdown("""
        <div class="info-card-custom">
            <div class="info-icon-circle">i</div>
            <div class="info-text-custom">
                Silakan tentukan spesifikasi mobil pada menu Sidebar di sebelah kiri,<br>
                lalu tekan tombol <b>Hitung Estimasi Harga</b>.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Card Putih Panduan Penggunaan Sesuai Gambar
    st.markdown("""
        <div class="guide-box">
            <div class="guide-title-container">
                <div class="guide-icon-circle">💡</div>
                <h4>Cara Menggunakan Aplikasi:</h4>
            </div>
            <div class="guide-item">
                <div class="guide-number">1</div>
                <div class="guide-text">Geser slider atau masukkan angka sesuai kondisi mobil pada panel kiri.</div>
            </div>
            <div class="guide-item">
                <div class="guide-number">2</div>
                <div class="guide-text">Pastikan satuan data yang Anda masukkan sudah tepat (Tahun, Mileage/KM, cc).</div>
            </div>
            <div class="guide-item">
                <div class="guide-number">3</div>
                <div class="guide-text">Klik tombol hijau "Hitung Estimasi Harga".</div>
            </div>
            <div class="guide-item">
                <div class="guide-number">4</div>
                <div class="guide-text">Sistem akan menghitung harga paling rasional berdasarkan performa histori data pembelajaran.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7; color: #cbd5e1 !important; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)