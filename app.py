import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title='Prediksi Harga Mobil',
    page_icon='🚗',
    layout='centered', # Dikembalikan ke 'centered' agar box panduan tampak pas di tengah
    initial_sidebar_state='expanded'
)

# ---------- Custom CSS (Akurasi Tinggi Sesuai Gambar) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Background Utama Aplikasi (Gelap Kebiruan) */
    .stApp {
        background-color: #0d1424 !important;
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Override warna teks global bawaan streamlit agar kontras */
    .stApp p, .stApp span, .stApp label, .stApp h3 {
        color: #ffffff !important;
    }
    
    /* Judul Utama dengan Gradasi Ungu-Biru */
    .main-title {
        text-align: center; 
        font-size: 2.8rem;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        text-align: center; 
        font-size: 1rem; 
        color: #94a3b8 !important; 
        margin-top: 5px;
        margin-bottom: 25px;
    }
    
    /* Card Info Awal (Biru Gelap Transparan) */
    .info-card-custom {
        background: rgba(30, 41, 59, 0.5);
        padding: 20px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .info-icon-circle {
        background-color: #3b82f6;
        color: white !important;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    /* Tombol Utama Sidebar Berwarna Ungu Gradasi */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
    }
    
    /* Box Panduan Putih Bersih */
    .guide-box {
        background-color: #ffffff !important; 
        padding: 30px; 
        border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .guide-box h4 {
        margin: 0 0 25px 0; 
        color: #0f172a !important; 
        font-weight: 700;
        font-size: 1.25rem;
    }
    .guide-item {
        display: flex;
        align-items: flex-start;
        gap: 15px;
        margin-bottom: 18px;
    }
    .guide-number {
        background-color: #3b82f6;
        color: white !important;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .guide-text {
        color: #475569 !important;
        font-size: 0.95rem;
        line-height: 1.5;
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
        # Fallback dummy data untuk keperluan pengujian UI
        model = None
        scaler = None
        fitur = ['Year', 'Mileage (KM)', 'Kapasitas Mesin (cc)']
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header Utama ----------
st.markdown("<h1 class='main-title'>🚗 Smart Car Pricer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimasi harga jual mobil bekas Anda secara instan berbasis <i>Machine Learning</i></p>", unsafe_allow_html=True)

# ---------- Sidebar Input ----------
with st.sidebar:
    st.markdown("### ⚙️ Spesifikasi Mobil")
    st.caption("Sesuaikan detail kondisi kendaraan di bawah ini:")
    st.markdown("---")
    
    input_user = {}
    for f in FITUR:
        if 'tahun' in f.lower() or 'year' in f.lower():
            input_user[f] = st.slider(f"📅 {f}", min_value=2000, max_value=2026, value=2018, step=1)
        elif 'mileage' in f.lower() or 'kilometer' in f.lower() or 'km' in f.lower():
            input_user[f] = st.number_input(f"🛣️ {f}", min_value=0, value=50000, step=1000, format="%d")
        elif 'mesin' in f.lower() or 'cc' in f.lower():
            input_user[f] = st.selectbox(f"🔌 {f}", options=[1000, 1200, 1500, 2000, 2500, 3000], index=2)
        else:
            input_user[f] = st.number_input(f"📊 {f}", value=0.0, step=0.1, format='%.2f')
            
    st.markdown("<br>", unsafe_allow_html=True)
    btn_prediksi = st.button('🚀 Hitung Estimasi Harga', type='primary', use_container_width=True)

# ---------- Area Utama Konten ----------

# Box Info Biru (Sesuai Gambar)
st.markdown("""
    <div class="info-card-custom">
        <div class="info-icon-circle">i</div>
        <div style="color: #cbd5e1 !important; font-size: 0.95rem;">
            Silakan tentukan spesifikasi mobil pada menu Sidebar di sebelah kiri,<br>
            lalu tekan tombol <span style="color: #3b82f6 !important; font-weight: 600;">Hitung Estimasi Harga</span>.
        </div>
    </div>
""", unsafe_allow_html=True)

# Box Panduan Cara Menggunakan (Tampil Tunggal Secara Centered)
st.markdown("""
    <div class="guide-box">
        <h4>💡 Cara Menggunakan Aplikasi:</h4>
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
st.markdown("<br><br><hr style='opacity: 0.05;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.5; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)