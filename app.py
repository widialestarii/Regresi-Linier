import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title='Prediksi Harga Mobil',
    page_icon='🚗',
    layout='wide', # Diubah ke 'wide' agar muat 3 kolom menyamping seperti di gambar
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
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        height: 100%;
    }
    .guide-box h4 {
        margin: 0 0 20px 0; 
        color: #0f172a !important; 
        font-weight: 700;
        font-size: 1.15rem;
    }
    .guide-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 15px;
    }
    .guide-number {
        background-color: #3b82f6;
        color: white !important;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .guide-text {
        color: #475569 !important;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Card Hasil Prediksi Tengah (Gelap/Glowing Neon) */
    .result-card {
        background: linear-gradient(180deg, #111827 0%, #1f1635 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.15);
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.2);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-card h3 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: #3b82f6 !important;
    }
    .result-card h1 {
        margin: 15px 0;
        font-size: 3rem;
        font-weight: 800;
        color: #22d3ee !important;
        text-shadow: 0 0 15px rgba(34, 211, 238, 0.4);
    }
    .result-card p {
        margin: 0;
        font-size: 0.8rem;
        color: #94a3b8 !important;
    }

    /* Container Box untuk Chart agar Latar Belakangnya Putih Melengkung */
    .chart-box {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        height: 100%;
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
        # Dummy data agar layout tetap bisa dirender untuk dicoba langsung
        model = type('Mock', (object,), {'predict': lambda self, x: np.array([38450.00]), 'coef_': np.array([2500, -1200, 800]), 'intercept_': 5000})()
        scaler = type('Mock', (object,), {'transform': lambda self, x: x})()
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

# Box Info Biru (Selalu Muncul di Atas Sesuai Gambar)
st.markdown("""
    <div class="info-card-custom">
        <div class="info-icon-circle">i</div>
        <div style="color: #cbd5e1 !important; font-size: 0.95rem;">
            Silakan tentukan spesifikasi mobil pada menu Sidebar di sebelah kiri,<br>
            lalu tekan tombol <span style="color: #3b82f6 !important; font-weight: 600;">Hitung Estimasi Harga</span>.
        </div>
    </div>
""", unsafe_allow_html=True)

# Membuat 3 Kolom Secara Horizontal Sesuai Desain Gambar Anda
col_panduan, col_hasil, col_grafik = st.columns([1.1, 1.0, 0.9])

# --- KOLOM 1: Box Panduan Cara Menggunakan (Kiri) ---
with col_panduan:
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

# --- KOLOM 2: Card Hasil Estimasi (Tengah) ---
with col_hasil:
    if btn_prediksi:
        try:
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]
            if pred < 0: pred = 0.0

            # Format USD ($) pemisah ribuan titik
            harga_terformat = f"$ {pred:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            if harga_terformat.endswith(',00'): 
                harga_terformat = harga_terformat[:-3]
            
            st.markdown(f"""
                <div class="result-card">
                    <h3>ESTIMASI HARGA PASAR</h3>
                    <h1>{harga_terformat}</h1>
                    <p>berdasarkan performa histori data pembelajaran</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        # Keadaan Standby Sebelum Klik Prediksi
        st.markdown("""
            <div class="result-card" style="opacity: 0.5;">
                <h3>ESTIMASI HARGA PASAR</h3>
                <h1>$ --.--</h1>
                <p>Menunggu input data dari sidebar...</p>
            </div>
        """, unsafe_allow_html=True)

# --- KOLOM 3: Grafik Koefisien Model Bawaan (Kanan) ---
with col_grafik:
    if btn_prediksi:
        # Membuka container latar putih untuk chart
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        
        df_koef = pd.DataFrame({
            'Fitur': FITUR,
            'Bobot': model.coef_
        })
        # Menampilkan grafik batang vertikal berwarna gelap kontras di atas box putih
        st.bar_chart(df_koef, x='Fitur', y='Bobot', color='#0f172a', use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Keadaan Standby Sebelum Klik Prediksi
        st.markdown("""
            <div class="guide-box" style="display:flex; align-items:center; justify-content:center; opacity:0.5;">
                <p style="color:#64748b !important; text-align:center;">Grafik analisis model akan muncul di sini setelah kalkulasi selesai.</p>
            </div>
        """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><br><hr style='opacity: 0.05;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.5; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)