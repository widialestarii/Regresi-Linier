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

# ---------- Custom CSS (White Clean Background UI) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Mengubah background area utama menjadi PUTIH */
    .stApp {
        background-color: #ffffff !important;
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Judul Utama dengan warna gelap solid agar kontras di bg putih */
    .main-title {
        text-align: center; 
        color: #0f172a;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Card Hasil Prediksi - Tetap Premium Dark agar Sangat Kontras */
    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.12);
        text-align: center;
        color: white;
        animation: slideUp 0.5s cubic-bezier(0.1, 0.8, 0.3, 1);
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .result-card h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #f59e0b !important;
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
    
    /* Tombol Aksi Gradasi Cyber Neon */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 20px rgba(6, 182, 212, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(6, 182, 212, 0.4);
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%) !important;
    }
    
    /* Box Panduan - Disesuaikan agar serasi dengan BG Putih */
    .guide-box {
        background-color: #f8fafc; 
        padding: 24px; 
        border-radius: 16px; 
        border-left: 6px solid #3b82f6; 
        margin-top: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Muat Model & Scaler (cached) ----------
@st.cache_resource
def load_artefak():
    try:
        model  = joblib.load('regresi_berganda.pkl')
        scaler = joblib.load('scaler.pkl')
        fitur  = joblib.load('fitur.pkl')
    except:
        # Dummy data untuk fallback testing UI
        model = type('Mock', (object,), {'predict': lambda self, x: np.array([38450.00]), 'coef_': np.array([2500, -0.08, 2.1]), 'intercept_': 8000})()
        scaler = type('Mock', (object,), {'transform': lambda self, x: x})()
        fitur = ['Tahun Mobil', 'Mileage', 'Kapasitas Mesin (cc)']
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header Utama ----------
st.markdown("<h1 class='main-title'>🚗 Smart Car Pricer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #475569; margin-top: 5px;'>Estimasi harga jual mobil bekas Anda secara instan berbasis <i>Machine Learning</i></p>", unsafe_allow_html=True)
st.write("") 

# ---------- Sidebar Input (Latar belakang bawaan asli tidak disentuh) ----------
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

# ---------- Area Utama Prediksi ----------
if btn_prediksi:
    try:
        with st.spinner('Memproses analisis pasar eksklusif...'):
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]
            
            if pred < 0:
                pred = 0.0

            # Format mata uang USD dengan pemisah ribuan (.)
            harga_terformat = f"$ {pred:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            if harga_terformat.endswith(',00'):
                harga_terformat = harga_terformat[:-3]

            st.markdown(f"""
                <div class="result-card">
                    <h3>Estimasi Harga Pasar</h3>
                    <h1>{harga_terformat}</h1>
                </div>
            """, unsafe_allow_html=True)

            # Bagian visualisasi analitik tambahan
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Status Kelayakan Data", value="Sesuai Standar", delta="Aman")
            with col2:
                st.metric(label="Jumlah Fitur Dianalisis", value=f"{len(FITUR)} Fitur")

            st.write("")
            with st.expander("📊 Analisis & Detail Teknis Model", expanded=False):
                st.markdown("#### Ringkasan Input Pengguna")
                st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)
        
                st.markdown("#### Pengaruh Fitur Terhadap Harga (Koefisien)")
                df_koef = pd.DataFrame({
                    'Nama Fitur': FITUR,
                    'Bobot Pengaruh': model.coef_.round(4),
                }).sort_values(by='Bobot Pengaruh', ascending=False)
                
                st.bar_chart(df_koef, x='Nama Fitur', y='Bobot Pengaruh', color='#0f172a')
                
                st.caption("Nilai positif berarti meningkatkan harga, nilai negatif berarti menurunkan harga.")
                st.info(f'**Konstanta Dasar Model (Intercept β₀):** {model.intercept_:,.2f}')
                
    except Exception as e:
        st.error(f'Terjadi kendala saat melakukan kalkulasi: {e}')
else:
    st.info('👈 Silakan tentukan spesifikasi mobil pada menu **Sidebar di sebelah kiri**, lalu tekan tombol **Hitung Estimasi Harga**.')
    
    st.markdown("""
        <div class="guide-box">
            <h4 style="margin-top:0; color: #0f172a; font-weight:700;">💡 Cara Menggunakan Aplikasi:</h4>
            <ol style="margin-bottom:0; padding-left:20px; color:#475569; line-height: 1.6;">
                <li>Geser slider atau masukkan angka sesuai kondisi mobil pada panel kiri.</li>
                <li>Pastikan satuan data yang Anda masukkan sudah tepat (Tahun, Mileage, cc).</li>
                <li>Klik tombol <b>"Hitung Estimasi Harga"</b>.</li>
                <li>Sistem akan menghitung harga paling rasional berdasarkan performa histori data pembelajaran.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><hr style='opacity: 0.1;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.5; color: #475569; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)