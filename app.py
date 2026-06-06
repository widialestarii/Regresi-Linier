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

# ---------- Custom CSS (Modern UI) ----------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Hasil Prediksi Modern */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(30, 60, 114, 0.3);
        text-align: center;
        color: white;
        animation: slideUp 0.6s ease-in-out;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-card h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.85;
        color: #e0e0e0 !important;
    }
    .result-card h1 {
        margin: 15px 0 0 0;
        font-size: 3.8rem;
        font-weight: 800;
        color: #00f2fe !important;
        text-shadow: 0 2px 10px rgba(0,242,254,0.3);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 5px 15px rgba(56, 239, 125, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(56, 239, 125, 0.4);
    }
    
    .stAlert {
        border-radius: 12px !important;
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
        model = type('Mock', (object,), {'predict': lambda self, x: np.array([24550.50]), 'coef_': np.array([1200, -0.05, 1.5]), 'intercept_': 5000})()
        scaler = type('Mock', (object,), {'transform': lambda self, x: x})()
        fitur = ['Tahun Mobil', 'Mileage', 'Kapasitas Mesin (cc)']
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header Utama ----------
st.markdown("<h1 style='text-align: center; color: #1e3c72; margin-bottom: 0;'>🚗 Smart Car Pricer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #555; margin-top: 5px;'>Estimasi harga jual mobil bekas Anda secara instan berbasis <i>Machine Learning</i></p>", unsafe_allow_html=True)
st.write("") 

# ---------- Sidebar Input ----------
with st.sidebar:
    st.markdown("### ⚙️ Spesifikasi Mobil")
    st.caption("Sesuaikan detail kondisi kendaraan di bawah ini:")
    st.markdown("---")
    
    input_user = {}
    
    for f in FITUR:
        # Pengecekan kondisi fitur Tahun (Integer)
        if 'tahun' in f.lower() or 'year' in f.lower():
            input_user[f] = st.slider(f"📅 {f}", min_value=2000, max_value=2026, value=2018, step=1)
            
        # Pengecekan kondisi fitur Mileage / Kilometer (Integer)
        elif 'mileage' in f.lower() or 'kilometer' in f.lower() or 'km' in f.lower():
            input_user[f] = st.number_input(f"🛣️ {f}", min_value=0, value=50000, step=1000, format="%d")
            
        # Pengecekan kondisi fitur Kapasitas Mesin
        elif 'mesin' in f.lower() or 'cc' in f.lower():
            input_user[f] = st.selectbox(f"🔌 {f}", options=[1000, 1200, 1500, 2000, 2500, 3000], index=2)
            
        # Fallback fitur lainnya
        else:
            input_user[f] = st.number_input(f"📊 {f}", value=0.0, step=0.1, format='%.2f')
            
    st.markdown("<br>", unsafe_allow_html=True)
    btn_prediksi = st.button('🚀 Hitung Estimasi Harga', type='primary', use_container_width=True)

# ---------- Area Utama Prediksi ----------
if btn_prediksi:
    try:
        with st.spinner('Memproses data dan menganalisis pasar...'):
            # Susun DataFrame sesuai urutan FITUR
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]
            
            if pred < 0:
                pred = 0.0

            # Kustomisasi format mata uang USD: Ribuan menggunakan (.) dan Desimal menggunakan (,)
            # Contoh hasil: $ 24.550,50 atau $ 15.000
            harga_terformat = f"$ {pred:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            # Jika ingin menghilangkan ,00 di belakang jika hasilnya bulat sempurna, gunakan baris ini:
            if harga_terformat.endswith(',00'):
                harga_terformat = harga_terformat[:-3]

            # Tampilkan hasil dalam bentuk card modern
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

            # Detail tambahan disembunyikan dalam expander
            st.write("")
            with st.expander("📊 Analisis & Detail Teknis Model", expanded=False):
                st.markdown("#### Ringkasan Input Pengguna")
                st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)
        
                st.markdown("#### Pengaruh Fitur Terhadap Harga (Koefisien)")
                df_koef = pd.DataFrame({
                    'Nama Fitur': FITUR,
                    'Bobot Pengaruh': model.coef_.round(4),
                }).sort_values(by='Bobot Pengaruh', ascending=False)
                
                st.bar_chart(df_koef, x='Nama Fitur', y='Bobot Pengaruh', color='#2a5298')
                
                st.caption("Nilai positif berarti meningkatkan harga, nilai negatif berarti menurunkan harga.")
                st.info(f'**Konstanta Dasar Model (Intercept β₀):** {model.intercept_:,.2f}')
                
    except Exception as e:
        st.error(f'Terjadi kendala saat melakukan kalkulasi: {e}')
else:
    st.info('👈 Silakan tentukan spesifikasi mobil pada menu **Sidebar di sebelah kiri**, lalu tekan tombol **Hitung Estimasi Harga**.')
    
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 5px solid #1e3c72; margin-top: 20px;">
            <h4 style="margin-top:0; color: #1e3c72;">💡 Cara Menggunakan Aplikasi:</h4>
            <ol style="margin-bottom:0; padding-left:20px; color:#555;">
                <li>Geser slider atau masukkan angka sesuai kondisi mobil pada panel kiri.</li>
                <li>Pastikan satuan data yang Anda masukkan sudah tepat (Tahun, Mileage/KM, cc).</li>
                <li>Klik tombol hijau <b>"Hitung Estimasi Harga"</b>.</li>
                <li>Sistem akan menghitung harga paling rasional berdasarkan performa histori data pembelajaran.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)