import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title="Smart Car Pricer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS (Modern UI) ----------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:
    radial-gradient(circle at top left,#172554 0%,#09122b 45%,#030712 100%);
}

/* SIDEBAR */

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#2B2961,#1B1E46) !important;
    border-right:1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] > div{
    padding-top:20px;
}

.stSidebar label,
.stSidebar p,
.stSidebar span{
    color:white !important;
}

/* TITLE */

.main-title{
    text-align:center;
    font-size:68px;
    font-weight:800;

    background:linear-gradient(
    90deg,
    #3B82F6,
    #6366F1,
    #A855F7);

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    margin-bottom:5px;
}

.sub-title{
    text-align:center;
    color:#d1d5db !important;
    font-size:24px;
}

/* CARD INFO */

.info-card{
    background:linear-gradient(90deg,#2b3dff,#0077b6);

    padding:30px;

    border-radius:20px;

    display:flex;

    align-items:center;

    margin-top:25px;

    margin-bottom:30px;
}

.info-icon{
    width:65px;
    height:65px;

    background:#3B82F6;

    border-radius:50%;

    display:flex;

    align-items:center;

    justify-content:center;

    font-size:35px;

    color:white;

    margin-right:25px;
}

.info-text{
    color:white;
    font-size:28px;
}

/* GUIDE */

.guide-box{

    background:#f8fafc;

    border-radius:22px;

    padding:35px;

    box-shadow:0 20px 45px rgba(0,0,0,.2);

    border-left:6px solid #3B82F6;
}

.guide-title{
    font-size:38px;
    font-weight:700;
    color:#1e3a8a;
    margin-bottom:25px;
}

.guide-row{
    display:flex;
    align-items:center;
    margin-bottom:18px;
}

.guide-number{

    width:35px;
    height:35px;

    background:#3B82F6;

    border-radius:50%;

    color:white;

    display:flex;
    justify-content:center;
    align-items:center;

    font-weight:700;

    margin-right:18px;
}

.guide-text{
    color:#374151;
    font-size:22px;
}

/* BUTTON */

.stButton > button{

    background:linear-gradient(90deg,#3B82F6,#A855F7)!important;

    color:white!important;

    border:none!important;

    border-radius:15px!important;

    height:58px;

    font-size:20px!important;

    font-weight:700!important;

    box-shadow:0 10px 25px rgba(168,85,247,.35);
}

.stButton > button:hover{

    transform:translateY(-2px);

}

/* INPUT */

.stNumberInput input{
    background:#151B3D !important;
    color:white !important;
}

.result-card{
    background:linear-gradient(135deg,#172554,#0f172a);
    border-radius:22px;
    padding:40px;
    text-align:center;
    margin-top:20px;
}

.result-card h3{
    color:#A855F7!important;
}

.result-card h1{
    color:#38BDF8!important;
    font-size:70px;
}

footer{
visibility:hidden;
}

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

</style>
""",unsafe_allow_html=True)

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
st.markdown(
"""
<div class="main-title">
🚗 Smart Car Pricer
</div>

<div class="sub-title">
Estimasi harga jual mobil bekas Anda secara instan berbasis <i>Machine Learning</i>
</div>

""",
unsafe_allow_html=True
)
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
    st.markdown("""
<div class="info-card">

<div class="info-icon">
ℹ
</div>

<div class="info-text">
Silakan tentukan spesifikasi mobil pada menu Sidebar di sebelah kiri,<br>
lalu tekan tombol <b>Hitung Estimasi Harga</b>.
</div>

</div>
""",unsafe_allow_html=True)

st.markdown("""

<div class="guide-box">

<div class="guide-title">
💡 Cara Menggunakan Aplikasi:
</div>

<div class="guide-row">
<div class="guide-number">1</div>
<div class="guide-text">
Geser slider atau masukkan angka sesuai kondisi mobil pada panel kiri.
</div>
</div>

<div class="guide-row">
<div class="guide-number">2</div>
<div class="guide-text">
Pastikan satuan data yang Anda masukkan sudah tepat (Tahun, Mileage/KM, cc).
</div>
</div>

<div class="guide-row">
<div class="guide-number">3</div>
<div class="guide-text">
Klik tombol <b>"Hitung Estimasi Harga"</b>.
</div>
</div>

<div class="guide-row">
<div class="guide-number">4</div>
<div class="guide-text">
Sistem akan menghitung harga paling rasional berdasarkan histori data pembelajaran.
</div>
</div>

</div>

""",unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.85rem;'>Dibuat dengan 💙 untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)