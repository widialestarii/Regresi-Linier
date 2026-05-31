import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi halaman ----------
st.set_page_config(
    page_title='Prediksi Regresi Linear',
    page_icon='📈',
    layout='centered',
    initial_sidebar_state='expanded'
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    /* Kotak Hasil Prediksi */
    .result-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
        animation: fadeIn 0.8s ease-out;
        margin-bottom: 20px;
    }
    .result-card h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
        color: white;
    }
    .result-card h1 {
        margin: 10px 0 0 0;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Mempercantik tombol */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Muat model & scaler (cached) ----------
@st.cache_resource
def load_artefak():
    model  = joblib.load('regresi_berganda.pkl')
    scaler = joblib.load('scaler.pkl')
    fitur  = joblib.load('fitur.pkl')
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header ----------
st.markdown("<h1 style='text-align: center;'>📈 Prediksi Regresi Linear</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; opacity: 0.8;'>Masukkan nilai tiap fitur di sidebar, lalu klik <b>Prediksi</b> untuk melihat hasil estimasi.</p>", unsafe_allow_html=True)
st.divider()

# ---------- Input di sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Input Fitur")
    st.caption("Silakan atur nilai fitur di bawah ini:")
    
    input_user = {}
    for f in FITUR:
        input_user[f] = st.number_input(
            label=f"**{f}**",
            value=0.0,
            step=0.1,
            format='%.4f',
        )
    st.markdown("<br>", unsafe_allow_html=True)
    btn_prediksi = st.button('🚀 Mulai Prediksi', type='primary', use_container_width=True)

# ---------- Area Utama Prediksi ----------
if btn_prediksi:
    try:
        with st.spinner('Menghitung prediksi...'):
            # Susun DataFrame sesuai urutan FITUR (hindari warning feature names)
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]

            # Tampilkan hasil dalam bentuk card modern
            st.markdown(f"""
                <div class="result-card">
                    <h3>Estimasi Nilai Target</h3>
                    <h1>{pred:,.4f}</h1>
                </div>
            """, unsafe_allow_html=True)

            # Detail tambahan disembunyikan dalam expander agar UI tetap rapi
            with st.expander("📊 Lihat Detail Teknis & Input", expanded=False):
                st.markdown("#### Input yang Digunakan")
                st.dataframe(pd.DataFrame([input_user]), use_container_width=True)
        
                st.markdown("#### Koefisien Model (Terstandarisasi)")
                df_koef = pd.DataFrame({
                    'Fitur': FITUR,
                    'Koefisien': model.coef_.round(4),
                })
                st.dataframe(df_koef, use_container_width=True, hide_index=True)
                st.info(f'Intercept (β₀) = {model.intercept_:.4f}')
                
    except Exception as e:
        st.error(f'Terjadi error saat melakukan prediksi: {e}')
else:
    st.info('👈 Silakan isi nilai fitur di sidebar terlebih dahulu, lalu klik tombol **🚀 Mulai Prediksi**.')

# ---------- Footer ----------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.9rem;'>🚀 Dibuat untuk <b>PPKD Jakarta Selatan — Kejuruan Data Analyst</b></p>", unsafe_allow_html=True)