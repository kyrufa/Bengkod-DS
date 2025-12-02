import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

#Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_vc_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Judul dan Deskripsi ---
st.title("📡 Telco Customer Churn Prediction")
st.markdown("""
            Aplikasi ini menggunakan Machine Learning dengan model (**Voting Classifier**) untuk memprediksi 
            apakah pelanggan akan berhenti berlangganan (Churn) atau tidak. 
            Silakan isi formulir di sebelah kiri (sidebar) untuk mendapatkan prediksi.
""")

# --- 3. Sidebar Input Fitur ---
st.sidebar.header("📝 Input Data Pelanggan")

def user_input_features():
    # Demografi
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    
    # Layanan
    tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Akun
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=tenure * 50.0)

    # Simpan input ke dalam dictionary
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return data

input_data = user_input_features()

# Fungsi untuk mengubah input user menjadi format encoding yang diterapkan pada model
def preprocess_input(data):
    columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'gender_Male', 'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes', 
        'PhoneService_Yes', 'MultipleLines_Yes', 
        'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 
        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check'
    ]
    
    # Buat dataframe kosong dengan kolom yang sesuai
    df = pd.DataFrame(0, index=[0], columns=columns)
    
    # Isi Data Numerik
    df['tenure'] = data['tenure']
    df['MonthlyCharges'] = data['MonthlyCharges']
    df['TotalCharges'] = data['TotalCharges']
    
    # Isi Data Kategorikal (Manual Mapping agar akurat)
    if data['gender'] == 'Male': df['gender_Male'] = 1
    if data['SeniorCitizen'] == 'Yes': df['SeniorCitizen_1'] = 1
    if data['Partner'] == 'Yes': df['Partner_Yes'] = 1
    if data['Dependents'] == 'Yes': df['Dependents_Yes'] = 1
    if data['PhoneService'] == 'Yes': df['PhoneService_Yes'] = 1
    if data['MultipleLines'] == 'Yes': df['MultipleLines_Yes'] = 1
    
    if data['InternetService'] == 'Fiber optic': df['InternetService_Fiber optic'] = 1
    elif data['InternetService'] == 'No': df['InternetService_No'] = 1
    # DSL tidak perlu diset karena diwakili oleh 0 di kedua kolom (baseline)
    
    if data['OnlineSecurity'] == 'Yes': df['OnlineSecurity_Yes'] = 1
    if data['OnlineBackup'] == 'Yes': df['OnlineBackup_Yes'] = 1
    if data['DeviceProtection'] == 'Yes': df['DeviceProtection_Yes'] = 1
    if data['TechSupport'] == 'Yes': df['TechSupport_Yes'] = 1
    if data['StreamingTV'] == 'Yes': df['StreamingTV_Yes'] = 1
    if data['StreamingMovies'] == 'Yes': df['StreamingMovies_Yes'] = 1
    
    if data['Contract'] == 'One year': df['Contract_One year'] = 1
    elif data['Contract'] == 'Two year': df['Contract_Two year'] = 1
    
    if data['PaperlessBilling'] == 'Yes': df['PaperlessBilling_Yes'] = 1
    
    if data['PaymentMethod'] == 'Credit card (automatic)': df['PaymentMethod_Credit card (automatic)'] = 1
    elif data['PaymentMethod'] == 'Electronic check': df['PaymentMethod_Electronic check'] = 1
    elif data['PaymentMethod'] == 'Mailed check': df['PaymentMethod_Mailed check'] = 1

    return df

# Proses data input
input_df = preprocess_input(input_data)

# Tampilkan data input 
with st.expander("Lihat Data Input (Raw)"):
    st.write(pd.DataFrame([input_data]))

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Prediksi Churn"):
        if model is not None:
            # prediksi
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ **Prediksi: CHURN (Berhenti)**")
                st.write(f"Probabilitas Churn: **{probability*100:.2f}%**")
                st.info("Saran: Tawarkan diskon atau kontrak jangka panjang untuk mempertahankan pelanggan ini.")
            else:
                st.success(f"✅ **Prediksi: NO CHURN (Bertahan)**")
                st.write(f"Probabilitas Churn: **{probability*100:.2f}%**")
                st.balloons()
        else:
            st.error("Model belum dimuat. Pastikan file .joblib tersedia.")

with col2:
    # Visualisasi sederhana probabilitas
    if 'probability' in locals():
        st.write("### Tingkat Risiko")
        st.progress(int(probability * 100))
        if probability > 0.5:
            st.warning("Risiko Tinggi")
        else:
            st.success("Risiko Rendah")