# Customer Churn Prediction Dashboard

Dashboard ini dibangun menggunakan [Streamlit](https://streamlit.io/) dan beberapa library pendukung seperti Pandas, NumPy, Matplotlib, Seaborn, Plotly, dan CatBoost untuk memprediksi kemungkinan churn (kehilangan pelanggan). Dashboard ini juga menampilkan insight model melalui visualisasi seperti grafik batang, scatter plot, histogram, dan pie chart.

## Fitur Utama

- **Prediksi Churn:** Input data keuangan pelanggan secara interaktif untuk memprediksi kemungkinan churn menggunakan model CatBoost.
- **Visualisasi Data:** Menampilkan berbagai grafik untuk analisis perbandingan saldo, distribusi umur, pola kredit-debit, dan analisis churn berdasarkan pekerjaan atau status pensiun.
- **Insight Model:** Menampilkan feature importance (pentingnya fitur) dan insight terkait faktor-faktor utama yang mempengaruhi churn.
- **Custom Theme:** Tampilan dashboard dengan tema biru dan putih yang disesuaikan menggunakan custom CSS.
- **Penggunaan Data Dummy:** Jika file data atau model tidak ditemukan, dashboard akan menghasilkan data contoh untuk demonstrasi.

## Struktur Proyek
├── app.py # File utama Streamlit yang berisi kode dashboard 
├── catboost_model_top.pkl # Model CatBoost terlatih 
├── data_prediction.csv # Data prediksi
├── churn_dataset_clean.csv # Dataset pembersihan untuk analisis churn 
├── icon.png # Icon untuk sidebar 
└── README.md 


## Instalasi

Pastikan Python 3.7 ke atas telah terinstall. Kemudian, instal dependency yang diperlukan menggunakan `pip`:

```bash
pip install streamlit pandas numpy matplotlib seaborn catboost plotly


## Menjalankan Dashboard

Untuk menjalankan dashboard, gunakan perintah berikut di terminal:

streamlit run app.py

Dashboard akan terbuka di browser dengan URL lokal (misalnya, http://localhost:8501).



## Penggunaan Churn Prediction:

Masukkan detail keuangan pelanggan pada input form di tab "📊 Churn Prediction".
Tekan tombol Predict Churn Probability untuk mendapatkan probabilitas churn.
Hasil prediksi akan ditampilkan dalam bentuk gauge chart beserta rekomendasi tindakan retensi (jika risiko tinggi) atau engagement (jika risiko rendah).
Model Insights:

Di tab "🔍 Model Insights" Anda dapat melihat grafik Feature Importance dan insight terkait fitur-fitur utama.
Berbagai visualisasi data akan ditampilkan dalam beberapa sub-tab untuk analisis saldo, distribusi umur, pola kredit-debit, dan analisis churn.

## Konfigurasi
Custom CSS: Tampilan dashboard telah disesuaikan dengan tema biru dan putih melalui blok CSS di awal kode.
Caching Data & Model: Menggunakan @st.cache_resource dan @st.cache_data untuk meningkatkan performa dengan caching model dan data.
Input Data: Jika file data_prediction.csv tidak ditemukan, kode akan menghasilkan data contoh menggunakan NumPy dan Pandas untuk demonstrasi.
Retensi dan Engagement: Berdasarkan hasil prediksi, dashboard memberikan saran retensi untuk pelanggan dengan risiko tinggi dan saran engagement untuk pelanggan dengan risiko rendah.


## Dependencies
Streamlit: Untuk membangun antarmuka web interaktif.
Pandas & NumPy: Untuk pengolahan data.
Matplotlib & Seaborn: Untuk visualisasi data statis (opsional).
Plotly: Untuk visualisasi data interaktif.
CatBoost: Untuk model prediksi churn.
Pickle: Untuk memuat model yang telah dilatih.


##Troubleshooting
Model Tidak Ditemukan: Jika file catboost_model_top.pkl tidak ditemukan, dashboard akan menggunakan model dummy dan menampilkan peringatan.
Data Tidak Ditemukan: Jika file data_prediction.csv tidak ada, dashboard akan membuat data contoh untuk demonstrasi. Pastikan file CSV disimpan di direktori yang sama dengan app.py.
Gambar Icon Tidak Muncul: Jika file icon.png tidak ada, sidebar akan menampilkan informasi bahwa icon tidak ditemukan.

