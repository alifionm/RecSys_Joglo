import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. KONFIGURASI HALAMAN (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Rekomendasi Wisata Jogja-Solo",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CACHING DATA & MODEL (Agar Web Cepat)
# ==========================================
@st.cache_data
def load_data():
    try:
        df_wisata = pd.read_csv('data/Data_Wisata_Clean.csv')
        
        # PERBAIKAN: Jika kolom 'Deskripsi_Wisata_Clean' tidak ada, gunakan 'Deskripsi_Wisata'
        if 'Deskripsi_Wisata_Clean' not in df_wisata.columns:
            if 'Deskripsi_Wisata' in df_wisata.columns:
                df_wisata['Deskripsi_Wisata_Clean'] = df_wisata['Deskripsi_Wisata']
            else:
                df_wisata['Deskripsi_Wisata_Clean'] = "" # Fallback aman jika keduanya tidak ada
                
        # Pastikan tidak ada nilai kosong (NaN) agar TF-IDF tidak error
        df_wisata['Deskripsi_Wisata_Clean'] = df_wisata['Deskripsi_Wisata_Clean'].fillna('')
        
        return df_wisata
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}. Pastikan file ada di folder 'data/'.")
        return pd.DataFrame()

# ==========================================
# 3. ALGORITMA REKOMENDASI HYBRID
# ==========================================
class HybridRecommenderApp:
    def __init__(self, df_tourism, svd_model):
        self.df_tourism = df_tourism.reset_index(drop=True)
        self.svd_model = svd_model
        
        # Bangun TF-IDF saat aplikasi pertama kali dimuat
        self.tfidf = TfidfVectorizer()
        tfidf_matrix = self.tfidf.fit_transform(self.df_tourism['Deskripsi_Wisata_Clean'])
        self.cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_similar_items(self, wisata_id, top_n=30):
        try:
            idx = self.df_tourism.index[self.df_tourism['ID_Wisata'] == wisata_id][0]
        except IndexError:
            return pd.DataFrame()
            
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        tourism_indices = [i[0] for i in sim_scores]
        return self.df_tourism.iloc[tourism_indices].copy()

    def recommend(self, user_id, target_wisata_id, top_n=5):
        # Tahap 1: Content-Based (Ambil 30 wisata mirip)
        candidates = self.get_similar_items(target_wisata_id, top_n=30)
        
        if candidates.empty:
            return pd.DataFrame()

        # Tahap 2: Collaborative (SVD Prediksi Rating)
        predicted_ratings = []
        for _, row in candidates.iterrows():
            est_rating = self.svd_model.predict(user_id, row['ID_Wisata']).est
            predicted_ratings.append(est_rating)
            
        candidates['Prediksi_Rating'] = predicted_ratings
        
        # Tahap 3: Hybrid Sort
        final_recommendations = candidates.sort_values(by='Prediksi_Rating', ascending=False).head(top_n)
        return final_recommendations

# ==========================================
# 4. MEMBANGUN ANTARMUKA WEB (UI)
# ==========================================
def main():
    # Load assets
    df_wisata = load_data()
    svd_model = load_model()

    if df_wisata.empty or svd_model is None:
        st.warning("Sistem dihentikan karena data atau model tidak ditemukan.")
        return

    # Inisialisasi Mesin Rekomendasi
    recommender = HybridRecommenderApp(df_wisata, svd_model)

    # --- SIDEBAR: Input Pengguna ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2060/2060284.png", width=100)
        st.header("Preferensi Wisatawan")
        st.markdown("Masukkan profil dan destinasi terakhir yang Anda kunjungi.")
        
        # Input User ID (Disimulasikan)
        user_id = st.number_input("User ID Anda:", min_value=1, max_value=10000, value=1, step=1)
        
        # UX: Menampilkan NAMA wisata di dropdown, bukan ID (Biar user friendly)
        daftar_nama_wisata = df_wisata['Nama_Wisata'].tolist()
        wisata_terpilih = st.selectbox("Wisata Terakhir yang Anda Suka:", daftar_nama_wisata)
        
        # Konversi nama kembali ke ID
        id_wisata_terpilih = df_wisata.loc[df_wisata['Nama_Wisata'] == wisata_terpilih, 'ID_Wisata'].values[0]
        kategori_terpilih = df_wisata.loc[df_wisata['Nama_Wisata'] == wisata_terpilih, 'Kategori'].values[0]
        
        st.write("---")
        jumlah_rekomendasi = st.slider("Jumlah Rekomendasi:", min_value=3, max_value=10, value=5)
        
        tombol_cari = st.button("🔍 Temukan Rekomendasi", use_container_width=True, type="primary")

    # --- MAIN PAGE: Hero Section ---
    st.title("🗺️ Sistem Rekomendasi Pariwisata Jogja-Solo")
    st.markdown("**Platform Pintar (Hybrid AI)** untuk menemukan destinasi liburan terbaik yang disesuaikan dengan selera Anda.")
    st.write("---")

    # --- TAMPILAN HASIL REKOMENDASI ---
    if tombol_cari:
        st.info(f"Menganalisis kemiripan dengan **{wisata_terpilih}** ({kategori_terpilih}) dan mencocokkan dengan profil rating Anda...")
        
        # Eksekusi AI
        hasil = recommender.recommend(user_id=user_id, target_wisata_id=id_wisata_terpilih, top_n=jumlah_rekomendasi)
        
        if hasil.empty:
            st.error("Maaf, destinasi tidak ditemukan di database kami.")
        else:
            st.success(f"Tadaa! Ini Top-{jumlah_rekomendasi} Rekomendasi Khusus Untuk Anda:")
            
            # Membuat kartu-kartu visual menggunakan st.columns
            for index, row in hasil.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(f"📍 {row['Nama_Wisata']}")
                        st.caption(f"Kategori: {row['Kategori']} | ID Wisata: {row['ID_Wisata']}")
                        # Menampilkan sebagian deskripsi agar rapi
                        desc = str(row.get('Deskripsi_Wisata', 'Tidak ada deskripsi tersedia.'))
                        st.write(f"{desc[:150]}..." if len(desc) > 150 else desc)
                        
                    with col2:
                        # Menampilkan prediksi rating layaknya badge
                        st.metric(label="⭐ Prediksi Rating", value=f"{row['Prediksi_Rating']:.2f} / 5.0")
                    
                    st.divider() # Garis pembatas antar kartu

if __name__ == "__main__":
    main()