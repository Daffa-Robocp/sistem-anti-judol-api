from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Inisialisasi Aplikasi Server
app = FastAPI(
    title="Gateway Anti-Judol",
    description="Server API untuk mendeteksi spam komentar judi online"
)

print("Memuat model kecerdasan buatan...")
# 2. Memuat "Otak" AI yang sudah dilatih pada Tahap 2
# Pastikan file .pkl ini berada di folder yang sama dengan main.py
try:
    model = joblib.load('model_judol.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model AI berhasil dimuat! Server siap digunakan.")
except:
    print("⚠️ ERROR: File .pkl tidak ditemukan. Jalankan Tahap 2 (train_model.py) terlebih dahulu.")

# 3. Mendefinisikan Struktur Data yang Akan Diterima Server
class Komentar(BaseModel):
    platform: str      # Contoh: "youtube", "instagram"
    comment_id: str    # Contoh: "Ugw_123456"
    teks: str          # Isi komentar yang akan dicek

# 4. Membuat Endpoint (Jalur Masuk Data)
@app.post("/cek-komentar/")
async def cek_komentar(data: Komentar):
    # A. Bersihkan teks (ubah jadi huruf kecil semua)
    teks_bersih = data.teks.lower()
    
    # B. Ubah teks menjadi angka, lalu minta AI memprediksi
    vektor = vectorizer.transform([teks_bersih])
    hasil_prediksi = model.predict(vektor)
    
    # C. Logika Keputusan (0 = Aman, 1 = Judol)
    if hasil_prediksi[0] == 1:
        # Jika AI mendeteksi ini adalah spam judol
        hapus_komentar(data.platform, data.comment_id)
        return {
            "status": "DIBLOKIR", 
            "pesan": "Komentar mengandung spam judi online."
        }
    
    # Jika aman
    return {
        "status": "AMAN", 
        "pesan": "Komentar bersih."
    }

# 5. Fungsi Simulasi Eksekusi
def hapus_komentar(platform, comment_id):
    # Di masa depan, kode API YouTube/Instagram yang sesungguhnya diletakkan di sini
    print(f"🚨 [TINDAKAN] Menghapus komentar dengan ID '{comment_id}' di {platform.upper()}")
