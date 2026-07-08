# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Siapkan data sederhana (Nanti bisa diganti dengan baca file CSV/Database)
data = {
    'komentar': [
        "Wah videonya sangat bermanfaat, terima kasih!",
        "link gacor rtp live hari ini depo 10k wd jutaan",
        "Bang coba bahas materi yang lain dong",
        "situs terpercaya pasti bayar jp paus bosku klik profil"
    ],
    'label': [0, 1, 0, 1] # 0 = Aman, 1 = Judol
}
df = pd.DataFrame(data)

# 2. Ekstraksi Fitur Teks
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['komentar'])
y = df['label']

# 3. Latih Model
model = LogisticRegression()
model.fit(X, y)

# 4. Simpan Model dan Vectorizer ke dalam file
joblib.dump(model, 'model_judol.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model AI berhasil dilatih dan disimpan!")