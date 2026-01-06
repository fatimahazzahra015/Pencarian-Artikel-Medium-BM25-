# Import library Flask
from flask import Flask, request, render_template

# Import pustaka untuk pemrosesan data
import pandas as pd
import re

# Import pustaka untuk preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
import nltk

# Unduh data stopword dan tokenizer dari NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Inisialisasi Flask
app = Flask(__name__)

# ----------------------------------------
# LOAD & PERSIAPAN DATASET
# ----------------------------------------

# Membaca dataset Medium berbahasa Indonesia
df = pd.read_csv("medium_indonesia_scraped.csv")

# Tambahkan kolom 'updatedTags' jika belum ada
if "updatedTags" not in df.columns:
    df["updatedTags"] = "-"

# Mengisi nilai kosong (NaN) dengan nilai default
df.fillna({
    "text": "",
    "title": "Tanpa Judul",
    "authors": "Tidak Diketahui",
    "updatedTags": "-",
    "timestamp": "-",
    "image_url": "",
}, inplace=True)

# Pastikan kolom 'text' bertipe string
df["text"] = df["text"].astype(str)

# Ambil daftar isi artikel (corpus) untuk digunakan dalam indexing BM25
corpus = df["text"].tolist()

# Daftar stopword bahasa Indonesia
stop_words = set(stopwords.words("indonesian"))

# Regex untuk menghapus kata pendek (<= 2 karakter)
shortword = re.compile(r'\W*\b\w{1,2}\b')

# ----------------------------------------
# FUNGSI TOKENISASI DOKUMEN
# ----------------------------------------
def tokenize(text):
    """
    Membersihkan teks, menghapus karakter khusus, stopwords, dan kata pendek.
    Mengembalikan token-token dari teks.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    tokens = [w for w in wordpunct_tokenize(text) if w.lower() not in stop_words]
    tokens = shortword.sub('', ' '.join(tokens)).split()
    return tokens

# ----------------------------------------
# FUNGSI PEMISAH PARAGRAF: PREVIEW & DETAIL
# ----------------------------------------
def smart_cut(text, max_chars=500):
    """
    Memisahkan teks menjadi dua bagian: preview (ringkasan pendek) dan detail (lanjutan),
    berdasarkan jumlah karakter maksimum.
    """
    paragraphs = [p.strip() for p in text.strip().split('\n') if p.strip()]
    preview_paragraphs = []
    detail_paragraphs = []
    total_len = 0
    for p in paragraphs:
        if total_len + len(p) <= max_chars:
            preview_paragraphs.append(p)
            total_len += len(p)
        else:
            detail_paragraphs.append(p)
    return preview_paragraphs, detail_paragraphs

# Tokenisasi seluruh dokumen
tokenized_corpus = [tokenize(doc) for doc in corpus]

# Buat index BM25 dari dokumen yang sudah ditokenisasi
bm25 = BM25Okapi(tokenized_corpus)

# -----------------
# ROUTE UTAMA 
# -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []              # Hasil pencarian
    query = ""                # Query pengguna
    selected_author = request.form.get("author", "")
    selected_tag = request.form.get("tag", "")
    selected_year = request.form.get("year_filter", "")
    limit = request.form.get("max_results", "20")

    # Ambil daftar penulis unik dari dataset
    authors = sorted(df["authors"].dropna().unique())

    # Ambil daftar tag unik dari kolom updatedTags
    tags = sorted(set(
        tag.strip()
        for row in df.get("updatedTags", pd.Series(["-"] * len(df))).dropna()
        for tag in str(row).split(",")
        if tag.strip() != "-"
    ))

    # Ambil daftar tahun dari kolom timestamp
    years = sorted({str(ts)[:4] for ts in df["timestamp"] if str(ts)[:4].isdigit()})

    # ----------------------------------------
    # PROSES POST (Saat Formulir Dikirim)
    # ----------------------------------------
    if request.method == "POST":
        query = request.form.get("query", "")  # Ambil query dari form
        query_tokens = tokenize(query)         # Tokenisasi query
        scores = bm25.get_scores(query_tokens) # Hitung skor BM25 untuk setiap dokumen
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        try:
            limit = int(limit)
        except ValueError:
            limit = 20

        count = 0
        for i in top_n:
            if scores[i] <= 0:
                continue

            row = df.iloc[i]

            # Filter berdasarkan penulis (author)
            if selected_author and str(row.get("authors", "")) != selected_author:
                continue

            # Filter berdasarkan tag
            if selected_tag and selected_tag not in str(row.get("updatedTags", "")):
                continue

            # Filter berdasarkan tahun
            if selected_year:
                year = str(row.get("timestamp", ""))[:4]
                if year != selected_year:
                    continue

            # Buat ringkasan preview dan detail
            preview, detail = smart_cut(str(row.get("text", "")))

            # Simpan hasil pencarian
            results.append({
                "title": str(row.get("title", "Tanpa Judul")),
                "authors": str(row.get("authors", "Tidak Diketahui")),
                "tags": str(row.get("updatedTags", "-")),
                "timestamp": str(row.get("timestamp", "-"))[:4],
                "url": str(row.get("url", "#")),
                "preview": preview,
                "detail": detail,
                "show_detail": bool(detail),
                "score": round(scores[i], 3),
                "image_url": str(row.get("image_url", "")),
            })

            count += 1
            if count >= limit:
                break

    # Render halaman utama dengan data hasil pencarian dan opsi filter
    return render_template("index.html", results=results, query=query,
                        authors=authors, selected_author=selected_author,
                        tags=tags, selected_tag=selected_tag,
                        year_filter=selected_year,
                        max_results=limit)

if __name__ == "__main__":
    app.run(debug=True)
