# Import Flask dan fungsi terkait untuk membuat aplikasi web
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify  # Import fungsi-fungsi utama Flask
import os  # Untuk operasi file dan direktori
import csv  # Untuk membaca file CSV
import pandas as pd  # Untuk manipulasi data, terutama file CSV
from sklearn.feature_extraction.text import CountVectorizer  # Untuk representasi teks sebagai vektor fitur
from sklearn.metrics.pairwise import cosine_similarity  # Untuk menghitung kesamaan kosinus
from PyPDF2 import PdfReader  # Untuk membaca file PDF
import docx  # Untuk membaca file dokumen Word
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Untuk stemming kata dalam bahasa Indonesia

# Inisialisasi aplikasi Flask
app = Flask(__name__)  # Membuat instance aplikasi Flask

# Konfigurasi folder dokumen untuk upload
DOCUMENTS_FOLDER = 'dokumen'  # Lokasi folder untuk dokumen yang diunggah
IMG_FOLDER = 'img'  # Lokasi folder untuk gambar
app.config['UPLOAD_FOLDER'] = DOCUMENTS_FOLDER  # Menentukan folder unggahan dokumen
STOPWORDS_FILE = 'stopwordbahasa.csv'  # File CSV berisi daftar stopword

@app.route('/img/<filename>')  # Route untuk melayani file gambar
def serve_image(filename):
    return send_from_directory(IMG_FOLDER, filename)  # Mengirim file gambar dari folder

def extract_text_from_file(file_path):  # Fungsi untuk ekstraksi teks dari berbagai jenis file
    ext = file_path.split('.')[-1].lower()  # Mendapatkan ekstensi file
    text = ''  # Variabel untuk menyimpan teks yang diekstrak
    try:
        if ext == 'pdf':  # Jika file PDF
            reader = PdfReader(file_path)  # Membaca file PDF
            for page in reader.pages:  # Iterasi setiap halaman
                text += page.extract_text() or ''  # Ekstraksi teks
        elif ext == 'txt':  # Jika file teks
            with open(file_path, 'r', encoding='utf-8') as f:  # Membuka file teks
                text = f.read()  # Membaca isi file
        elif ext == 'csv':  # Jika file CSV
            df = pd.read_csv(file_path)  # Membaca CSV ke DataFrame
            text = df.to_string(index=False)  # Mengonversi DataFrame ke string
        elif ext == 'docx':  # Jika file Word
            doc = docx.Document(file_path)  # Membuka file Word
            text = ' '.join([para.text for para in doc.paragraphs])  # Menggabungkan teks dari setiap paragraf
        else:
            print(f"Unsupported file format: {ext}")  # Format file tidak didukung
    except Exception as e:  # Menangkap error saat membaca file
        print(f"Error reading file {file_path}: {e}")
    return text.strip()  # Mengembalikan teks tanpa spasi berlebih

@app.route('/')  # Route utama
def index():
    document_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith(('pdf', 'txt', 'csv', 'docx'))]  # Daftar dokumen di folder
    return render_template('index.html', documents=document_files)  # Menampilkan halaman utama

def load_stopwords():  # Fungsi untuk memuat daftar stopword
    stopwords_set = set()  # Set untuk menyimpan stopword
    try:
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:  # Membuka file stopword
            reader = csv.reader(f)  # Membaca file sebagai CSV
            for row in reader:  # Iterasi setiap baris
                stopwords_set.update(row[0].split(','))  # Menambahkan stopword ke set
    except Exception as e:  # Menangkap error saat memuat stopword
        print(f"Error loading stopwords: {e}")
    return stopwords_set  # Mengembalikan set stopword

def case_folding(text):  # Fungsi untuk mengubah teks menjadi huruf kecil
    return text.lower()  # Mengembalikan teks dalam huruf kecil

def tokenize(text):  # Fungsi untuk memecah teks menjadi token
    text = case_folding(text)  # Mengubah teks menjadi huruf kecil
    return text.split()  # Memisahkan teks berdasarkan spasi

def filtration(text, stopwords_set):  # Fungsi untuk menghilangkan stopword dari teks
    tokens = tokenize(text)  # Memecah teks menjadi token
    filtered_tokens = [word for word in tokens if word not in stopwords_set]  # Menghapus stopword
    return ' '.join(filtered_tokens)  # Menggabungkan token yang tersisa

def stemming(text):  # Fungsi untuk stemming teks
    factory = StemmerFactory()  # Membuat factory stemmer
    stemmer = factory.create_stemmer()  # Membuat stemmer
    return stemmer.stem(text)  # Mengembalikan teks yang sudah distem

def preprocess(text, stopwords_set):  # Fungsi untuk preprocessing teks
    text = filtration(text, stopwords_set)  # Memfilter teks
    text = stemming(text)  # Stemming teks
    return text  # Mengembalikan teks yang sudah diproses

def vectorize_documents(documents, query, stopwords_set):  # Fungsi untuk membuat vektor dokumen dan query
    processed_docs = [preprocess(doc, stopwords_set) for doc in documents]  # Preprocessing dokumen
    processed_query = preprocess(query, stopwords_set)  # Preprocessing query
    vectorizer = CountVectorizer()  # Membuat objek CountVectorizer
    X = vectorizer.fit_transform(processed_docs + [processed_query])  # Membuat representasi vektor
    doc_vectors = X[:-1].toarray()  # Vektor dokumen
    query_vector = X[-1].toarray()  # Vektor query
    return doc_vectors, query_vector  # Mengembalikan vektor dokumen dan query

def calculate_similarity(doc_vectors, query_vector):  # Fungsi untuk menghitung kesamaan
    similarities = cosine_similarity(query_vector, doc_vectors)[0]  # Menghitung kesamaan kosinus
    ranked_documents = similarities.argsort()[::-1]  # Mengurutkan dokumen berdasarkan kesamaan
    return ranked_documents, similarities  # Mengembalikan urutan dokumen dan nilai kesamaan

@app.route('/search', methods=['GET'])  # Route untuk pencarian
def search():
    query = request.args.get('query', '').strip()  # Mendapatkan query pencarian
    if not query:  # Jika query kosong
        return render_template('result.html', query=query, documents=[], error="Masukkan query pencarian.")  # Menampilkan pesan error

    document_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith(('pdf', 'txt', 'csv', 'xlsx', 'docx'))]  # Daftar dokumen
    documents = [extract_text_from_file(os.path.join(DOCUMENTS_FOLDER, file)) for file in document_files]  # Ekstraksi teks dokumen
    stopwords_set = load_stopwords()  # Memuat stopword

    doc_vectors, query_vector = vectorize_documents(documents, query, stopwords_set)  # Membuat vektor dokumen dan query
    ranked_documents, similarities = calculate_similarity(doc_vectors, query_vector)  # Menghitung kesamaan

    results = []  # Daftar hasil pencarian
    for i in ranked_documents:  # Iterasi dokumen yang diurutkan
        if similarities[i] > 0:  # Jika kesamaan lebih dari 0
            document_name = document_files[i]  # Nama dokumen
            snippet = documents[i][:200]  # Cuplikan teks
            file_format = document_name.split('.')[-1].upper()  # Format file
            similarity_score = similarities[i]  # Nilai kesamaan
            results.append((document_name, snippet, file_format, similarity_score))  # Menambahkan hasil ke daftar

    return render_template('result.html', query=query, documents=results)  # Menampilkan hasil pencarian

@app.route('/document/<filename>')  # Route untuk melihat dokumen
def view_document(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Lokasi file
    if not os.path.exists(file_path):  # Jika file tidak ditemukan
        return "Dokumen tidak ditemukan.", 404  # Mengembalikan pesan error

    original_text = extract_text_from_file(file_path)  # Ekstraksi teks asli
    stopwords_set = load_stopwords()  # Memuat stopword
    tokenized_text = tokenize(original_text)  # Tokenisasi teks asli
    filtered_text = filtration(original_text, stopwords_set)  # Memfilter teks asli
    stemmed_text = stemming(filtered_text)  # Stemming teks

    return render_template(  # Menampilkan halaman dengan teks asli, tokenisasi, filtering, dan stemming
        'document.html',
        filename=filename,
        original_text=original_text,
        tokenized_text=' '.join(tokenized_text),
        filtered_text=filtered_text,
        stemmed_text=stemmed_text
    )

@app.route('/download/<filename>')  # Route untuk mengunduh dokumen
def download_document(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Lokasi file
    if os.path.exists(file_path):  # Jika file ditemukan
        return send_file(file_path, as_attachment=True)  # Mengunduh file
    return "File tidak ditemukan.", 404  # Mengembalikan pesan error

@app.route('/kata-dasar', methods=['POST'])  # Route untuk menghitung kata dasar
def kata_dasar():
    filename = request.form['filename']  # Mendapatkan nama file dari form
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Lokasi file
    original_text = extract_text_from_file(file_path)  # Ekstraksi teks asli
    stopwords_set = load_stopwords()  # Memuat stopword
    stemmed_text = preprocess(original_text, stopwords_set)  # Preprocessing teks
    stemmed_words = stemmed_text.split()  # Memisahkan kata dasar
    word_count = {}  # Kamus untuk menghitung jumlah kata
    for word in stemmed_words:  # Iterasi setiap kata
        word_count[word] = word_count.get(word, 0) + 1  # Menambah jumlah kata

    return jsonify({'kata_dasar': word_count})  # Mengembalikan hasil dalam format JSON

if __name__ == '__main__':  # Jika file dijalankan sebagai program utama
    app.run(debug=True)  # Menjalankan server Flask dengan debug aktif
