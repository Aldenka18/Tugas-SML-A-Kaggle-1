# **Tugas Kaggle 1 Statistical Machine Learning**

## *Anggota Kelompok:*
1. Aldenka Rifqi Ganendra Murti / 5003231015
2. Moh. Nafri Rehanata / 5003231124
3. Rafli Maulana / 5003231127
___

## Gambaran Umum

Proyek ini bertujuan untuk memprediksi atrisi karyawan (employee attrition) menggunakan teknik-teknik statistical machine learning. Prosesnya meliputi memuat data karyawan, melakukan analisis data eksplorasi (EDA), pre-processing data, pemilihan fitur yang relevan, melatih berbagai model klasifikasi, melakukan tuning hiperparameter untuk model dengan kinerja terbaik, mengevaluasi model akhir, dan menghasilkan prediksi.

## Data

Proyek ini menggunakan tiga dataset:
* train.csv: Berisi data latih dengan berbagai fitur dan variabel target 'Attrition'.
* test.csv: Berisi data uji dengan fitur yang sama, yang perlu diprediksi.
* sample_submission.csv: Menyediakan format untuk file submisi.

## Alur Kerja

1.  *Membuat Venv:* Membuat virtual environment dengan requirement.txt
  
2.  *Import Library:* Library Python yang diperlukan seperti pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, dan LightGBM diimpor.

3.  *Load Data:* Dataset train, test, dan sample_submission dimuat menggunakan pandas.

4.  *Analisis Data Eksplorasi (EDA):*
    * *Pemeriksaan Awal:* Struktur data, tipe data, dan jumlah data non-null diperiksa menggunakan .info(). Tidak ditemukan nilai yang hilang pada awalnya.
    * *Pemetaan Kategori:* Kode numerik untuk variabel kategori ordinal (misalnya, Education, EnvironmentSatisfaction) diganti dengan label string yang bermakna. JobLevel dan StockOptionLevel diubah menjadi tipe object.
    * *Penanganan Duplikat:* Entri id yang duplikat diidentifikasi dalam set data latih dan dihapus, dengan mempertahankan data yang pertama muncul.
    * *Penghapusan Kolom yang Tidak Relevan:* Kolom dengan nilai konstan (EmployeeCount, Over18, StandardHours) atau pengidentifikasi unik (id, EmployeeNumber) dihapus dari set data latih dan uji.
    * *Statistika Deskriptif:* Statistik ringkasan dihasilkan untuk kolom numerik dan kategorikal menggunakan .describe().
    * *Visualisasi:*
        * Boxplot digunakan untuk membandingkan distribusi fitur-fitur numerik yang sama antara set data latih dan uji.
        * Histogram/Plot distribusi dibuat untuk fitur-fitur numerik dalam set data latih untuk memvisualisasikan distribusinya dan menilai kemiringan (skewness).

5.  *Train Test Split:*
    * Data latih (train.csv) dipisahkan menjadi fitur (X) dan variabel target (y = 'Attrition').
    * Fitur dan target selanjutnya dipisahkan menjadi set latih (X_train, y_train) dan set validasi (X_val, y_val) dengan rasio 80/20. Stratifikasi berdasarkan variabel target digunakan untuk menjaga proporsi kelas.

6.  *Pre-procesing:*
    * ColumnTransformer digunakan untuk menerapkan langkah-langkah pra-pemrosesan yang berbeda pada kolom numerik dan kategorikal.
    * *Pipeline Numerik:* Termasuk PowerTransformer (metode Yeo-Johnson) untuk koreksi kemiringan dan MinMaxScaler untuk penskalaan fitur ke rentang [0, 1].
    * *Pipeline Kategorikal:* Termasuk OneHotEncoder (menghapus kategori pertama untuk menghindari multikolinearitas dan menangani kategori yang tidak dikenal yang mungkin ada di data validasi/uji).

7.  *Pelatihan Model & Pemilihan Fitur:*
    * Beberapa model klasifikasi dievaluasi: SVC, RandomForestClassifier, LogisticRegression, XGBClassifier, dan LGBMClassifier.
    * Sebuah pipeline yang menggabungkan pra-pemrosesan, SelectKBest (menggunakan skor f_classif) untuk pemilihan fitur, dan model klasifikasi dibuat.
    * Cross-Validation Stratified K-Fold manual (3-fold) dilakukan pada set latih (X_train, y_train) untuk setiap model.
    * Proses ini diulang untuk berbagai nilai k (jumlah fitur yang dipilih: 45, 50, 55, 60, 62).
    * Rata-rata AUC (Area Under the ROC Curve) dihitung untuk setiap kombinasi model dan k.
    * Regresi Logistik secara umum menunjukkan skor AUC yang paling tinggi dan konsisten di berbagai nilai k, terutama dengan k=60 dan k=62.

8.  *Tuning Hiperparameter:*
    * GridSearchCV dilakukan pada pipeline Regresi Logistik menggunakan cross-validation terstratifikasi 5-fold.
    * Pencarian grid ini melakukan tuning pada jumlah fitur yang dipilih (k untuk SelectKBest) dan hiperparameter Regresi Logistik (penalty, C, solver).
    * Parameter terbaik yang ditemukan adalah: k=60, C=1, penalty='l1', dan solver='liblinear'.
    * Skor AUC cross-validation terbaik yang dicapai adalah sekitar 0.8636.

9.  *Evaluasi:*
    * Pipeline terbaik (Regresi Logistik dengan hiperparameter yang telah dituning dan k=60) dari hasil GridSearchCV dievaluasi pada set validasi yang terpisah (X_val, y_val).
    * Metrik Kinerja yang Dihasilkan:
        * AUC: 0.9327
        * Akurasi: 0.9295
        * Presisi: 0.9118
        * Recall: 0.6200
        * F1 Score: 0.7381

10.  *Prediksi:*
    * Pipeline terbaik digunakan untuk memprediksi probabilitas atrisi pada dataset test.csv.
    * Probabilitas prediksi disimpan ke dalam file submission_015 124 127_Final.csv sesuai format Kaggle.

11. *Penyimpanan Model:*
    * Pipeline akhir dengan kinerja terbaik (termasuk pra-pemrosesan, pemilihan fitur, dan model Regresi Logistik yang telah dituning) disimpan menggunakan pickle ke file Best Logistic Regression Pipeline.pkl untuk penggunaan di masa depan.

## Hasil

Kinerja terbaik pada set validasi dicapai menggunakan model *Regresi Logistik* dengan penalti L1, C=1, dan solver liblinear, yang dikombinasikan dengan pemilihan *60 fitur teratas* menggunakan SelectKBest. Pipeline ini menghasilkan *AUC validasi sebesar 0,9327*.
