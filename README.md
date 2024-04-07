# Indonesian Toxic Comment Classification

## Background

Penggunaan internet dan media sosial di Indonesia telah mengalami pertumbuhan pesat dalam beberapa tahun terakhir. Seiring dengan perkembangan teknologi dan ketersediaan akses internet yang semakin mudah, banyak orang di Indonesia telah bergabung dengan berbagai platform media sosial seperti Facebook, Twitter, Instagram, YouTube, bahkan media sosial lokal seperti Kaskus. Berdasarkan data yang diterbitkan oleh datareportal.com pada Januari 2023, jumlah pengguna internet di Indonesia mencapai angka yang sangat signifikan, yaitu sekitar 212,9 juta orang atau 77% dari total populasi. Selain itu, tercatat sekitar 167 juta orang atau 60,4% dari total populasi merupakan pengguna media sosial aktif. Jumlah ini menandakan bahwa lebih dari setengah populasi Indonesia memiliki akses ke internet dan berpotensi menggunakan platform media sosial.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/image/Data-Tren-Pengguna-Internet-dan-Media-sosial-di-Indonesia-Tahun-2023.png" width="50%" height="50%">
  <br>
  Gambar 1. Data Jumlah Pengguna internet di Indonesia
</p>

Tingginya penetrasi internet dan media sosial di Indonesia juga menyebabkan banyak orang menghabiskan waktu harian mereka untuk berinteraksi di platform tersebut. Menurut data dari sumber yang sama, pada Januari 2023, rata-rata pengguna internet di Indonesia menghabiskan waktu lebih dari 3 jam per hari untuk beraktivitas di media sosial. Fenomena ini menunjukkan betapa signifikannya peran media sosial dalam kehidupan sehari-hari masyarakat Indonesia.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/image/Waktu-Orang-Indonesia-Mengakses-Media-Digital-Tahun-2023.png" width="50%" height="50%">
  <br>
  Gambar 1. Data Jumlah Waktu Orang Indonesia Mengakses Media Digital
</p>

Sayangnya, pertumbuhan penggunaan media sosial juga membawa dampak negatif. Salah satunya adalah adanya komentar-komentar toxic di berbagai platform media sosial. Komentar-komentar ini bisa berupa ujaran kebencian, pornografi, radikalisme, pelecehan, intimidasi, atau ancaman yang ditujukan kepada individu atau kelompok tertentu. Komentar toxic seperti ini dapat mengakibatkan perpecahan masyarakat, kerusuhan antar individu dan kelompok, gangguan emosional, dan bahkan berpotensi membahayakan kesejahteraan mental para korbannya. Pada tahun 2020, Microsoft merilis “Indeks Keberadaban Digital” atau “Digital Civility Index” yang menunjukkan tingkat keberadaban pengguna internet atau netizen sepanjang tahun 2020. Hasilnya memprihatinkan karena menunjukkan bahwa tingkat keberadaban (civility) netizen Indonesia sangat rendah. Laporan yang didasarkan atas survei pada 16.000 responden di 32 negara itu menunjukkan Indonesia ada di peringkat 29 dan menjadi negara dengan warga netizen paling tidak beradab di Asia Tenggara. Oleh karenanya, pendeteksian komentar-komentar yang mengandung unsur toxic menjadi sesuatu yang harus dikritisi.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/image/DCI 2020.jpg" width="50%" height="50%">
  <br>
  Gambar 1. Digital Civility Index 2020
</p>

Dalam menghadapi dan menangani masalah komentar toxic di media sosial, diperlukan suatu pendekatan yang efektif dan proaktif. Salah satu solusi yang dapat diadopsi adalah dengan membangun sebuah sistem klasifikasi komentar toxic berbahasa indonesia menggunakan model machine learning. Sistem ini bertujuan untuk secara otomatis mengidentifikasi komentar-komentar dalam bahasa indonesia yang mengandung unsur toxic sehingga dapat segera ditindaklanjuti oleh pihak platform media sosial. Dengan demikian, diharapkan akan tercipta lingkungan media sosial yang lebih aman, positif, dan harmonis bagi seluruh penggunanya.

## Problem Scope

Pada project ini akan digunakan sebuah data komentar berbahasa indonesia yang ada pada sosial media Twitter, Instagram, dan Kaskus yang diambil dari [Github](https://github.com/ahmadizzan/netifier). Dataset ini mengandung informasi sebagai berikut:


*   `original_text`: komentar yang dimuat
*   `source`: sumber media sosial dari komentar yang dimuat
*   `pornografi`: label apakah komentar mengandung unsur pornografi (0 tidak, 1 iya)
*   `sara`: label apakah komentar mengandung unsur sara (0 tidak, 1 iya)
*   `radikalisme`: label apakah komentar mengandung unsur radikalisme (0 tidak, 1 iya)
*   `pencemaran_nama_baik`: label apakah komentar mengandung unsur pencemaran nama baik (0 tidak, 1 iya)


Selanjutnya, untuk mengidentifikasi apakah komentar tersebut toxic atau tidak, kita akan menggunakan multilabel yaitu variabel `pornografi`, `sara`, `radikalisme` dan `pencemaran_nama_baik`. Sedangkan untuk prediktornya akan menggunakan variabel `original_text` yang akan diekstrak dan dibersihkan lebih lanjut agar lebih mendukung dan mempermudah pemodelan.

Untuk modelnya sendiri akan menggunakan algoritma Naive Bayes, Random Forest, dan XGBoost. Nantinya masing-masing model tersebut akan dibandingkan performanya untuk mengetahui model mana yang paling baik untuk digunakan.

## Installation

Untuk menjalankan proyek ini secara lokal, Anda perlu mengikuti langkah-langkah berikut:

1. Clone repositori ini:
   ```bash
   git clone https://github.com/syahvan/indo-toxic-comment-classification.git
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd indo-toxic-comment-classification
   ```
3. Pasang dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

## Usage

[![Aplikasi Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indo-toxic-comment-detector.streamlit.app/)

Anda dapat mengakses aplikasi Streamlit yang telah di-deploy [di sini](https://indo-toxic-comment-detector.streamlit.app/).

## References

[1] DataReportal. "Digital 2023: Indonesia." [Online]. Available: https://datareportal.com/reports/digital-2023-indonesia

[2] Ahmad Izzan. "Github: Indonesian Social Media Post Toxicity Dataset." [Online]. Available: https://github.com/ahmadizzan/netifier 

[3] Voice of America Indonesia. "Indeks Keberadaban Digital: Indonesia Terburuk se-Asia Tenggara." [Online]. Available: https://www.voaindonesia.com/a/indeks-keberadaban-digital-indonesia-terburuk-se-asia-tenggara/5794123.html