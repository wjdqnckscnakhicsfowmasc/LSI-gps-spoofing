# Serah terima B2 ke Tim FPGA

## Progres yang tercatat

- B2 membangun dataset GPS dan encoder heatmap 32×32.
- Model `time_5m_centered` mencapai akurasi 76% pada 100 sampel dari 20 sesi uji dalam evaluasi software.
- Ekspor golden vectors sudah dijalankan. Log B2 mencatat 43 file `.memh` lulus audit software.
- Simulasi RTL terhadap golden vectors dan hasil pengujian DE10-Lite belum terverifikasi. B1 sudah menerima pesan NMEA dari modul GPS. Data GPS dengan posisi valid masih perlu diverifikasi.

## Paket yang disiapkan B2

Salin skrip encoder dan referensi fixed-point, checkpoint model yang diekspor, serta paket `.memh` beserta manifest dari folder kerja `team_b_system`. Catat versi model dan encoder yang dipakai oleh setiap ekspor. Sertakan input, kedua logit referensi, kelas prediksi referensi, dan label dataset per sampel. Satu sampel dapat memiliki beberapa file `.memh`.

A2 dan B2 menyepakati urutan piksel dan bobot, signedness, lebar accumulator, skala dan shift, pembulatan, saturasi, serta aturan pemilihan kelas saat logit seri. Gunakan pengaturan dari ekspor Python yang sebenarnya sebagai acuan. Heatmap 32×32 berisi 1.024 piksel. Payload piksel berukuran 1.024 byte jika formatnya satu byte per piksel. Sepakati juga framing UART dan format respons sebelum uji integrasi.

## Checkpoint Tim A

1. A1 menyiapkan proyek Quartus untuk DE10-Lite dan memastikan proyek sederhana berhasil dikompilasi.
2. A2 menjalankan ulang inferensi fixed-point Python dengan satu golden vector dan menyimpan hasilnya.
3. A1 memasukkan satu golden vector ke testbench CNN. A1 dan A2 mencocokkan kedua logit dan kelas dengan Python. Setelah cocok, jalankan seluruh vektor dan catat perbedaan.
4. Uji UART dan inferensi pada board dengan input yang sama. Simpan hasil uji, penggunaan resource, dan latensi.

Bandingkan RTL dengan prediksi fixed-point untuk menilai kesetaraan implementasi. Gunakan label dataset untuk menghitung akurasi deteksi.
