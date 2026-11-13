# LSI Design 2027 — GPS Spoofing Detection

Proyek ini mendeteksi anomali lintasan GPS dengan CNN ringan. Raspberry Pi mengubah data GPS menjadi heatmap grayscale 32×32. Target implementasi CNN adalah Terasic DE10-Lite (Intel MAX 10).

| Tim | Tanggung jawab |
| --- | --- |
| A1 | RTL, testbench, sintesis, dan pengujian DE10-Lite |
| A2 | Kuantisasi, verifikasi aritmetika RTL terhadap Python, dan evaluasi FPGA |
| B1 | Akuisisi GPS, Raspberry Pi, dan integrasi UART |
| B2 | Dataset, encoder, pelatihan CNN, golden vectors, dan validasi |

Repo ini berisi dokumentasi awal. Kode, model, dan vektor uji dari proyek sebelumnya perlu disalin dari folder kerja masing-masing tim.

- [Progres dan serah terima B2 ke Tim A](docs/handoff-b2-fpga.md)
- [Panduan mengisi dan mengunggah repo](docs/setup.md)
