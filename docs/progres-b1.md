# Progres B1 — Raspberry Pi, GPS, dan UART

B1 menyiapkan akuisisi GPS normal dan integrasi Raspberry Pi dengan FPGA DE10-Lite. Checkpoint berikut merangkum hasil yang sudah tercatat.

| Pekerjaan | Status |
| --- | --- |
| Raspberry Pi | Pi 4 (RAM 2 GB), microSD 32 GB, dan Raspberry Pi OS Lite 64-bit sudah disiapkan |
| Akses | SSH dari Windows melalui jaringan lokal pernah berhasil. Tailscale terpasang dan aksesnya pernah berhasil |
| UART | Port hardware aktif. `/dev/serial0` mengarah ke `ttyS0` pada Pi yang diuji. `python3-serial` terpasang |
| Penerimaan GPS/BeiDou | Pesan NMEA terbaca berulang pada 9600 baud, 8N1. Modul melaporkan `ANTENNA OK` |
| Validasi posisi | GGA menunjukkan fix 0 dan nol satelit digunakan. RMC berstatus `V`. Koordinat valid belum terverifikasi |
| Integrasi FPGA | Uji loopback 115200 baud, UART dua arah Pi ↔ DE10-Lite, dan hasil klasifikasi FPGA belum tercatat |

## Setup dan uji serial

Pada `sudo raspi-config`, buka **Interface Options → Serial Port**, pilih **No** untuk login shell lewat serial dan **Yes** untuk port serial hardware. Reboot Pi.

Untuk pembacaan satu arah, sambungkan TX modul GPS ke RX Pi (GPIO15, pin fisik 10) dan GND modul ke GND Pi (pin fisik 6). Sambungkan VCC sesuai spesifikasi modul. Level logika TX modul harus kompatibel dengan GPIO 3,3 V. RX dan PPS modul dibiarkan kosong saat uji baca. Tegangan suplai dan level TX aktual belum tercatat melalui pengukuran.

```bash
ls -l /dev/serial0
sudo apt update && sudo apt install -y python3-serial
sudo stty -F /dev/serial0 9600 raw -echo cs8 -parenb -cstopb -crtscts
sudo timeout 15 cat /dev/serial0
```

## Checkpoint berikutnya

1. Uji antena dengan pandangan terbuka ke langit. Konfirmasi kualitas fix GGA lebih dari 0 dan status RMC `A` dengan koordinat terisi.
2. Rekam posisi diam dengan interval sampel 30 detik setelah fix valid. Catat jumlah sampel valid sebelum menyerahkan data ke B2.
3. Uji TX Pi melalui loopback. Lanjutkan ke uji UART Pi ↔ DE10-Lite bersama Tim A. Gunakan input dan kelas referensi dari [serah terima B2](handoff-b2-fpga.md).

Sumber progres: dokumentasi B1 tanggal 11 November 2026 dan catatan pengujian berikutnya. Skrip asli dan rekaman GPS belum tersedia untuk diverifikasi di repo.
