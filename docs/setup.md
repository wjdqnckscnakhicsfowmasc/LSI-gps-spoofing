# Setup repositori

Struktur berikut diisi saat kode dan artefak asli tersedia.

| Folder | Isi |
| --- | --- |
| `software/` | Akuisisi GPS, encoder, training, dan inferensi fixed-point |
| `fpga/` | RTL, testbench, dan konfigurasi proyek Quartus |
| `artifacts/golden_vectors/` | Manifest, input, bobot, dan output referensi yang dipakai testbench |
| `docs/` | Progres, format antarmuka, dan hasil pengujian |

Salin kode asli dari `Documents\team_b_system` dan proyek Quartus Tim A. Simpan log koordinat asli dan dataset besar di luar repo. Folder dan nama file bisa disesuaikan dengan struktur proyek tim.

## Mulai bekerja dari repo GitHub

Beri anggota tim akses ke repo private melalui pengaturan GitHub. Setelah menerima akses, jalankan dari PowerShell:

```powershell
git clone https://github.com/wjdqnckscnakhicsfowmasc/LSI-design-.git
cd LSI-design-
git switch -c tambah-kode-tim
```

Salin file yang akan dibagikan ke folder yang sesuai. Periksa `git status`, tambahkan file yang dimaksud, lalu commit dan push branch. Buka pull request ke `main` untuk review tim.
