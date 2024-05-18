# Mellomall.pi

Mellomall.pi adalah marketplace online yang dibangun menggunakan Raspberry Pi dan Python. Marketplace ini menyediakan fitur-fitur seperti chatbot AI, pratinjau produk AR dan VR, integrasi drone dan kendaraan autonom terbang, teknologi blockchain, dan antarmuka pengontrol suara.

# Prasyarat

1. Raspberry Pi 4
2. Kartu microSD dengan kapasitas minimal 16GB
3. Alimentasi Raspberry Pi
4. Monitor, keyboard, dan mouse
5. Koneksi internet

# Instalasi

1. Format kartu microSD dengan Raspberry Pi OS.
2. Salin file imej Raspberry Pi OS ke kartu microSD.
3. Masuk ke Raspberry Pi dan update sistem operasi.

```
1. sudo apt update
2. sudo apt upgrade
```

4. Install dependensi yang dibutuhkan.

```
1. sudo apt install python3 python3-pip
2. pip3 install flask flask-socketio flask-cors tensorflow tensorflow-gpu opencv-python-headless
```

5. Buat folder kerja untuk mellomall.pi.

```
1. mkdir ~/mellomall.pi
2. cd ~/mellomall.pi
```

6. Download file-file dari repo github ke folder kerja.

```
1. git clone https://github.com/KOSASIH/mellomall.pi.git
```

7. Buka file mellomall.py dengan teks editor Anda.

```
1. nano ~/mellomall.pi/mellomall.py
```

8. Edit alamat IP, port, dan direktori kerja pada baris-baris berikut.

```
1. IP = "0.0.0.0"
2. PORT = 8000
3. WORK_DIR = "/home/pi/mellomall.pi"
```

9. Jalankan server mellomall.pi.

```
1. python3 ~/mellomall.pi/mellomall.py
```

10. Akses

Mellomall.pi dapat diakses melalui alamat IP Raspberry Pi Anda pada port 8000. Misalnya, jika alamat IP Raspberry Pi Anda adalah 192.168.1.100, maka mellomall.pi dapat diakses melalui alamat http://192.168.1.100:8000.

Penting: Anda perlu menambahkan perizinan pada browser Anda untuk mengizinkan akses kamera dan mikrofon saat menggunakan antarmuka pengontrol suara.

# Teknologi yang Digunakan

- Python 3
- Flask
- Flask-SocketIO
- Flask-CORS
- TensorFlow
- TensorFlow-GPU
- OpenCV

# Catatan

Mellomall.pi belum dilengkapi dengan fitur-fitur seperti manajemen user, keamanan, dan pelaporan error. Oleh karena itu, sebelum menggunakan mellomall.pi untuk bisnis yang sersius, Anda perlu melakukan modifikasi dan peningkatan keamanan.
