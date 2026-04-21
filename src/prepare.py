# =========================
# DATA COLLECTION + PREPARE
# =========================

import pandas as pd
import os

print("Memuat data mentah...")

# =========================
# DATA COLLECTION
# =========================
# Ambil data dari folder raw
df_valid = pd.read_csv("data/raw/Cleaned_Kompas_v2 valid.csv")
df_hoax = pd.read_csv("data/raw/Cleaned_TurnBackHoax_v3.csv")

print("Data berhasil dibaca")

# Cek kolom
print("\nKolom valid:")
print(df_valid.columns.tolist())

print("\nKolom hoax:")
print(df_hoax.columns.tolist())

# =========================
# LABELING
# =========================
df_valid["label"] = 0   # berita valid
df_hoax["label"] = 1    # hoaks

# =========================
# GABUNGKAN DATA
# =========================
df = pd.concat([df_valid, df_hoax], ignore_index=True)

# =========================
# PILIH KOLOM TEKS
# =========================
# Karena dataset kamu punya clean_text
df = df[["clean_text", "label"]]

# Rename biar konsisten
df = df.rename(columns={
    "clean_text": "text"
})

# Hapus data kosong
df = df.dropna()

# Hapus duplikat (opsional bagus untuk prepare)
df = df.drop_duplicates()

print("\nJumlah data akhir:", len(df))

# =========================
# SIMPAN HASIL
# =========================
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/clean.csv", index=False)

print("\nData collection + prepare selesai!")
print("Output: data/processed/clean.csv")