# =========================
# DATA PREPROCESSING
# =========================

import pandas as pd

print("Memuat data...")

# Load hasil prepare
df = pd.read_csv("data/processed/clean.csv")

# =========================
# CEK DATA
# =========================
print("Jumlah data awal:", len(df))

# =========================
# HAPUS DUPLIKAT
# =========================
df = df.drop_duplicates()

# =========================
# HAPUS NILAI KOSONG
# =========================
df = df.dropna()

# =========================
# LOWERCASE
# =========================
df["text"] = df["text"].str.lower()

# =========================
# HAPUS SPASI BERLEBIH
# =========================
df["text"] = df["text"].str.strip()

# =========================
# CEK HASIL
# =========================
print("Jumlah data setelah preprocessing:", len(df))

# =========================
# SIMPAN
# =========================
df.to_csv("data/processed/preprocessed.csv", index=False)

print("Preprocessing selesai!")