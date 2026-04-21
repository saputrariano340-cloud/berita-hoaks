# =========================
# EXPLORATORY DATA ANALYSIS
# =========================

import pandas as pd
import matplotlib.pyplot as plt

print("Memuat data hasil prepare...")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/clean.csv")

print("Data berhasil dimuat")

# =========================
# UKURAN DATA
# =========================
print("\nUkuran dataset:")
print(df.shape)

# =========================
# INFO DATA
# =========================
print("\nInfo dataset:")
print(df.info())

# =========================
# CEK MISSING VALUE
# =========================
print("\nMissing values:")
print(df.isnull().sum())

# =========================
# DISTRIBUSI LABEL
# =========================
print("\nDistribusi label:")
print(df["label"].value_counts())

df["label"].value_counts().plot(kind="bar")

plt.title("Distribusi Berita Valid vs Hoaks")
plt.xlabel("Label (0=Valid, 1=Hoaks)")
plt.ylabel("Jumlah Data")
plt.show()

# =========================
# PANJANG TEKS
# =========================
df["text_length"] = df["text"].astype(str).apply(len)

print("\nStatistik panjang teks:")
print(df["text_length"].describe())

df["text_length"].hist()

plt.title("Distribusi Panjang Teks")
plt.xlabel("Jumlah Karakter")
plt.ylabel("Frekuensi")
plt.show()

print("\nEDA selesai!")