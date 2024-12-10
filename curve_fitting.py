import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Veri setini oluşturun
data = {
    "Oda Sayısı": [1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7],
    "Banyo Sayısı": [1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6],
    "Şehir Merkezi Mesafesi (km)": [10, 8, 6, 4, 3, 3, 2, 1, 1, 0.5, 0.2],
    "Fiyat": [200000, 250000, 300000, 400000, 500000, 700000, 900000, 1100000, 1300000, 1500000, 2500000],
    "Metrekare": [50, 70, 90, 120, 150, 200, 250, 300, 350, 400, 400]
}

# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)

# Bağımsız değişkenlerin isimlerini alalım
independent_columns = ["Oda Sayısı", "Banyo Sayısı", "Şehir Merkezi Mesafesi (km)", "Metrekare"]


# Eğri uyumlama için bir polinom fonksiyonu tanımlayalım (2. derece polinom)
def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


# Her bağımsız değişken için eğri uyumlama işlemi yapalım
for column in independent_columns:
    X = df[column]
    y = df["Fiyat"]

    # curve_fit fonksiyonu ile parametreleri tahmin edelim
    params, covariance = curve_fit(poly2, X, y)

    # Tahmin edilen parametreler
    a, b, c = params
    print(f"{column} için eğri fit parametreleri: a={a:.2f}, b={b:.2f}, c={c:.2f}")

    # Fit edilen eğriyi hesaplayalım
    y_fit = poly2(X, a, b, c)

    # Veriyi ve uyumlu eğriyi görselleştirelim
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Veriler")  # Gerçek veriler
    plt.plot(X, y_fit, color="red", label="Eğri Uyumlama")  # Fit edilen eğri
    plt.xlabel(column)
    plt.ylabel("Fiyat")
    plt.title(f"{column} ile Fiyat Arasındaki İlişki (Curve Fitting)")
    plt.legend()
    plt.show()

    # Yeni bir ev için tahmin yapalım (örneğin, Metrekare = 120, Oda Sayısı = 5, vb.)
    yeni_ev_degeri = X.mean()  # Ortalama değeri alabiliriz ya da spesifik bir değer girilebilir
    yeni_ev_fiyat_tahmini = poly2(yeni_ev_degeri, *params)
    print(f"Yeni ev için {column} değeri {yeni_ev_degeri} ile tahmin edilen fiyat: {yeni_ev_fiyat_tahmini:,.2f} TL\n")
