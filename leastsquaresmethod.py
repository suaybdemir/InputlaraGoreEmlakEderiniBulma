# Gerekli kütüphaneleri import edin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# İnterpolasyon yaparak eksik metrekare değerini doldurun (lineer interpolasyon)
df["Metrekare"] = df["Metrekare"].interpolate(method='linear')

# Korelasyon matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# Bağımsız ve bağımlı değişkenleri ayırın
X = df.drop(columns=["Fiyat"])
y = df["Fiyat"]

# Veri setini eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Dereceden 5. Dereceye kadar polinomsal regresyon ve En Küçük Kareler Yöntemi
for degree in range(1, 6):
    # Polinomsal dönüşüm uygulayın
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # En Küçük Kareler (Least Squares) yöntemi ile polinomsal regresyon modeli oluşturun
    poly_model = LinearRegression(fit_intercept=True)
    poly_model.fit(X_train_poly, y_train)

    # Modelin performansını değerlendirin
    y_pred_poly = poly_model.predict(X_test_poly)

    print(f"\n{degree}. Dereceden Polinomsal Regresyon (En Küçük Kareler Yöntemi):")
    print("R2 Skoru:", r2_score(y_test, y_pred_poly))
    print("Ortalama Kare Hatası (MSE):", mean_squared_error(y_test, y_pred_poly))

    # Gerçek vs Tahmin değerleri
    results_df_poly = pd.DataFrame({"Gerçek Değerler": y_test, "Tahmin Değerler": y_pred_poly})
    print("\nGerçek ve Tahmin Değerleri Karşılaştırması:")
    print(results_df_poly)

    # Gerçek ve tahmin edilen değerleri görselleştirin
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_poly, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
    plt.xlabel("Gerçek Fiyatlar")
    plt.ylabel("Tahmin Fiyatlar")
    plt.title(f"Gerçek vs Tahmin Edilen Fiyatlar ({degree}. Derece Polinomsal Regresyon)")
    plt.show()

    # Yeni bir ev için tahmin yapalım
    yeni_ev = pd.DataFrame({
        "Oda Sayısı": [5],
        "Banyo Sayısı": [4],
        "Şehir Merkezi Mesafesi (km)": [3],
        "Metrekare": [120],  # Burada interpolasyonla hesaplanan değer kullanılabilir
    })

    yeni_ev_poly = poly.transform(yeni_ev)  # Polinomsal dönüşüm
    yeni_ev_fiyat_tahmini_poly = poly_model.predict(yeni_ev_poly)
    print(f"Yeni ev için tahmin edilen fiyat ({degree}. Derece): {yeni_ev_fiyat_tahmini_poly[0]:,.2f} TL")
