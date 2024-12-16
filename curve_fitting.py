import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Veri setini oluşturun
data = pd.read_excel("burdur_emlak_final.xlsx")

# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)
df = df.select_dtypes(include=[np.number])

# Bağımsız değişkenlerin isimlerini alalım
independent_columns = ["m2_brut","m2_net", "bina_yasi", "bulundugu_kat", "kat_sayisi","balkon","esyali","site","oda","salon"]

# Modelin performansını değerlendirecek fonksiyon
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, r2, mae

# Polinom fonksiyonu tanımlayalım (2. derece)
def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c

# Her bağımsız değişken için eğri uyumlama işlemi yapalım
for column in independent_columns:
    X = df[column].values.reshape(-1, 1)  # Veriyi uygun şekilde şekillendirelim
    y = df["fiyat"]

    # curve_fit fonksiyonu ile parametreleri tahmin edelim
    try:
        params, covariance = curve_fit(poly2, X.flatten(), y)
    except Exception as e:
        print(f"Error fitting curve for {column}: {e}")
        continue

    # Tahmin edilen parametreler
    a, b, c = params
    print(f"{column} için eğri fit parametreleri: a={a:.2f}, b={b:.2f}, c={c:.2f}")

    # Fit edilen eğriyi hesaplayalım
    y_fit = poly2(X, a, b, c)

    # Modelin performansını değerlendirelim
    mse, r2, mae = evaluate_model(y, y_fit)
    print(f"Model Performansı ({column}):")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2 Score: {r2:.2f}")
    print(f"  MAE: {mae:.2f}")

    # Veriyi ve uyumlu eğriyi görselleştirelim
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Veriler")  # Gerçek veriler
    plt.plot(X, y_fit, color="red", label="Eğri Uyumlama")  # Fit edilen eğri
    plt.xlabel(column)
    plt.ylabel("Fiyat")
    plt.title(f"{column} ile Fiyat Arasındaki İlişki (Curve Fitting)")
    plt.legend()
    plt.show()

    yeni_ev_degeri = X.mean()  # Ortalama değeri alabiliriz ya da spesifik bir değer girilebilir
    yeni_ev_fiyat_tahmini = poly2(yeni_ev_degeri, *params)
    print(f"Yeni ev için {column} değeri {yeni_ev_degeri} ile tahmin edilen fiyat: {yeni_ev_fiyat_tahmini:,.2f} TL\n")

# **Daha Genel Model: Polynomial Regression (Birden fazla bağımsız değişken ile)**

# Bağımsız değişkenlerin tümünü alalım
X = df[independent_columns]
y = df["fiyat"]

# Polinom özellikler oluşturmak için PolynomialFeatures kullanıyoruz
poly = PolynomialFeatures(degree=2)  # 2. dereceden polinomlar
X_poly = poly.fit_transform(X)

# Modeli eğitelim
model = LinearRegression()
model.fit(X_poly, y)

# Tahminler yapalım
y_pred = model.predict(X_poly)

# Model performansını değerlendirelim
mse, r2, mae = evaluate_model(y, y_pred)
print(f"Genel Model Performansı:")
print(f"  MSE: {mse:.2f}")
print(f"  R2 Score: {r2:.2f}")
print(f"  MAE: {mae:.2f}")

# Gerçek ve tahmin edilen değerleri görselleştirelim
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='dotted')
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Fiyatlar")
plt.title("Gerçek vs Tahmin Edilen Fiyatlar")
plt.show()

yeni_ev = np.array([[120, 100, 10, 3, 5, 1, 0, 1, 3, 1]])  # Örnek yeni ev özellikleri
yeni_ev_poly = poly.transform(yeni_ev)
tahmini_fiyat = model.predict(yeni_ev_poly)
print(f"Yeni ev için tahmin edilen kira fiyatı: {tahmini_fiyat[0]:,.2f} TL")