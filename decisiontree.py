import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Bağımsız ve bağımlı değişkenleri ayırın
X = df.drop(columns=["Fiyat"])
y = df["Fiyat"]

# Veri setini eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor modeli oluşturun
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Modelin tahminlerini yapalım
y_pred_tree = decision_tree_model.predict(X_test)

# Modelin performansını değerlendirelim
print("Decision Tree Modeli:")
print("R2 Skoru:", r2_score(y_test, y_pred_tree))
print("Ortalama Kare Hatası (MSE):", mean_squared_error(y_test, y_pred_tree))

# Gerçek ve tahmin edilen değerleri görselleştirelim
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tree, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
plt.title("Decision Tree - Gerçek vs Tahmin Edilen Fiyatlar")
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.show()

# Yeni bir ev için tahmin yapalım
yeni_ev = pd.DataFrame({
    "Oda Sayısı": [5],
    "Banyo Sayısı": [4],
    "Şehir Merkezi Mesafesi (km)": [3],
    "Metrekare": [120],
})

yeni_ev_fiyat_tahmini_tree = decision_tree_model.predict(yeni_ev)

print(f"\nYeni ev için Decision Tree ile tahmin edilen fiyat: {yeni_ev_fiyat_tahmini_tree[0]:,.2f} TL")
