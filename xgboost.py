# Gerekli kütüphaneleri import edin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

# Gradient Boosting Regressor modeli oluşturun
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)

# Modelin performansını değerlendirin
y_pred = gb_model.predict(X_test)

print("R2 Skoru:", r2_score(y_test, y_pred))
print("Ortalama Kare Hatası (MSE):", mean_squared_error(y_test, y_pred))

# Özellik önemlerini çıkarın
feature_importance = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': feature_importance
}).sort_values(by='Önem', ascending=True)

print("\nÖzellik Önemleri:")
print(importance_df)

# Özellik önemlerini görselleştirin
plt.figure(figsize=(8, 6))
sns.barplot(x=importance_df["Önem"], y=importance_df["Özellik"], palette="viridis")
plt.title("Özellik Önemleri")
plt.show()

# Gerçek vs Tahmin değerleri
results_df = pd.DataFrame({"Gerçek Değerler": y_test, "Tahmin Değerler": y_pred})
print("\nGerçek ve Tahmin Değerleri Karşılaştırması:")
print(results_df)

# Gerçek ve tahmin edilen değerleri görselleştirin
plt.figure(figsize=(8, 6))
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(y_test, y_pred, alpha=0.7, color='blue',s=10,cmap='hsv')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Fiyatlar")
plt.title("Gerçek vs Tahmin Edilen Fiyatlar")
plt.show()

yeni_ev = pd.DataFrame({
    "Oda Sayısı": [5],
    "Banyo Sayısı": [4],
    "Şehir Merkezi Mesafesi (km)": [3],
    "Metrekare": [120],
})

yeni_ev_fiyat_tahmini = gb_model.predict(yeni_ev)
print(f"Yeni ev için tahmin edilen fiyat: {yeni_ev_fiyat_tahmini[0]:,.2f} TL")
