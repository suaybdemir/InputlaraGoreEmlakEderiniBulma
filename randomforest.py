import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oluşturun
data = pd.read_excel("burdur_emlak_final.xlsx")

# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)
df = df.select_dtypes(include=[np.number])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.duplicated()]

# Bağımsız ve bağımlı değişkenleri ayıralım
X = df.drop(columns=["fiyat"])
y = df["fiyat"]

# Veri setini eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# **Random Forest Regressor Modeli**
# =========================================================

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Modelin tahminlerini yapalım
y_pred_rf = random_forest_model.predict(X_test)

# Modelin performansını değerlendirelim
print("Random Forest Modeli (Regresyon):")
print(f"R2 Skoru: {r2_score(y_test, y_pred_rf):.2f}")
print(f"Ortalama Kare Hatası (MSE): {mean_squared_error(y_test, y_pred_rf):.2f}")

# Gerçek ve tahmin edilen değerleri görselleştirelim
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
plt.title("Random Forest - Gerçek vs Tahmin Edilen Fiyatlar (Regresyon)")
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.show()

# Yeni ev için tahmin yapalım
required_columns = X_train.columns.tolist()
yeni_ev = pd.DataFrame({
        "m2_brut": [120],
        "m2_net": [100],
        "bina_yasi": [10],
        "bulundugu_kat": [3],
        "kat_sayisi": [5],
        "balkon": [1],
        "esyali": [0],
        "site": [1],
        "oda": [3],
        "salon": [1]
    })[required_columns]

yeni_ev_fiyat_tahmini_rf = random_forest_model.predict(yeni_ev)
print(f"\nYeni ev için Random Forest ile tahmin edilen fiyat: {yeni_ev_fiyat_tahmini_rf[0]:,.2f} TL")

# =========================================================
# **Convert continuous target variable to categories (low, medium, high)**
# =========================================================

# Kategorik hedef değişkeni oluşturuyoruz
price_threshold_low = y.quantile(0.33)  # 33. yüzdelik (Düşük kategori)
price_threshold_high = y.quantile(0.66)  # 66. yüzdelik (Yüksek kategori)

# Yeni hedef değişkeni
y_class = pd.cut(y, bins=[0, price_threshold_low, price_threshold_high, np.inf], labels=['Low', 'Medium', 'High'])

# Veriyi yeniden bölüyoruz (Yeni sınıflandırma hedefi ile)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# =========================================================
# **Random Forest Classifier Modeli**
# =========================================================

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapalım
y_pred_class = classifier.predict(X_test)

# Precision, Recall (Sensitivity), Accuracy hesaplayalım
precision = precision_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
recall = recall_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
accuracy = accuracy_score(y_test, y_pred_class)

# Confusion Matrix (Karışıklık Matrisi)
cm = confusion_matrix(y_test, y_pred_class)

# Specificity (True Negative Rate) her sınıf için hesaplanır
specificity = cm.diagonal() / cm.sum(axis=1)

# Sonuçları yazdıralım
print("\nRandom Forest Modeli (Sınıflandırma):")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity}")

# =========================================================
# **Confusion Matrix Görselleştirme**
# =========================================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix (Sınıflandırma)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
