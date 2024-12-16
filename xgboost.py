import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oluşturun
data = pd.read_excel("burdur_emlak_final.xlsx")
#data = pd.read_csv("antalya_kiralik_ev.csv")


pd.set_option('display.max_columns', None)
# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)

df = df.select_dtypes(include=[np.number])

# Korelasyon matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# =========================================================
# **Gradient Boosting Regressor (Regresyon)**
# =========================================================

# Bağımsız ve bağımlı değişkenleri ayıralım
X = df.drop(columns=["fiyat"])
y = df["fiyat"]

# Veri setini eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Regressor modelini kuruyoruz
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)

# Modelin performansını değerlendirelim
y_pred = gb_model.predict(X_test)

# R2 ve MSE hesaplayalım
print("\nGradient Boosting Modeli (Regresyon):")
print(f"R2 Skoru: {r2_score(y_test, y_pred):.2f}")
print(f"Ortalama Kare Hatası (MSE): {mean_squared_error(y_test, y_pred):.2f}")

# **Özellik önemlerini çıkaralım**
feature_importance = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': feature_importance
}).sort_values(by='Önem', ascending=True)

# Özelliklerin önemini görselleştirelim
plt.figure(figsize=(8, 6))
sns.barplot(x=importance_df["Önem"], y=importance_df["Özellik"], palette="viridis")
plt.title("Özellik Önemleri")
plt.xlabel("Önem Derecesi")
plt.ylabel("Özellikler")
plt.show()

# Gerçek vs Tahmin değerlerini görselleştirelim
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Fiyatlar")
plt.title("Gerçek vs Tahmin Edilen Fiyatlar")
plt.show()

# Yeni ev için tahmin yapalım
#required_columns = X_train.columns.tolist()
#yeni_ev = pd.DataFrame({
#    "m2_net": [120],
#    "bina_yasi": [5],
#    "bulundugu_kat": [3],
#    "kat_sayisi": [6],
#    "balkon": [1],
#    "esyali": [0],
#    "oda": [3],
#    "salon": [1]
#})[required_columns]

#yeni_ev_fiyat_tahmini = gb_model.predict(yeni_ev)
#print(f"Yeni ev için tahmin edilen fiyat: {yeni_ev_fiyat_tahmini[0]:,.2f} TL")

# =========================================================
# **Sınıflandırma (Random Forest Classifier)**
# =========================================================

# Fiyat aralıklarına göre kategorik hedef değişkeni oluşturuyoruz
price_threshold_low = y.quantile(0.33)  # Düşük kategori için eşik
price_threshold_high = y.quantile(0.66)  # Yüksek kategori için eşik

y_class = pd.cut(y, bins=[0, price_threshold_low, price_threshold_high, np.inf], labels=['Low', 'Medium', 'High'])

# Veriyi eğitim ve test olarak ayıralım (sınıflandırma için)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Random Forest Classifier modelini kuruyoruz
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapalım
y_pred_class = classifier.predict(X_test)

# Model performansını değerlendirelim
precision = precision_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
recall = recall_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
accuracy = accuracy_score(y_test, y_pred_class)

# Karışıklık matrisi (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred_class)

# Specificity (True Negative Rate) hesaplayalım
specificity = cm.diagonal() / cm.sum(axis=1)

# Sonuçları yazdıralım
print("\nRandom Forest Modeli (Sınıflandırma):")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity}")

# **Confusion Matrix Görselleştirme**
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix (Sınıflandırma)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
