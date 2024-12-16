# Gerekli kütüphaneleri import edin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Veri setini oluşturun
data = pd.read_excel("burdur_emlak_final.xlsx")

# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)
df = df.select_dtypes(include=[np.number])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.duplicated()]
# İnterpolasyon yaparak eksik metrekare değerini doldurun (lineer interpolasyon)
#df["m2_net"] = df["m2_net"].interpolate(method='linear')

# Korelasyon matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# Bağımsız ve bağımlı değişkenleri ayıralım
X = df.drop(columns=["fiyat"])
y = df["fiyat"]

# Veri setini eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Dereceden 5. Dereceye kadar polinomsal regresyon ve En Küçük Kareler Yöntemi
for degree in range(1, 4):
    # Polinomsal dönüşüm uygulayın
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # En Küçük Kareler (Least Squares) yöntemi ile polinomsal regresyon modeli oluşturun
    poly_model = LinearRegression(fit_intercept=True)
    poly_model.fit(X_train_poly, y_train)

    # Modelin performansını değerlendirelim
    y_pred_poly = poly_model.predict(X_test_poly)

    print(f"\n{degree}. Dereceden Polinomsal Regresyon (En Küçük Kareler Yöntemi):")
    print("R2 Skoru:", r2_score(y_test, y_pred_poly))
    print("Ortalama Kare Hatası (MSE):", mean_squared_error(y_test, y_pred_poly))

    # Gerçek ve tahmin edilen değerleri görselleştirelim
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_poly, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
    plt.xlabel("Gerçek Fiyatlar")
    plt.ylabel("Tahmin Fiyatlar")
    plt.title(f"Gerçek vs Tahmin Edilen Fiyatlar ({degree}. Derece Polinomsal Regresyon)")
    plt.show()

    # Yeni Ev için Tahmin
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

    yeni_ev_poly = poly.transform(yeni_ev)  # Polinomsal dönüşüm
    yeni_ev_fiyat_tahmini_poly = poly_model.predict(yeni_ev_poly)
    print(f"Yeni ev için tahmin edilen fiyat ({degree}. Derece): {yeni_ev_fiyat_tahmini_poly[0]:,.2f} TL")

# Convert continuous target variable to categories (low, medium, high)
price_threshold_low = y.quantile(0.33)  # 33rd percentile (Low category)
price_threshold_high = y.quantile(0.66)  # 66th percentile (High category)

# Create a new target variable with categorical values
y_class = pd.cut(y, bins=[0, price_threshold_low, price_threshold_high, np.inf], labels=['Low', 'Medium', 'High'])

# Split the data again with the new classification target
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train a classifier (Random Forest for classification)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred_class = classifier.predict(X_test)

# Compute Precision, Recall (Sensitivity), Specificity, and Accuracy
precision = precision_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
recall = recall_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
accuracy = accuracy_score(y_test, y_pred_class)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)

# Calculate specificity (True Negative Rate) for each class
specificity = cm.diagonal() / cm.sum(axis=1)
# To get a global specificity (weighted average across classes):
class_support = cm.sum(axis=1)  # The number of samples in each class (row sum)
weighted_specificity = np.average(specificity, weights=class_support)

# Print the results
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity for each class: {specificity}")
print(f"Weighted Specificity: {weighted_specificity:.2f}")

# Optionally: Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
