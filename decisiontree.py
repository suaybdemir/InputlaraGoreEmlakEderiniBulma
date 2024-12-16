import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, \
    confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Veri setini oluşturun
data = pd.read_excel("burdur_emlak_final.xlsx")

pd.set_option('display.max_columns', None)
# Veriyi DataFrame'e dönüştürün
df = pd.DataFrame(data)
df = df.select_dtypes(include=[np.number])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.duplicated()]

# Bağımsız ve bağımlı değişkenleri ayıralım
X = df[["m2_brut","m2_net", "bina_yasi", "bulundugu_kat", "kat_sayisi","balkon","esyali","site","oda","salon"]]
y = df["fiyat"]


# Decision Tree Regressor Modeli için Fonksiyon
def decision_tree_regressor(X_train, X_test, y_train, y_test):
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    # Tahminler
    y_pred_tree = decision_tree_model.predict(X_test)

    # Model Performansı
    print("Decision Tree Modeli Performansı:")
    print("R2 Skoru:", r2_score(y_test, y_pred_tree))
    print("Ortalama Kare Hatası (MSE):", mean_squared_error(y_test, y_pred_tree))

    # Gerçek ve Tahmin Edilen Fiyatları Görselleştir
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_tree, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dotted')
    plt.title("Decision Tree - Gerçek vs Tahmin Edilen Fiyatlar")
    plt.xlabel("Gerçek Fiyatlar")
    plt.ylabel("Tahmin Edilen Fiyatlar")
    plt.show()

    return decision_tree_model


# Yeni Ev Tahmini
def predict_new_house(decision_tree_model):
    # Yeni Ev için Örnek Veri
    required_columns = X.columns.tolist()
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
    })
    # Tahmin
    yeni_ev_fiyat_tahmini_tree = decision_tree_model.predict(yeni_ev)
    print(f"\nYeni ev için Decision Tree ile tahmin edilen fiyat: {yeni_ev_fiyat_tahmini_tree[0]:,.2f} TL")


# Price Kategorize Etme ve Random Forest Classifier Modeli için Fonksiyon
def random_forest_classifier(X_train, X_test, y_train, y_test, y_class):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Predict on the test data
    y_pred_class = classifier.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # Precision, Recall, Accuracy
    precision = precision_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
    recall = recall_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class))
    accuracy = accuracy_score(y_test, y_pred_class)

    # Specificity (True Negative Rate)
    specificity = cm.diagonal() / cm.sum(axis=1)

    # Print Results
    print(f"\nRandom Forest Classifier Performansı:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Specificity: {specificity}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature Importance
    feature_importance = classifier.feature_importances_
    importance_df = pd.DataFrame({
        'Özellik': X.columns,
        'Önem': feature_importance
    }).sort_values(by='Önem', ascending=False)

    print("\nÖzelliklerin Önemi:")
    print(importance_df)

    return classifier


# Veri setini eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
decision_tree_model = decision_tree_regressor(X_train, X_test, y_train, y_test)

# Yeni Ev Fiyat Tahmini
predict_new_house(decision_tree_model)

# Convert continuous target variable to categories (low, medium, high)
price_threshold_low = y.quantile(0.33)  # 33rd percentile (Low category)
price_threshold_high = y.quantile(0.66)  # 66th percentile (High category)

# Create a new target variable with categorical values
y_class = pd.cut(y, bins=[0, price_threshold_low, price_threshold_high, np.inf], labels=['Low', 'Medium', 'High'])

# Split the data again with the new classification target
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Random Forest Classifier
random_forest_classifier(X_train, X_test, y_train, y_test, y_class)
