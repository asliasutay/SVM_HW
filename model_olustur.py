import numpy as np
import pandas as pd
from faker import Faker
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import seaborn as sns


# Faker nesnesi oluştur
fake = Faker()


# Veri üretme fonksiyonu
#Adayın toplam yazılım deneyimi (0–10 yıl arası)
#Teknik sınavdan alınan puan (0–100 arası)
#Tecrübesi 2 yıldan az ve sınav puanı 60'tan düşük olanlar işe alınmıyor.
def create_data(n=200):
    data = []
    for _ in range(n):
        experience = fake.random_int(min=0, max=10)
        technical_score = fake.random_int(min=0, max=100)
        label = 1 if (experience < 2 and technical_score < 60) else 0

        data.append({
            'tecrube_yili': experience,
            'teknik_puan': technical_score,
            'etiket': label
        })

    return pd.DataFrame(data)


# Veri setini oluştur
df = create_data()
df.info()
df.describe()

#veri seti görselleştirilmesi
alindi = df[df["etiket"] == 0]
alinmadi = df[df["etiket"] == 1]

plt.figure(figsize=(10, 6))

# Başarılı adaylar (etiket = 0)
plt.scatter(alindi["tecrube_yili"], alindi["teknik_puan"], color="green", label="İşe alındı", alpha=0.6)

# Başarısız adaylar (etiket = 1)
plt.scatter(alinmadi["tecrube_yili"], alinmadi["teknik_puan"], color="red", label="İşe alınmadı", alpha=0.6)

plt.xlabel("Tecrübe Yılı")
plt.ylabel("Teknik Puan")
plt.title("Adayların Tecrübe ve Teknik Puanlarına Göre İşe Alım Durumu")
plt.legend()
plt.grid(True)
plt.savefig('veri_gorsellestirme.png')
plt.show()

# Özellikler ve hedef değişken
X = df[['tecrube_yili', 'teknik_puan']]
y = df['etiket']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Veriyi StandardScaler ile ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model eğitimi
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Tahminler
y_pred = model.predict(X_test_scaled)


# Karar sınırını görselleştirme
def plot_decision_boundary():
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.8)
    plt.xlabel('Tecrübe Yılı')
    plt.ylabel('Teknik Puan')
    plt.title('SVM Karar Sınırı')
    plt.savefig('karar_siniri_linear.png')
    plt.close()

# Karar sınırını çiz
plot_decision_boundary()
'''Modelin karar sınırı mantıklı ve beklendiği gibi.
Karar sınırının eğimi, oluşturulan (2 yıldan az ve 60 puandan düşük) kurala 
benzer şekilde düşük tecrübeye ve düşük puana sahip adayları ayıracak biçimde eğilmiş.'''

# Kullanıcıdan girdi alıp tahmin yapma fonksiyonu
def prediction_function(tecrube_yili, teknik_puan):
    yeni_veri = [[tecrube_yili, teknik_puan]]
    tahmin = model.predict(yeni_veri)[0]
    return "İşe Alınmadı" if tahmin == 1 else "İşe Alındı"

# Örnek kullanım
if __name__ == "__main__":
    print("\nÖrnek Tahminler:")
    print("Tecrübe: 1 yıl, Teknik Puan: 35 ->", prediction_function(1, 35))
    print("Tecrübe: 5 yıl, Teknik Puan: 80 ->", prediction_function(5, 80))

'''Örnek Tahminler:
Tecrübe: 1 yıl, Teknik Puan: 35 -> İşe Alınmadı
Tecrübe: 5 yıl, Teknik Puan: 80 -> İşe Alındı'''

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Linear Kernel Doğruluğu: {accuracy}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
etiketler = ["İşe Alındı", "İşe Alınmadı"]

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=etiketler,
            yticklabels=etiketler)

plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig('confusion_matrix_linear.png')
plt.close()

# Classification Report
report = classification_report(y_test, y_pred, target_names=["İşe Alındı (0)", "İşe Alınmadı (1)"])
print("\nClassification Report:")
print(report)

# Kernel'i 'rbf' (Radial Basis Function) olarak değiştirme
model_rbf = SVC(kernel='rbf', C=1, gamma='scale')
model_rbf.fit(X_train_scaled, y_train)

# Tahminler
y_pred_rbf = model_rbf.predict(X_test_scaled)

# Karar sınırını görselleştir
def plot_decision_boundary_rbf():
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, alpha=0.8)
    plt.xlabel('Tecrübe Yılı')
    plt.ylabel('Teknik Puan')
    plt.title('SVM (RBF Kernel) Karar Sınırı')
    plt.savefig('karar_siniri_rbf.png')
    plt.close()

plot_decision_boundary_rbf()

# Parametreler
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# En iyi parametreler
print("En İyi Parametreler:", grid_search.best_params_)

# En iyi model
best_model = grid_search.best_estimator_

# Tahminler
y_pred_best = best_model.predict(X_test_scaled)

# Performans
accuracy = accuracy_score(y_test, y_pred_best)
print("En İyi Modelin Doğruluğu:", accuracy)

#son model
model_rbf = SVC(kernel='rbf', C=10, gamma='scale')
model_rbf.fit(X_train_scaled, y_train)

# Tahminler
y_pred_rbf = model_rbf.predict(X_test_scaled)

accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF Kernel Doğruluğu: {accuracy_rbf}")

# Modeli kaydetme
joblib.dump(model_rbf, 'svm_model_rbf.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model ve scaler başarıyla kaydedildi.") 
