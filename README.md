# İşe Alım Tahmin API

Bu API, SVM (Support Vector Machine) modeli kullanarak işe alım tahminleri yapar. Model, adayların tecrübe yılı ve teknik puanlarına göre işe alınıp alınmayacaklarını tahmin eder.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```bash
uvicorn main:app --reload
```

## API Kullanımı

### Tahmin Yapma

**Endpoint:** `/tahmin`

**Method:** POST

**Request Body:**
```json
{
    "tecrube_yili": 5.0,
    "teknik_puan": 80.0
}
```

**Response:**
```json
{
    "tahmin": "İşe Alındı",
    "tecrube_yili": 5.0,
    "teknik_puan": 80.0
}
```

## API Dokümantasyonu

API dokümantasyonuna erişmek için:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 