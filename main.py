from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="İşe Alım Tahmin API",
    description="SVM tabanlı işe alım tahmin API'si",
    version="1.0.0"
)

# Model ve scaler'ı yükledik
try:
    model = joblib.load('svm_model_rbf.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    print(f"Model veya scaler yüklenirken hata oluştu: {e}")

class PredictionInput(BaseModel):
    tecrube_yili: float
    teknik_puan: float

class PredictionOutput(BaseModel):
    tahmin: str
    tecrube_yili: float
    teknik_puan: float

@app.get("/")
async def root():
    return {"mesaj": "İşe Alım Tahmin API'sine Hoş Geldiniz"}

@app.post("/tahmin", response_model=PredictionOutput)
async def tahmin_yap(input_data: PredictionInput):
    try:
        # Girdiyi numpy dizisine dönüştürdük
        input_array = np.array([[input_data.tecrube_yili, input_data.teknik_puan]])
        
        # Veriyi ölçekledik
        input_scaled = scaler.transform(input_array)
        
        # Tahmin yaptık
        tahmin = model.predict(input_scaled)[0]
        
        # Sonucu hazırladık
        sonuc = "İşe Alınmadı" if tahmin == 1 else "İşe Alındı"
        
        return PredictionOutput(
            tahmin=sonuc,
            tecrube_yili=input_data.tecrube_yili,
            teknik_puan=input_data.teknik_puan
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 

