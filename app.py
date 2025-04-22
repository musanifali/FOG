from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import StringIO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Change as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and imputer
try:
    model = joblib.load("naive_bayes_model.pkl")
    imputer = joblib.load("imputer.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e.filename}")

def predict_fog_from_csv(df: pd.DataFrame):
    # Drop target column if uploaded CSV includes it
    if 'FoG' in df.columns:
        df = df.drop(columns=['FoG'])

    # Impute missing values
    data_imputed = imputer.transform(df)

    # Predict using the loaded model
    predictions = model.predict(data_imputed)

    # You can also return all predictions if multiple rows
    return ["FOG Detected" if pred == 1 else "No FOG" for pred in predictions]

@app.post("/predict_fog")
async def predict_fog(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return {"error": "Only CSV files are allowed."}

    if file.content_type != 'text/csv':
        return {"error": "File must be of type text/csv."}

    try:
        content = await file.read()
        stringio = StringIO(content.decode('utf-8'))
        df = pd.read_csv(stringio)
        results = predict_fog_from_csv(df)
        return {"predictions": results}
    except UnicodeDecodeError:
        return {"error": "File is not UTF-8 encoded."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "API is running"}
