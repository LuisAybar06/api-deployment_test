from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from joblib import load

app = FastAPI()

@app.get("/health")
def read_root():
    return {"messeage": "Hello World"}


@app.post("/predict")
async def predict_banknote(file: UploadFile = File(...)):
    classifier = load("linear_regression.joblib")

    features_df = pd.read_csv("selected_features.csv")
    features = features_df["0"].to_list()

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    df = df[features]

    predictions = classifier.predict(df)

    return {
        "prediction": predictions.tolist()
    }