from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained pipeline model
model = joblib.load("xgb_model.pkl")


class SalesInput(BaseModel):
    Row_ID: int
    Order_ID: str
    Order_Date: str
    Ship_Date: str
    Ship_Mode: str
    Customer_ID: str
    Customer_Name: str
    Segment: str
    Country: str
    City: str
    State: str
    Postal_Code: int
    Region: str
    Product_ID: str
    Category: str
    Sub_Category: str
    Product_Name: str
    Quantity: int
    Discount: float
    Profit: float


@app.get("/")
def home():
    return {"message": "Sales Prediction API Running"}


@app.post("/predict")
def predict(data: SalesInput):

    df = pd.DataFrame([{
        "Row ID": data.Row_ID,
        "Order ID": data.Order_ID,
        "Order Date": data.Order_Date,
        "Ship Date": data.Ship_Date,
        "Ship Mode": data.Ship_Mode,
        "Customer ID": data.Customer_ID,
        "Customer Name": data.Customer_Name,
        "Segment": data.Segment,
        "Country": data.Country,
        "City": data.City,
        "State": data.State,
        "Postal Code": data.Postal_Code,
        "Region": data.Region,
        "Product ID": data.Product_ID,
        "Category": data.Category,
        "Sub-Category": data.Sub_Category,
        "Product Name": data.Product_Name,
        "Quantity": data.Quantity,
        "Discount": data.Discount,
        "Profit": data.Profit
    }])

    prediction = model.predict(df)

    return {
        "Predicted_Sales": float(prediction[0])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
