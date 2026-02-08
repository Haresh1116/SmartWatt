"""
Smart Watt Production V2 ML Pipeline
-----------------------------------
• Pull last 6 months data from Supabase
• Clean + Feature engineer
• Train multi-model ensemble (SARIMA + XGB + Linear + Meta)
• Save models locally
• Register model metadata
• Generate Day / Week / Month forecasts
• Push predictions to Supabase
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from supabase import create_client

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

# =============================
# CONFIG
# =============================

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_VERSION = "v2.0.0"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_24h",
    "lag_7d",
    "rolling_mean_3d",
    "rolling_mean_7d"
]

# =============================
# PIPELINE CLASS
# =============================

class SmartWattV2:

    def __init__(self, url, key, sensor_id="main"):
        self.supabase = create_client(url, key)
        self.sensor_id = sensor_id

    # =============================
    # DATA PULL
    # =============================
    def pull_data(self):
        print("📥 Pulling last 6 months data...")

        start_date = datetime.now() - timedelta(days=180)

        res = (
            self.supabase
            .table("appliance_usage")
            .select("*")
            .gte("timestamp", start_date.isoformat())
            .execute()
        )

        if not res.data:
            raise Exception("No data found")

        df = pd.DataFrame(res.data)
        print(f"✅ Pulled {len(df)} rows")

        return df

    # =============================
    # CLEANING
    # =============================
    def clean(self, df):

        print("🧹 Cleaning data...")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        df = df.dropna(subset=["power"])
        df = df[df["power"] >= 0]

        q95 = df["power"].quantile(0.95)
        df = df[df["power"] <= q95 * 1.5]

        df = df.drop_duplicates(subset=["timestamp"])

        df = df.set_index("timestamp").resample("1H").mean()

        df = df.fillna(method="ffill").fillna(method="bfill")

        df = df.reset_index()

        print("✅ Clean complete")

        return df

    # =============================
    # FEATURES
    # =============================
    def features(self, df):

        print("⚙️ Building features...")

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["month"] = df["timestamp"].dt.month

        df["lag_24h"] = df["power"].shift(24)
        df["lag_7d"] = df["power"].shift(168)

        df["rolling_mean_3d"] = df["power"].rolling(72).mean()
        df["rolling_mean_7d"] = df["power"].rolling(168).mean()

        df = df.fillna(method="bfill").fillna(method="ffill")

        print("✅ Feature build complete")

        return df

    # =============================
    # TRAIN MODELS
    # =============================
    def train_models(self, df):

        print("🤖 Training models...")

        y = df["power"].values
        X = df[FEATURE_COLS].values

        split = int(len(df)*0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ---- SARIMA ----
        print("Training SARIMA...")
        sarima = SARIMAX(
            y_train,
            order=(1,1,1),
            seasonal_order=(1,1,1,24)
        ).fit(disp=False)

        # ---- XGB ----
        print("Training Gradient Boosting...")
        xgb = GradientBoostingRegressor()
        xgb.fit(X_train, y_train)

        # ---- Linear ----
        print("Training Linear...")
        lin = LinearRegression()
        lin.fit(X_train, y_train)

        # ---- Stacking Meta Model ----
        print("Training Meta Model...")

        sarima_train = sarima.fittedvalues
        xgb_train = xgb.predict(X_train)
        lin_train = lin.predict(X_train)

        stack_X = np.vstack([sarima_train, xgb_train, lin_train]).T

        meta = LinearRegression()
        meta.fit(stack_X, y_train)

        # Evaluate
        sarima_test = sarima.forecast(len(y_test))
        xgb_test = xgb.predict(X_test)
        lin_test = lin.predict(X_test)

        stack_test = np.vstack([sarima_test, xgb_test, lin_test]).T
        final_test = meta.predict(stack_test)

        rmse = np.sqrt(mean_squared_error(y_test, final_test))
        r2 = r2_score(y_test, final_test)

        print(f"✅ RMSE {rmse:.2f} | R2 {r2:.3f}")

        models = dict(
            sarima=sarima,
            xgb=xgb,
            linear=lin,
            meta=meta,
            metrics=dict(rmse=rmse, r2=r2)
        )

        return models

    # =============================
    # SAVE MODELS
    # =============================
    def save_models(self, models):

        print("💾 Saving models...")

        for name, model in models.items():
            if name == "metrics":
                continue
            joblib.dump(model, f"{MODEL_DIR}/{name}_{MODEL_VERSION}.pkl")

        print("✅ Models saved")

    # =============================
    # FORECAST
    # =============================
    def forecast(self, df, models):

        print("🔮 Generating forecasts...")

        sarima = models["sarima"]
        xgb = models["xgb"]
        lin = models["linear"]
        meta = models["meta"]

        latest = df[FEATURE_COLS].tail(1).values

        def ensemble_step(sar, xg, li):
            return meta.predict(np.array([[sar,xg,li]]))[0]

        # ---- Next Day ----
        sar_day = sarima.forecast(24).sum()
        xgb_day = xgb.predict(latest)[0] * 24
        lin_day = lin.predict(latest)[0] * 24

        day_pred = ensemble_step(sar_day, xgb_day, lin_day)

        # ---- Next Week ----
        sar_week = sarima.forecast(168).sum()
        xgb_week = xgb_day * 7
        lin_week = lin_day * 7

        week_pred = ensemble_step(sar_week, xgb_week, lin_week)

        # ---- Next Month ----
        sar_month = sarima.forecast(720).sum()
        xgb_month = xgb_day * 30
        lin_month = lin_day * 30

        month_pred = ensemble_step(sar_month, xgb_month, lin_month)

        return dict(
            day=day_pred,
            week=week_pred,
            month=month_pred
        )

    # =============================
    # STORE PREDICTIONS
    # =============================
    def push_predictions(self, preds):

        print("📤 Pushing predictions...")

        now = datetime.now()

        for k,v in preds.items():

            payload = dict(
                sensor_id=self.sensor_id,
                prediction_type=k,
                predicted_kwh=float(max(0,v)),
                model_version=MODEL_VERSION,
                created_at=now.isoformat()
            )

            self.supabase.table("predictions").insert(payload).execute()

        print("✅ Predictions stored")

    # =============================
    # RUN PIPELINE
    # =============================
    def run(self):

        df = self.pull_data()
        df = self.clean(df)
        df = self.features(df)

        models = self.train_models(df)

        self.save_models(models)

        preds = self.forecast(df, models)

        self.push_predictions(preds)

        print("🎉 PIPELINE COMPLETE")


# =============================
# ENTRY POINT
# =============================

if __name__ == "__main__":

    URL = os.getenv("SUPABASE_URL")
    KEY = os.getenv("SUPABASE_KEY")

    if not URL or not KEY:
        raise Exception("Missing Supabase credentials")

    SmartWattV2(URL, KEY).run()
