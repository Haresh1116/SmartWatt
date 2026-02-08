"""
Smart Watt ML Ensemble Pipeline
Runs weekly to train models and generate predictions
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client, Client
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

class SmartWattML:
    def __init__(self, supabase_url: str, supabase_key: str, sensor_id: str = "main"):
        """Initialize ML pipeline with Supabase connection"""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.sensor_id = sensor_id
        self.model_version = "v1.0.0"
        self.scaler = MinMaxScaler()
        
    def pull_last_6_months_data(self) -> pd.DataFrame:
        """
        Pull raw sensor data from last 6 months from appliance_usage table
        """
        print("[Step 1] Pulling last 6 months data from Supabase...")
        
six_months_ago = datetime.now() - timedelta(days=180)
six_months_ago_str = six_months_ago.strftime('%Y-%m-%d')
        
        response = self.supabase.table('appliance_usage') \
            .select('*') \
            .gte('timestamp', six_months_ago_str) \
            .execute()
        
        if not response.data:
            raise ValueError("❌ No data found in last 6 months!")
        
        df = pd.DataFrame(response.data)
        print(f"✅ Pulled {len(df)} records from {six_months_ago_str} to now")
        return df
    
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Clean raw data
        - Remove nulls
        - Remove negatives
        - Handle duplicates
        - Remove extreme outliers
        """
        print("\n[Step 2] Cleaning data...")
        
        df = df.copy()
        
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
        
        initial_count = len(df)
        
df = df.dropna(subset=['power', 'energy_kwh'])
        print(f"  → Removed nulls: {initial_count - len(df)} rows")
        
        initial_count = len(df)
        df = df[(df['power'] >= 0) & (df['energy_kwh'] >= 0)]
        print(f"  → Removed negatives: {initial_count - len(df)} rows")
        
        Q95 = df['power'].quantile(0.95)
        initial_count = len(df)
        df = df[df['power'] <= Q95 * 1.5]
        print(f"  → Removed outliers: {initial_count - len(df)} rows")
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'sensor_id'])
        print(f"  → Removed duplicates: {initial_count - len(df)} rows")
        
        df = df.set_index('timestamp')
df = df.resample('1H').agg({
            'power': 'mean',
            'energy_kwh': 'sum',
            'voltage': 'mean',
            'current': 'mean',
            'sensor_id': 'first'
        }).reset_index()
        
df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Cleaned data: {len(df)} records remaining")
        return df
    
def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Feature Engineering
        Convert time-series → ML-friendly features
        """
        print("\n[Step 3] Feature engineering...")
        
        df = df.copy()
        
df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
df['lag_24h'] = df['power'].shift(24)
df['lag_7d'] = df['power'].shift(24 * 7)
        
df['rolling_mean_3d'] = df['power'].rolling(window=24*3, min_periods=1).mean()
df['rolling_mean_7d'] = df['power'].rolling(window=24*7, min_periods=1).mean()
df['rolling_std_7d'] = df['power'].rolling(window=24*7, min_periods=1).std()
        
df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Generated {len(df.columns)} features")
        return df
    
def train_ensemble_models(self, df: pd.DataFrame) -> dict:
        """
        Step 3: Train Ensemble Models
        - SARIMA (Seasonal)
        - XGBoost (Pattern)
        - Linear Regression (Stability)
        """
        print("\n[Step 4] Training ensemble models...")
        
        y = df['power'].values
        split_idx = int(len(df) * 0.8)
        
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        models = {}
        predictions_train = {}
        predictions_test = {}
        
        # Model 1: SARIMA (Seasonal)
        print("  → Training SARIMA model...")
        try:
            model_sarima = SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result_sarima = model_sarima.fit(disp=False)
            
            pred_train_sarima = result_sarima.fittedvalues.values
            pred_test_sarima = result_sarima.get_forecast(steps=len(y_test)).predicted_mean.values
            
            models['sarima'] = result_sarima
            predictions_train['sarima'] = pred_train_sarima
            predictions_test['sarima'] = pred_test_sarima
            print("    ✅ SARIMA trained")
        except Exception as e:
            print(f"    ⚠️ SARIMA failed: {e}, using fallback")
            predictions_train['sarima'] = y_train
            predictions_test['sarima'] = y_test
        
        # Model 2: XGBoost (Pattern)
        print("  → Training XGBoost model...")
        
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'month', 
                       'lag_24h', 'lag_7d', 'rolling_mean_3d', 'rolling_mean_7d']
        
        X = df[feature_cols].values
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        
        model_xgb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model_xgb.fit(X_train, y_train)
        
        pred_train_xgb = model_xgb.predict(X_train)
        pred_test_xgb = model_xgb.predict(X_test)
        
        models['xgboost'] = model_xgb
        predictions_train['xgboost'] = pred_train_xgb
        predictions_test['xgboost'] = pred_test_xgb
        print("    ✅ XGBoost trained")
        
        # Model 3: Linear Regression (Stability)
        print("  → Training Linear Regression model...")
        
        model_linear = LinearRegression()
        model_linear.fit(X_train, y_train)
        
        pred_train_linear = model_linear.predict(X_train)
        pred_test_linear = model_linear.predict(X_test)
        
        models['linear'] = model_linear
        predictions_train['linear'] = pred_train_linear
        predictions_test['linear'] = pred_test_linear
        print("    ✅ Linear Regression trained")
        
        # Ensemble Averaging
        print("  → Creating ensemble...")
        
        weights = {'sarima': 0.4, 'xgboost': 0.4, 'linear': 0.2}
        
        ensemble_train = (
            weights['sarima'] * predictions_train['sarima'] +
            weights['xgboost'] * predictions_train['xgboost'] +
            weights['linear'] * predictions_train['linear']
        )
        
        ensemble_test = (
            weights['sarima'] * predictions_test['sarima'] +
            weights['xgboost'] * predictions_test['xgboost'] +
            weights['linear'] * predictions_test['linear']
        )
        
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_test))
        r2 = r2_score(y_test, ensemble_test)
        
        print(f"    ✅ Ensemble created (RMSE: {rmse:.2f}, R²: {r2:.4f})")
        
        return {
            'models': models,
            'weights': weights,
            'ensemble_test': ensemble_test,
            'y_test': y_test,
            'metrics': {'rmse': rmse, 'r2': r2}
        }
    
def generate_predictions(self, df: pd.DataFrame, ensemble_models: dict) -> dict:
        """
        Step 4: Generate next week and next month predictions
        """
        print("\n[Step 5] Generating predictions...")
        
        X_latest = df[['hour', 'day_of_week', 'is_weekend', 'month', 
                       'lag_24h', 'lag_7d', 'rolling_mean_3d', 'rolling_mean_7d']].tail(1).values
        
        model_xgb = ensemble_models['models']['xgboost']
        model_linear = ensemble_models['models']['linear']
        weights = ensemble_models['weights']
        
        next_week_pred_xgb = model_xgb.predict(X_latest)[0]
        next_week_pred_linear = model_linear.predict(X_latest)[0]
        next_week_pred_sarima = ensemble_models['ensemble_test'].mean()
        
        next_week_prediction = (
            weights['sarima'] * next_week_pred_sarima +
            weights['xgboost'] * next_week_pred_xgb +
            weights['linear'] * next_week_pred_linear
        )
        
        next_month_prediction = next_week_prediction * 4.3
        
        confidence = max(0.5, min(1.0, ensemble_models['metrics']['r2']))
        
        predictions = {
            'next_week': {
                'predicted_kwh': max(0, next_week_prediction),
                'confidence_score': confidence,
                'prophet': next_week_pred_sarima,
                'xgboost': next_week_pred_xgb,
                'linear': next_week_pred_linear
            },
            'next_month': {
                'predicted_kwh': max(0, next_month_prediction),
                'confidence_score': confidence,
                'prophet': next_week_pred_sarima * 4.3,
                'xgboost': next_week_pred_xgb * 4.3,
                'linear': next_week_pred_linear * 4.3
            }
        }
        
        print(f"✅ Next Week Prediction: {predictions['next_week']['predicted_kwh']:.2f} kWh")
        print(f"✅ Next Month Prediction: {predictions['next_month']['predicted_kwh']:.2f} kWh")
        
        return predictions
    
def store_predictions_to_supabase(self, predictions: dict) -> bool:
        """
        Step 5: Store predictions in Supabase predictions table
        """
        print("\n[Step 6] Storing predictions in Supabase...")
        
        today = datetime.now().date()
        training_date = datetime.now()
        
        try:
            weekly_payload = {
                'sensor_id': self.sensor_id,
                'prediction_date': str(today + timedelta(days=7)),
                'prediction_type': 'weekly',
                'predicted_kwh': float(predictions['next_week']['predicted_kwh']),
                'confidence_score': float(predictions['next_week']['confidence_score']),
                'model_version': self.model_version,
                'prophet_prediction': float(predictions['next_week']['prophet']),
                'xgboost_prediction': float(predictions['next_week']['xgboost']),
                'linear_prediction': float(predictions['next_week']['linear']),
                'training_date': training_date.isoformat()
            }
            
            self.supabase.table('predictions').upsert(weekly_payload).execute()
            print("✅ Weekly prediction stored")
            
            monthly_payload = {
                'sensor_id': self.sensor_id,
                'prediction_date': str(today + timedelta(days=30)),
                'prediction_type': 'monthly',
                'predicted_kwh': float(predictions['next_month']['predicted_kwh']),
                'confidence_score': float(predictions['next_month']['confidence_score']),
                'model_version': self.model_version,
                'prophet_prediction': float(predictions['next_month']['prophet']),
                'xgboost_prediction': float(predictions['next_month']['xgboost']),
                'linear_prediction': float(predictions['next_month']['linear']),
                'training_date': training_date.isoformat()
            }
            
            self.supabase.table('predictions').upsert(monthly_payload).execute()
            print("✅ Monthly prediction stored")
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing predictions: {e}")
            return False
    
def run(self):
        """
        Execute full pipeline
        """
        print("=" * 60)
        print("🤖 Smart Watt ML Pipeline - Autonomous Training")
        print("=" * 60)
        
        try:
            df = self.pull_last_6_months_data()
            df_clean = self.clean_data(df)
            df_features = self.feature_engineering(df_clean)
            ensemble_models = self.train_ensemble_models(df_features)
            predictions = self.generate_predictions(df_features, ensemble_models)
            success = self.store_predictions_to_supabase(predictions)
            
            if success:
                print("\n" + "=" * 60)
                print("✅ PIPELINE COMPLETE - Ready for next week!")
                print("=" * 60)
                return True
            else:
                print("\n" + "=" * 60)
                print("❌ PIPELINE FAILED - Check logs")
                print("=" * 60)
                return False
                
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("❌ SUPABASE_URL and SUPABASE_KEY environment variables required!")
    
    pipeline = SmartWattML(SUPABASE_URL, SUPABASE_KEY, sensor_id="main")
    success = pipeline.run()
    
    exit(0 if success else 1)