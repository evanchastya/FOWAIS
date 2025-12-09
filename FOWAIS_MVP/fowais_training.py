# ==========================================
# FOWAIS: AI Training Module (Data Scientist Layer)
# Python 3.14 Compatible - NO TENSORFLOW
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- 0. CONFIGURATION & REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# --- 1. DATA GENERATION ---
print("Generating synthetic dataset...")
X = []
y = []

for _ in range(2000):
    base_sales = random.randint(20, 100)
    trend = random.choice([0.9, 1.0, 1.1]) 
    
    w1 = int(base_sales * random.uniform(0.9, 1.1))
    w2 = int(w1 * trend * random.uniform(0.9, 1.1))
    w3 = int(w2 * trend * random.uniform(0.9, 1.1))
    
    is_holiday = random.choice([0, 1])
    
    w4 = int(w3 * trend)
    if is_holiday:
        w4 = int(w4 * 1.2)
        
    X.append([w1, w2, w3, is_holiday])
    y.append(w4)

X_arr = np.array(X, dtype=np.float32)
y_arr = np.array(y, dtype=np.float32)

# --- 2. PREPROCESSING ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_arr, test_size=0.2, random_state=SEED)

# --- 3. MODEL ARCHITECTURE (The "Brain") ---
# Using Scikit-Learn's Neural Network (MLP) instead of TensorFlow
print("Initializing Scikit-Learn MLP Regressor...")
model = MLPRegressor(
    hidden_layer_sizes=(64, 32), # Same architecture as before
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=SEED
)

# --- 4. TRAINING ---
print("Training FOWAIS Neural Network...")
model.fit(X_train, y_train)

score = model.score(X_val, y_val)
print(f"Training Complete. Validation R^2 Score: {score:.4f}")

# --- 5. SAVE ARTIFACTS ---
# We now use joblib for BOTH the scaler and the model
model_path = ARTIFACTS_DIR / 'fowais_brain.pkl'
scaler_path = ARTIFACTS_DIR / 'fowais_scaler.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Artifacts saved to: {ARTIFACTS_DIR}")

# --- 6. GENERATE MOCK INVENTORY FOR APP ---
products = ["Roti Tawar", "Susu UHT", "Sayur Bayam", "Daging Ayam", "Pisang", "Yoghurt", "Ikan Salmon"]
inventory = []
today = datetime.now()

for pid, name in enumerate(products):
    inventory.append({
        "id": pid + 1,
        "product_name": name,
        "stock_current": random.randint(10, 80),
        "expiry_date": (today + timedelta(days=random.randint(-1, 8))).strftime("%Y-%m-%d"),
        "sales_w1": random.randint(20, 60),
        "sales_w2": random.randint(20, 60),
        "sales_w3": random.randint(20, 60),
        "is_holiday_next_week": random.choice([0, 1])
    })

csv_path = ARTIFACTS_DIR / 'supermarket_inventory.csv'
pd.DataFrame(inventory).to_csv(csv_path, index=False)
print(f"Database created: {csv_path}")