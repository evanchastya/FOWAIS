import customtkinter as ctk
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tkinter import messagebox
from pathlib import Path
from typing import List

# ================= CONFIGURATION =================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# Define paths using pathlib
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "fowais_brain.pkl" # Note: .pkl extension now
SCALER_PATH = ARTIFACTS_DIR / "fowais_scaler.pkl"
DATA_PATH = ARTIFACTS_DIR / "supermarket_inventory.csv"

class FowaisSystem(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # --- 1. SETUP & LOADING ---
        self.title("FOWAIS - Food Waste Reduction Intelligent System")
        self.geometry("1100x600")
        
        # Validation
        if not self._check_artifacts():
            self.destroy()
            return

        # Load the "Brain" and Tools (Pure Scikit-Learn/Joblib)
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.inventory_df = pd.read_csv(DATA_PATH)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load system components:\n{e}")
            self.destroy()
            return

        # --- 2. UI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_frame()
        
        # Initial Data Load
        self.refresh_table()

    def _check_artifacts(self) -> bool:
        """Ensures all necessary files exist."""
        missing = []
        if not MODEL_PATH.exists(): missing.append(str(MODEL_PATH))
        if not SCALER_PATH.exists(): missing.append(str(SCALER_PATH))
        if not DATA_PATH.exists(): missing.append(str(DATA_PATH))
        
        if missing:
            msg = "Missing required files:\n" + "\n".join(missing) + "\n\nPlease run fowais_training.py first."
            messagebox.showerror("Configuration Error", msg)
            return False
        return True

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="FOWAIS\nDashboard", 
                     font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.btn_scan = ctk.CTkButton(self.sidebar, text="Scan Inventory (IoT)", command=self.refresh_table)
        self.btn_scan.grid(row=1, column=0, padx=20, pady=10)

        self.btn_predict = ctk.CTkButton(self.sidebar, text="Run AI Forecast", 
                                         fg_color="#3B8ED0", command=self.run_ai_prediction)
        self.btn_predict.grid(row=2, column=0, padx=20, pady=10)

        self.btn_donate = ctk.CTkButton(self.sidebar, text="Auto-Route Donations", 
                                        fg_color="#E04F5F", hover_color="#C0392B", 
                                        command=self.process_donations)
        self.btn_donate.grid(row=3, column=0, padx=20, pady=10)

    def _build_main_frame(self) -> None:
        self.main_frame = ctk.CTkScrollableFrame(self, label_text="Real-Time Inventory Status")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    # --- 3. CORE FUNCTIONS ---

    def refresh_table(self) -> None:
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        headers = ["Product", "Stock", "Expiry", "Days Left", "AI Forecast", "System Action"]
        for idx, h in enumerate(headers):
            ctk.CTkLabel(self.main_frame, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=idx, padx=10, pady=5, sticky="w")

        today = datetime.now()

        for idx, row in self.inventory_df.iterrows():
            r = idx + 1
            
            try:
                exp_date = datetime.strptime(row['expiry_date'], "%Y-%m-%d")
                days_left = (exp_date - today).days
            except ValueError:
                days_left = 0
            
            day_color = "white"
            if days_left < 0: day_color = "#E04F5F"
            elif days_left <= 2: day_color = "#E67E22"
            
            ctk.CTkLabel(self.main_frame, text=str(row['product_name'])).grid(row=r, column=0, padx=10, pady=2)
            ctk.CTkLabel(self.main_frame, text=str(row['stock_current'])).grid(row=r, column=1, padx=10, pady=2)
            ctk.CTkLabel(self.main_frame, text=str(row['expiry_date'])).grid(row=r, column=2, padx=10, pady=2)
            ctk.CTkLabel(self.main_frame, text=f"{days_left} days", text_color=day_color).grid(row=r, column=3, padx=10, pady=2)
            
            ai_val = row.get('predicted_demand', '---')
            action = row.get('action_status', 'Wait')
            
            action_color = "white"
            if action == "AUTO-DONATE": action_color = "#E67E22"
            elif action == "DISCOUNT (Overstock)": action_color = "#ffee58"
            elif action == "DISPOSE": action_color = "#E04F5F"
            elif action == "OPTIMAL": action_color = "#66bb6a"

            ctk.CTkLabel(self.main_frame, text=str(ai_val), text_color="#3B8ED0").grid(row=r, column=4, padx=10, pady=2)
            ctk.CTkLabel(self.main_frame, text=str(action), text_color=action_color, 
                         font=ctk.CTkFont(weight="bold")).grid(row=r, column=5, padx=10, pady=2)

    def run_ai_prediction(self) -> None:
        input_cols = ['sales_w1', 'sales_w2', 'sales_w3', 'is_holiday_next_week']
        
        data_to_predict = self.inventory_df[input_cols].values.astype(float)
        data_scaled = self.scaler.transform(data_to_predict)
        
        # sklearn .predict() returns a 1D array [val1, val2]
        predictions = self.model.predict(data_scaled)
        
        today = datetime.now()
        actions: List[str] = []
        preds: List[int] = []

        for i, pred_val in enumerate(predictions):
            # No need for pred_val[0] because sklearn returns scalar values in a 1D array
            demand = int(max(0, pred_val))
            preds.append(demand)
            
            row = self.inventory_df.iloc[i]
            exp_date = datetime.strptime(row['expiry_date'], "%Y-%m-%d")
            days_left = (exp_date - today).days
            stock = row['stock_current']

            if days_left < 0:
                actions.append("DISPOSE")
            elif days_left <= 2:
                actions.append("AUTO-DONATE")
            elif stock > (demand * 1.5):
                actions.append("DISCOUNT (Overstock)")
            else:
                actions.append("OPTIMAL")

        self.inventory_df['predicted_demand'] = preds
        self.inventory_df['action_status'] = actions
        
        self.refresh_table()
        messagebox.showinfo("FOWAIS AI", "Demand forecasted successfully!")

    def process_donations(self) -> None:
        if 'action_status' not in self.inventory_df.columns:
            messagebox.showwarning("System", "Run AI Forecast first.")
            return

        donations = self.inventory_df[self.inventory_df['action_status'] == "AUTO-DONATE"]
        
        if donations.empty:
            messagebox.showinfo("Donation System", "No critical items found for donation.")
        else:
            items_list = [f"- {r['product_name']} ({r['stock_current']} units)" for _, r in donations.iterrows()]
            items_str = "\n".join(items_list)
            messagebox.showinfo("Donation Dispatched", 
                                f"The following items have been routed to 'Panti Asuhan Kasih':\n\n{items_str}")

if __name__ == "__main__":
    app = FowaisSystem()
    app.mainloop()