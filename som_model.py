"""
SOM Customer Purchase Model
Trained on real Kaggle dataset: predict-customer-purchase-behavior-dataset
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from minisom_core import MiniSom

# ── exact columns from the real CSV ──────────────────────────
FEATURE_COLS = [
    "Age",
    "Gender",
    "AnnualIncome",
    "NumberOfPurchases",
    "ProductCategory",
    "TimeSpentOnWebsite",
    "LoyaltyProgram",
    "DiscountsAvailed",
]
TARGET_COL = "PurchaseStatus"

# human-readable labels for the UI
FEATURE_LABELS = {
    "Age":                "Age",
    "Gender":             "Gender (0=Female, 1=Male)",
    "AnnualIncome":       "Annual Income ($)",
    "NumberOfPurchases":  "Number of Past Purchases",
    "ProductCategory":    "Product Category (0-4)",
    "TimeSpentOnWebsite": "Time Spent on Website (min)",
    "LoyaltyProgram":     "Loyalty Program (0=No, 1=Yes)",
    "DiscountsAvailed":   "Discounts Availed",
}


def load_data(path="customer_purchase_data.csv"):
    df = pd.read_csv(path)
    return df


class SOMModel:
    def __init__(self, grid_size=10, sigma=1.5, lr=0.5, iterations=5000):
        self.grid_size     = grid_size
        self.sigma         = sigma
        self.lr            = lr
        self.iterations    = iterations
        self.som           = None
        self.scaler        = None
        self.cell_label    = {}        # (i,j) -> 0 or 1
        self.cell_rate     = {}        # (i,j) -> float purchase rate
        self.cell_count    = {}        # (i,j) -> int sample count
        self.qe            = None      # quantization error

    # ── training ─────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values

        # scale to [0,1]
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        g = self.grid_size
        self.som = MiniSom(g, g, len(FEATURE_COLS),
                           sigma=self.sigma,
                           learning_rate=self.lr,
                           random_seed=42)
        self.som.random_weights_init(X_scaled)
        self.som.train(X_scaled, self.iterations)
        self.qe = self.som.quantization_error(X_scaled)

        # label each cell by majority vote
        win_map = self.som.win_map(X_scaled)
        for cell, indices in win_map.items():
            cell_y = [y[i] for i in indices]
            rate = float(np.mean(cell_y))
            self.cell_rate[cell]  = round(rate, 4)
            self.cell_label[cell] = int(rate >= 0.5)
            self.cell_count[cell] = len(cell_y)

        # fill empty cells using nearest labelled neighbour
        for i in range(g):
            for j in range(g):
                if (i, j) not in self.cell_label:
                    self.cell_label[(i, j)] = self._nearest_label(i, j)
                    self.cell_rate[(i, j)]  = self._nearest_rate(i, j)
                    self.cell_count[(i, j)] = 0

    def _nearest_label(self, ri, rj):
        best_d, best = float("inf"), 0
        for (ci, cj), lbl in self.cell_label.items():
            d = (ri-ci)**2 + (rj-cj)**2
            if d < best_d: best_d, best = d, lbl
        return best

    def _nearest_rate(self, ri, rj):
        best_d, best = float("inf"), 0.0
        for (ci, cj), r in self.cell_rate.items():
            d = (ri-ci)**2 + (rj-cj)**2
            if d < best_d: best_d, best = d, r
        return best

    # ── single prediction ────────────────────────────────────
    def predict(self, input_dict: dict):
        row = np.array([input_dict[f] for f in FEATURE_COLS], dtype=np.float32)
        row_scaled = self.scaler.transform([row])[0]
        bmu   = self.som.winner(row_scaled)
        label = self.cell_label[bmu]
        rate  = self.cell_rate[bmu]
        return {
            "label":       label,
            "prediction":  "Will Purchase ✅" if label == 1 else "Will NOT Purchase ❌",
            "probability": rate,
            "bmu":         bmu,
        }

    # ── batch prediction ─────────────────────────────────────
    def batch_predict(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        results = []
        for x in X_scaled:
            bmu = self.som.winner(x)
            results.append({
                "Predicted":    self.cell_label[bmu],
                "Probability":  self.cell_rate[bmu],
            })
        return pd.DataFrame(results)

    # ── accuracy on any df ───────────────────────────────────
    def accuracy(self, df: pd.DataFrame):
        preds = self.batch_predict(df)
        return (preds["Predicted"].values == df[TARGET_COL].values).mean()

    # ── grid helpers ─────────────────────────────────────────
    def rate_grid(self):
        g = self.grid_size
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                grid[i, j] = self.cell_rate.get((i, j), 0.0)
        return grid

    def count_grid(self):
        g = self.grid_size
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                grid[i, j] = self.cell_count.get((i, j), 0)
        return grid

    def umatrix(self):
        return self.som.distance_map()
