"""
SOM Customer Purchase Model.
Wraps MiniSom with data loading, scaling, cell labelling,
prediction, and grid helper methods.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom_core import MiniSom

# ── columns from the real Kaggle CSV ─────────────────────────
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
    return pd.read_csv(path)


class SOMModel:
    def __init__(self, grid_size=8, sigma=1.2, lr=0.5, iterations=15000):
        self.grid_size  = grid_size
        self.sigma      = sigma
        self.lr         = lr
        self.iterations = iterations
        self.som        = None
        self.scaler     = None
        self.cell_label = {}   # (i,j) -> 0 or 1
        self.cell_rate  = {}   # (i,j) -> float  purchase rate 0-1
        self.cell_count = {}   # (i,j) -> int    number of training samples
        self.qe         = None

    # ── training ─────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        """
        Full training pipeline:
        1. Scale features to [0,1]
        2. Train SOM
        3. Label each cell by majority vote of samples that landed there
        """
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values

        self.scaler = MinMaxScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.som = MiniSom(
            self.grid_size, self.grid_size, len(FEATURE_COLS),
            sigma=self.sigma, learning_rate=self.lr, random_seed=42
        )
        self.som.random_weights_init(X_scaled)
        self.som.train = self._train_loop   # use internal loop

        rng = np.random.RandomState(42)
        for t in range(self.iterations):
            x   = X_scaled[rng.randint(len(X_scaled))]
            win = self.som.winner(x)
            self.som.update(x, win, t, self.iterations)

        self.qe = self.som.quantization_error(X_scaled)
        self._label_cells(X_scaled, y)

    def _train_loop(self, *args, **kwargs):
        pass  # placeholder — training done manually above

    def _label_cells(self, X_scaled, y):
        """Assign each cell a label via majority vote."""
        g       = self.grid_size
        win_map = self.som.win_map(X_scaled)

        for cell, indices in win_map.items():
            cell_y = [y[i] for i in indices]
            rate   = float(np.mean(cell_y))
            self.cell_rate[cell]  = round(rate, 4)
            self.cell_label[cell] = int(rate >= 0.5)
            self.cell_count[cell] = len(cell_y)

        # cells with no training samples inherit nearest neighbour's label
        for i in range(g):
            for j in range(g):
                if (i, j) not in self.cell_label:
                    self.cell_label[(i, j)] = self._nearest_label(i, j)
                    self.cell_rate[(i, j)]  = self._nearest_rate(i, j)
                    self.cell_count[(i, j)] = 0

    def _nearest_label(self, ri, rj):
        best_d, best = float("inf"), 0
        for (ci, cj), lbl in self.cell_label.items():
            d = (ri - ci) ** 2 + (rj - cj) ** 2
            if d < best_d:
                best_d, best = d, lbl
        return best

    def _nearest_rate(self, ri, rj):
        best_d, best = float("inf"), 0.0
        for (ci, cj), r in self.cell_rate.items():
            d = (ri - ci) ** 2 + (rj - cj) ** 2
            if d < best_d:
                best_d, best = d, r
        return best

    # ── prediction ───────────────────────────────────────────
    def predict(self, input_dict: dict):
        """
        Predict purchase for one customer.
        input_dict must contain all FEATURE_COLS keys.
        """
        row       = np.array([input_dict[f] for f in FEATURE_COLS],
                             dtype=np.float32)
        row_s     = self.scaler.transform([row])[0]
        bmu       = self.som.winner(row_s)
        label     = self.cell_label[bmu]
        rate      = self.cell_rate[bmu]
        return {
            "label":       label,
            "prediction":  "Will Purchase" if label == 1 else "Will Not Purchase",
            "probability": rate,
            "bmu":         bmu,
        }

    def batch_predict(self, df: pd.DataFrame):
        """Predict purchase for every row in df. Returns a DataFrame."""
        X     = df[FEATURE_COLS].values.astype(np.float32)
        X_s   = self.scaler.transform(X)
        rows  = []
        for x in X_s:
            bmu = self.som.winner(x)
            rows.append({
                "Predicted":   self.cell_label[bmu],
                "Probability": self.cell_rate[bmu],
            })
        return pd.DataFrame(rows)

    def accuracy(self, df: pd.DataFrame):
        """
        Fraction of correct predictions on df.
        Note: when df is the training set this is training accuracy,
        not a fair measure of generalisation.
        """
        preds  = self.batch_predict(df)
        y_pred = preds["Predicted"].values
        y_true = df[TARGET_COL].values
        return (y_pred == y_true).mean()

    # ── grid helpers for visualisation ───────────────────────
    def rate_grid(self):
        g    = self.grid_size
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                grid[i, j] = self.cell_rate.get((i, j), 0.0)
        return grid

    def count_grid(self):
        g    = self.grid_size
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                grid[i, j] = self.cell_count.get((i, j), 0)
        return grid

    def umatrix(self):
        return self.som.distance_map()