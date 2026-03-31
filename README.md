# Kohonen Network Customer Purchase Predictor

A Streamlit app that uses a Self-Organizing Map (SOM) to predict whether a customer will make a purchase, based on their profile.

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit UI — training progress, input form, result card, SOM visualisations |
| `som_model.py` | `SOMModel` wrapper — data loading, scaling, cell labelling, prediction |
| `minisom_core.py` | Pure-NumPy SOM implementation (no external ML dependencies) |
| `customer_purchase_data.csv` | 1,500-record Kaggle dataset |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. On first load the app trains an **8×8 SOM** on 1,500 customer records for 15,000 iterations.
2. Each grid cell is labelled **Will Purchase / Will Not Purchase** by majority vote of the training samples that landed there.
3. A new customer is mapped to their Best Matching Unit (BMU); the cell's label and purchase rate become the prediction.
4. The UI shows the result alongside a **U-Matrix** (cluster boundaries) and a **purchase-rate heatmap**, with the customer's BMU marked by a red cross.

## Input features

| Feature | Range |
|---|---|
| Age | 18 – 70 |
| Gender | Female / Male |
| Annual Income | $10,000 – $200,000 |
| Number of past purchases | 0 – 50 |
| Product category | Electronics, Clothing, Home Goods, Beauty, Sports |
| Time spent on website | 0 – 60 min |
| Loyalty program member | Yes / No |
| Discounts availed | 0 – 10 |

## Notes

- Accuracy is reported on the **training set** and is not a measure of generalisation.
- The SOM is seeded with `random_seed=42` for reproducibility.
- Training runs in-browser on first load and is cached for the session.
