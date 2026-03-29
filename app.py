"""
SOM Customer Purchase Predictor
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from som_model import SOMModel, load_data, FEATURE_COLS, FEATURE_LABELS, TARGET_COL

st.set_page_config(
    page_title="Purchase Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
* { font-family: 'Courier New', monospace; }

section[data-testid="stSidebar"] { display: none; }

.block-container { padding: 2rem 3rem; max-width: 1100px; }

h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: 0.02em;
     border-bottom: 1px solid #e2e2e2; padding-bottom: 0.6rem;
     margin-bottom: 1.5rem; }

h3 { font-size: 0.95rem; font-weight: 600; color: #444;
     text-transform: uppercase; letter-spacing: 0.08em;
     margin: 1.8rem 0 0.8rem; }

.stSlider > label, .stSelectbox > label,
.stRadio > label, .stNumberInput > label {
    font-size: 0.82rem !important;
    color: #555 !important;
    letter-spacing: 0.03em;
}

.stButton > button {
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 2rem !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    margin-top: 1rem;
}
.stButton > button:hover { background: #333 !important; }

.result-block {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}
.result-yes  { border-left: 4px solid #2d6a4f; }
.result-no   { border-left: 4px solid #c0392b; }

.result-label {
    font-size: 1.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    letter-spacing: 0.02em;
}
.result-sub {
    font-size: 0.82rem;
    color: #666;
    margin: 0;
}

.stat-row {
    display: flex;
    gap: 2rem;
    margin-bottom: 1.5rem;
}
.stat {
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    min-width: 120px;
}
.stat-val { font-size: 1.3rem; font-weight: 700; margin: 0; }
.stat-lbl { font-size: 0.72rem; color: #888; margin: 2px 0 0;
            text-transform: uppercase; letter-spacing: 0.05em; }

.tip {
    background: #f9f9f9;
    border-left: 3px solid #aaa;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #444;
    margin: 0.4rem 0;
    border-radius: 0 4px 4px 0;
}

hr { border: none; border-top: 1px solid #eee; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ── load & train ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    df    = load_data("customer_purchase_data.csv")
    model = SOMModel(grid_size=10, sigma=1.5, lr=0.5, iterations=5000)
    model.train(df)
    return model, df

with st.spinner("Training SOM..."):
    model, df = get_model()

acc = model.accuracy(df)


# ── matplotlib style ──────────────────────────────────────────
def base_fig(w=5.5, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")
        sp.set_linewidth(0.8)
    ax.tick_params(colors="#666", labelsize=7)
    return fig, ax


def plot_umatrix(bmu=None):
    um = model.umatrix()
    g  = model.grid_size
    fig, ax = base_fig()
    im = ax.imshow(um, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors="#666")
    cb.set_label("neuron distance", fontsize=8, color="#666")
    ax.set_xticks(range(0, g, 2))
    ax.set_yticks(range(0, g, 2))
    if bmu:
        ax.plot(bmu[1], bmu[0], "x", color="#c0392b",
                markersize=14, markeredgewidth=2.5, label="input")
    ax.set_title("U-Matrix", fontsize=9, color="#444", pad=8,
                 fontfamily="monospace")
    plt.tight_layout()
    return fig


def plot_rate_heatmap(bmu=None):
    grid = model.rate_grid()
    g    = model.grid_size
    fig, ax = base_fig()
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wr", ["#f5f5f5", "#c8e6c9", "#2d6a4f"])
    im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors="#666")
    cb.set_label("purchase rate", fontsize=8, color="#666")
    ax.set_xticks(range(0, g, 2))
    ax.set_yticks(range(0, g, 2))
    for i in range(g):
        for j in range(g):
            v = grid[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=5.5, color="#333")
    if bmu:
        ax.plot(bmu[1], bmu[0], "x", color="#c0392b",
                markersize=14, markeredgewidth=2.5, label="input")
    ax.set_title("Purchase rate per cell", fontsize=9, color="#444", pad=8,
                 fontfamily="monospace")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("<h1>Kohonen Network — Customer Purchase Prediction</h1>",
            unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-row">
    <div class="stat">
        <p class="stat-val">1,500</p>
        <p class="stat-lbl">training records</p>
        <p class="stat-lbl">from kaggle</p>
    </div>
    <div class="stat">
        <p class="stat-val">10 x 10</p>
        <p class="stat-lbl">SOM grid</p>
    </div>
    <div class="stat">
        <p class="stat-val">{acc:.1%}</p>
        <p class="stat-lbl">accuracy</p>
    </div>
    <div class="stat">
        <p class="stat-val">{model.qe:.3f}</p>
        <p class="stat-lbl">quantisation error</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════
st.markdown("<h3>Customer Profile</h3>", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    age              = st.slider("Age", 18, 70, 35)
    annual_income    = st.number_input("Annual Income ($)",
                                        10000, 200000, 60000, step=1000)
    num_purchases    = st.slider("Number of past purchases", 0, 50, 5)
    time_on_site     = st.slider("Time spent on website (min)",
                                  0.0, 60.0, 10.0, step=0.5)

with col_b:
    gender           = st.radio("Gender", [0, 1],
                                 format_func=lambda x: "Female" if x == 0 else "Male",
                                 horizontal=True)
    loyalty_program  = st.radio("Loyalty program member", [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 horizontal=True)
    product_category = st.selectbox("Product category", [0, 1, 2, 3, 4],
                                     format_func=lambda x:
                                     ["Electronics", "Clothing",
                                      "Home Goods", "Beauty", "Sports"][x])
    discounts        = st.slider("Discounts availed", 0, 10, 2)

predict_btn = st.button("Run prediction")


# ══════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════════
if predict_btn:
    customer = {
        "Age":                age,
        "Gender":             gender,
        "AnnualIncome":       annual_income,
        "NumberOfPurchases":  num_purchases,
        "ProductCategory":    product_category,
        "TimeSpentOnWebsite": time_on_site,
        "LoyaltyProgram":     loyalty_program,
        "DiscountsAvailed":   discounts,
    }
    result = model.predict(customer)

    st.markdown("<hr>", unsafe_allow_html=True)

    if result["label"] == 1:
        st.markdown(f"""
        <div class="result-block result-yes">
            <p class="result-label" style="color:#2d6a4f">Will Purchase</p>
            <p class="result-sub">
                Purchase probability: {result['probability']:.1%} &nbsp;|&nbsp;
                BMU cell: {result['bmu'][0]}, {result['bmu'][1]}
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-block result-no">
            <p class="result-label" style="color:#c0392b">Will Not Purchase</p>
            <p class="result-sub">
                Purchase probability: {result['probability']:.1%} &nbsp;|&nbsp;
                BMU cell: {result['bmu'][0]}, {result['bmu'][1]}
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<h3>Network Output</h3>", unsafe_allow_html=True)
    st.caption("Red cross marks where this customer lands on the trained SOM grid.")

    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.pyplot(plot_umatrix(bmu=result["bmu"]))
        st.caption("Dark areas are dense clusters. Light areas are cluster boundaries.")
    with m2:
        st.pyplot(plot_rate_heatmap(bmu=result["bmu"]))
        st.caption("Darker green means higher purchase rate in that region.")

    st.markdown("<h3>Notes</h3>", unsafe_allow_html=True)
    tips = []
    if loyalty_program == 0:
        tips.append("Not enrolled in loyalty program.")
    if discounts == 0:
        tips.append("No discounts used — consider offering one.")
    if num_purchases == 0:
        tips.append("First-time buyer — higher dropout risk.")
    if time_on_site < 3:
        tips.append("Very short session — low engagement.")
    if annual_income < 30000:
        tips.append("Lower income range — price may be a barrier.")
    if not tips:
        tips.append("Strong profile across all features.")

    for tip in tips:
        st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)