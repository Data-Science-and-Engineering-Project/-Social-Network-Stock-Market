

# 📈 Stock Market Social Network

## Project Goal
**Objective:** Predict which stocks an investment fund is likely to buy in the next quarter, using advanced graph-based machine learning. The system is designed for robust, unbiased, and out-of-sample prediction, with strict prevention of data/time leakage.

---

## How It Works: User Workflow
1. **Prepare Data:** Place all quarterly CSV files in the working directory. Each file must include: `CIK` (fund ID), `CUSIP` (stock ID), `VALUE`, `SSHPRNAMT`, `PERIOD_DATE`, `QUARTER`.
2. **Run the Pipeline:** Open the Jupyter notebook and execute all cells in order. The pipeline will:
	- Load and clean the data
	- Build graphs and extract features
	- Train the model (Q1-Q2)
	- Evaluate on out-of-sample data (Q3)
	- Predict for the next quarter (Q4)
3. **Get Recommendations:** In the last cell, select a fund (from those eligible) and receive top stock recommendations for Q4.

---

## Mathematical & Algorithmic Explanation

### 1. Graph Construction
The pipeline models the market as a bipartite graph $G=(F, S, E)$ where $F$ is the set of funds, $S$ is the set of stocks, and $E$ are edges representing fund-stock holdings. A projected fund-fund graph is also constructed, where edges represent similarity or co-holdings between funds.

### 2. Feature Engineering
For each node (fund or stock), the following features are computed:
- **Degree:** $deg(v)$, the number of edges for node $v$.
- **PageRank:** $PR(v)$, importance score from the PageRank algorithm.
- **Hubs & Authorities:** Computed using the HITS algorithm.
- **Closeness Centrality:** $C(v) = \frac{1}{\sum_{u \neq v} d(u,v)}$
- **Community Detection:** Using the Leiden algorithm to assign community labels.

### 3. Node Embeddings (GraphSAGE)
GraphSAGE is a graph neural network that learns low-dimensional representations (embeddings) for each node by aggregating features from its neighbors. For node $v$:
$$
h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{AGGREGATE}^{(k)} \left( \{ h_v^{(k-1)} \} \cup \{ h_u^{(k-1)}, \forall u \in N(v) \} \right) \right)
$$
where $h_v^{(k)}$ is the embedding at layer $k$, $N(v)$ are neighbors, $W^{(k)}$ are learnable weights, and $\sigma$ is a nonlinearity.

### 4. Model Training (LightGBM)
The final feature vector for each (fund, stock) pair is a concatenation of:
- Fund features
- Stock features
- Fund embedding
- Stock embedding

These are used to train a LightGBM classifier to predict whether a fund will buy a stock in the next quarter. The model is trained only on Q1-Q2 data, with negative sampling to balance the dataset.

### 5. Evaluation & Prediction
- **Evaluation:** Performed on Q3, only for funds and stocks seen in training. Metrics: AUC, precision, positive/negative ratio.
- **Prediction:** For Q4, the model recommends stocks for eligible funds (seen in training and present in Q4), ensuring no data/time leakage.

---

## Professional Notes on Data Integrity
- **No time leakage:** All features and embeddings for training are built using Q1-Q2 only.
- **No data leakage:** Evaluation is performed only on Q3, and prediction is for Q4 using only eligible funds.
- **No label leakage:** The model never sees future data during training or feature construction.

---

## Example Usage
```python
# In the notebook, set:
fund_id_to_predict = 'YOUR_FUND_CIK'  # e.g., '1325091'
# Or leave as None for random selection
```
Run the prediction cell to get the top recommended stocks for the selected fund in Q4.

---

## Artifacts & Reproducibility
All trained models, embeddings, and features are saved in the `artifacts/` directory for fast loading and reproducibility.

---

## Requirements
- Python 3.8+
- pandas, numpy, networkx, igraph, leidenalg
- torch, torch_geometric
- scikit-learn, lightgbm, joblib, pickle

---

## About
This project was developed by a professional data scientist for academic and research use. For questions or collaboration, please contact the author.