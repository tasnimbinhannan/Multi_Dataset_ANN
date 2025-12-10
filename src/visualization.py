# /mnt/data/crime_hotspot_ai/src/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    import folium
except Exception:  # optional dependency
    folium = None

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    return fig

def plot_acc_loss(history: dict):
    """
    Draw responsive loss/accuracy curves.
    Streamlit will scale width with `use_container_width=True`.
    """
    # Read keys with safe fallbacks
    loss = history.get("loss", [])
    vloss = history.get("val_loss", [])
    acc  = history.get("accuracy", history.get("acc", []))
    vacc = history.get("val_accuracy", history.get("val_acc", []))

    # Avoid giant fixed-size images: smaller base figsize + constrained layout
    fig, ax1 = plt.subplots(figsize=(8, 4), layout="constrained", dpi=100)
    ax2 = ax1.twinx()

    ax1.plot(loss, label="train_loss")
    if len(vloss):
        ax1.plot(vloss, "--", label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    if len(acc):
        ax2.plot(acc, ":", label="train_acc", color="tab:blue")
    if len(vacc):
        ax2.plot(vacc, ":", label="val_acc", color="tab:orange")
    ax2.set_ylabel("Accuracy")

    # Two legends (left: loss, right: acc)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Keep titles inside the page rhythm; avoid giant suptitles
    ax1.set_title("Training History")
    return fig

def plot_corr(df):
    # Light correlation viz using numeric columns only (Hour at least)
    tmp = pd.DataFrame({
        "Hour": pd.to_numeric(df.get("Hour", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
    })
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(tmp.corr(numeric_only=True), annot=True, cmap="crest", ax=ax)
    ax.set_title("Feature Correlation (numeric subset)")
    return fig

def plot_risk_dist(df):
    # If Risk labels exist, show them; else show CrimeType distribution as proxy
    col = "Risk" if "Risk" in df.columns else "CrimeType"
    s = df[col].astype(str)
    fig, ax = plt.subplots(figsize=(6, 3))
    s.value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution: {col}")
    ax.set_ylabel("Count")
    return fig

# ------------------- Patrol planning & visual tree -------------------

def _euclid(a, b): 
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _centroids_by_neighbourhood(df):
    if {"Latitude","Longitude"}.issubset(df.columns):
        grp = df.groupby("Neighbourhood")[["Latitude","Longitude"]].mean().dropna()
        return {name: (row["Latitude"], row["Longitude"]) for name, row in grp.iterrows()}
    # Fallback to synthetic centroids (keeps demo robust even if coords missing)
    rng = np.random.default_rng(42)
    return {nb: (49.25 + rng.random()*0.1, -123.2 + rng.random()*0.1) 
            for nb in df["Neighbourhood"].dropna().unique()[:20]}

def _greedy_sequence(nodes):
    # nodes: list of (name, (lat, lon)); first is HQ
    seq = [nodes[0]]
    remaining = nodes[1:].copy()
    while remaining:
        last = seq[-1][1]
        nxt = min(remaining, key=lambda n: _euclid(last, n[1]))
        seq.append(nxt); remaining.remove(nxt)
    return seq

def build_search_tree_and_route(df, top_k=7, hq=(49.2827, -123.1207)):
    # pick hotspots by frequency
    top_counts = df["Neighbourhood"].value_counts().head(top_k).index.tolist()
    centroids = _centroids_by_neighbourhood(df)
    nodes = [("HQ", hq)] + [(nb, centroids.get(nb)) for nb in top_counts if nb in centroids]

    # Greedy order (TSP-lite), then create a simple directed graph to mimic A* leg chaining
    sequence = _greedy_sequence(nodes)
    G = nx.DiGraph()
    for i in range(len(sequence) - 1):
        G.add_edge(sequence[i][0], sequence[i+1][0], weight=_euclid(sequence[i][1], sequence[i+1][1]))

    # Tree figure
    fig_tree = plt.figure(figsize=(5.2, 4))
    pos = {name: (coord[1], coord[0]) for name, coord in sequence}  # lon, lat for nicer spread
    nx.draw(G, pos, with_labels=True, node_size=600, arrows=True)
    plt.title("A* Exploration Tree (greedy leg order)")

    # Route figure
    fig_route = plt.figure(figsize=(5.2, 4))
    xs = [c[1] for _, c in sequence]; ys = [c[0] for _, c in sequence]
    plt.plot(xs, ys, marker="o")
    plt.title("Patrol Route (sequence)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")

    return {"tree_fig": fig_tree, "route_fig": fig_route, "nodes": sequence, "path": [n for n,_ in sequence]}

def render_route_map(nodes, path):
    if folium is None:
        return None
    try:
        m = folium.Map(location=nodes[0][1], zoom_start=12)
        for name, (lat, lon) in nodes:
            folium.Marker([lat, lon], tooltip=name).add_to(m)
        coords = [[lat, lon] for _, (lat, lon) in nodes]
        folium.PolyLine(coords).add_to(m)
        return m
    except Exception:
        return None
     
def plot_hour_vs_risk(df):
    """Heatmap of hour vs risk frequency."""
    df2 = df.copy()
    # Ensure Risk exists; if not, proxy with CrimeType
    if "Risk" not in df2.columns:
        # fallback: build a 'Risk' proxy from CrimeType counts
        counts = df2["CrimeType"].value_counts()
        top = set(counts.index[:max(1, int(len(counts)*0.3))])
        df2["Risk"] = np.where(df2["CrimeType"].isin(top), "High", "Medium")
    df2["Hour"] = pd.to_numeric(df2.get("Hour", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
    pv = pd.crosstab(df2["Hour"], df2["Risk"], normalize="index")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(pv, annot=False, cmap="viridis", ax=ax)
    ax.set_title("Hour × Risk (row‑normalized)")
    return fig

def plot_cooccurrence_day_crimetype(df, top_n=10):
    """Heatmap of DayOfWeek × top CrimeType frequencies."""
    df2 = df.copy()
    if "DayOfWeek" not in df2.columns or "CrimeType" not in df2.columns:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5, 0.5, "DayOfWeek/CrimeType not available.", ha="center", va="center")
        ax.axis("off")
        return fig
    top_types = df2["CrimeType"].value_counts().head(top_n).index
    pv = pd.crosstab(df2["DayOfWeek"], df2["CrimeType"].where(df2["CrimeType"].isin(top_types)))
    pv = pv[top_types]  # order columns
    fig, ax = plt.subplots(figsize=(7,4))
    sns.heatmap(pv, cmap="mako", ax=ax)
    ax.set_title(f"DayOfWeek × CrimeType (Top {top_n})")
    return fig