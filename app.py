# app.py
import streamlit as st
import pandas as pd
import os
from streamlit.components.v1 import html as st_html

from src.schema_checker import validate_or_map_schema, REQUIRED_FIELDS
from src.preprocessing import prepare_dataset, train_val_split_and_label
from src.training import load_or_train_model
from src.prediction import load_inference, predict_single
from src.visualization import (
    plot_confusion, plot_acc_loss, plot_corr, plot_risk_dist,
    build_search_tree_and_route, render_route_map
)
from src.visualization import plot_risk_dist, plot_hour_vs_risk, plot_cooccurrence_day_crimetype

st.set_page_config(page_title="Crime Hotspot AI", layout="wide")
# Custom CSS for buttons
st.markdown("""
<style>
button[kind="primary"], div.stButton > button, div.stFormSubmitButton > button {
    background-color: #2563eb !important;   /* blue background */
    color: white !important;                /* white text */
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    transition: background-color 0.2s ease;
}
button[kind="primary"]:hover, div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background-color: #1d4ed8 !important;   /* darker blue on hover */
}
</style>
""", unsafe_allow_html=True)
# Custom CSS for pills
st.markdown("""
<style>
.pill-wrap {
  display:flex; gap:.5rem; flex-wrap:wrap;
  margin-top:.25rem;       /* small gap after the label */
  margin-bottom:1.2rem;    /* bigger gap before the dataframe */
}
.pill {
  background:#f7f9fc; border:1px solid #e6e9ef;
  padding:.35rem .7rem; border-radius:999px;
  font-weight:600; font-size:.9rem; color:#1f2937;
  box-shadow: 0 1px 0 rgba(0,0,0,.03);
}
.pill .emoji { margin-right:.35rem; }
</style>
""", unsafe_allow_html=True)
st.title("Crime Hotspot Prediction + Patrol Path Planning")

# --- SIDEBAR: data source & options ---
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose dataset", ["Default", "Upload CSV"], index=0)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if mode == "Upload CSV" else None
retrain = st.sidebar.toggle("Retrain on current dataset", value=False)

def _df_fingerprint_for_training(df: pd.DataFrame) -> str:
    """Cheap signature so we know when the dataset changed."""
    cols = [c for c in ["DayOfWeek","Hour","Neighbourhood","CrimeType"] if c in df.columns]
    if not cols:
        return str((df.shape, tuple(df.columns)))
    sample = df[cols].head(10000)  # light hash
    h = pd.util.hash_pandas_object(sample, index=False).sum()
    return f"{h}-{df.shape}-{tuple(df.columns)}"

DATA_SIG_FILE = "models/dataset_sig.txt"

@st.cache_data(show_spinner=False)
def _map_html_from_nodes(seq):
    import folium
    m = folium.Map(location=seq[0][1], zoom_start=12)
    # markers
    for name, (lat, lon) in seq:
        folium.Marker([lat, lon], tooltip=name).add_to(m)
    # polyline
    folium.PolyLine([[lat, lon] for _, (lat, lon) in seq]).add_to(m)
    return m.get_root().render()

@st.cache_data(show_spinner=False)
def _df_fingerprint(df: pd.DataFrame) -> str:
    """Lightweight fingerprint so cache invalidates when data changes."""
    # only the columns we actually use for routing
    cols = [c for c in ["Neighbourhood", "Latitude", "Longitude"] if c in df.columns]
    if not cols:
        return str((df.shape, df.columns.tolist()))
    # hash on a small sample + shape/cols to keep it cheap
    sample = df[cols].head(5000)
    h = pd.util.hash_pandas_object(sample, index=False).sum()
    return f"{h}-{df.shape}-{tuple(df.columns)}"

@st.cache_data(show_spinner=False)
def compute_route_nodes(df: pd.DataFrame, top_k: int, hq_lat: float, hq_lon: float):
    """Cache just the route nodes/path (small), then draw figures each render."""
    from src.visualization import build_search_tree_and_route
    out = build_search_tree_and_route(df, top_k=top_k, hq=(hq_lat, hq_lon))
    # return only the light data; figures will be re-drawn
    return {"nodes": out["nodes"], "path": out["path"]}

def _draw_route_figures(nodes_path_dict):
    from src.visualization import nx, plt, _euclid  # local reuse
    from src.visualization import build_search_tree_and_route
    # Rebuild figs from nodes/path by calling a tiny wrapper
    # Easiest is to fake a minimal df with those nodes and call builder again.
    # But we already have figures in out; to avoid heavy cache object, redraw quickly:

    # Build a quick redraw using the original helper:
    # create a fake minimal structure as expected by plotting util
    # We'll reuse build_search_tree_and_route to keep style consistent

    # Nodes -> simple df to pass through generator
    fake = pd.DataFrame({
        "Neighbourhood": [n for n,_ in nodes_path_dict["nodes"] if n != "HQ"]
    })
    # The builder expects the real df; however, plotting only uses nodes passed in 'nodes'
    # So we just call a small plotting branch here:
    from src.visualization import nx, plt
    G = nx.DiGraph()
    seq = nodes_path_dict["nodes"]
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        G.add_edge(a[0], b[0])

    fig_tree = plt.figure(figsize=(5.2, 4))
    pos = {name: (coord[1], coord[0]) for name, coord in seq}
    nx.draw(G, pos, with_labels=True, node_size=600, arrows=True)
    plt.title("A* Exploration Tree (greedy leg order)")

    fig_route = plt.figure(figsize=(5.2, 4))
    xs = [c[1] for _, c in seq]; ys = [c[0] for _, c in seq]
    plt.plot(xs, ys, marker="o")
    plt.title("Patrol Route (sequence)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    return fig_tree, fig_route

# Cache model/encoders so they’re not reloaded on every submit
@st.cache_resource
def get_infer():
    return load_inference()

def load_default():
    return pd.read_csv("data/default_crime.csv")

# --- Load raw data with safe fallback ---
if mode == "Default" or uploaded is None:
    try:
        df_raw = load_default()
        source_label = "Default dataset"
    except Exception as e:
        st.sidebar.error(f"Failed to load default dataset: {e}")
        st.stop()
else:
    try:
        df_raw = pd.read_csv(uploaded)
        source_label = f"Uploaded: {uploaded.name}"
    except Exception:
        st.sidebar.error("Could not read CSV. Falling back to default dataset.")
        df_raw = load_default()
        source_label = "Default dataset (fallback)"

# --- Schema validation / mapping ---
ok, df_mapped, issues = validate_or_map_schema(df_raw)
schema_box = st.sidebar.container()
if ok:
    schema_box.success("Schema valid ✅")
else:
    schema_box.error(f"Schema issues: {issues} → Falling back to default dataset.")
    df_mapped = validate_or_map_schema(load_default())[1]
    source_label = "Default dataset (fallback)"

st.sidebar.caption(f"Using: {source_label}")

# --- Tabs ---
tab_over, tab_train, tab_predict, tab_route, tab_viz, tab_logs = st.tabs(
    ["Overview", "Preprocess & Train", "Predict", "Patrol Planner", "Visualizations", "Logs & About"]
)

with tab_over:
    st.subheader("Dataset Snapshot")
    st.markdown("<div style='margin-top:0.8rem; font-weight:600;'>Required fields</div>", unsafe_allow_html=True)
    icons = {
        "DayOfWeek": "🗓️",
        "Hour": "⏰",
        "Neighbourhood": "🏙️",
        "CrimeType": "🚨",
    }
    chips = "".join(
        f'<span class="pill" title="{name}"><span class="emoji">{icons.get(name,"🔹")}</span>{name}</span>'
        for name in REQUIRED_FIELDS
    )
    st.markdown(f'<div class="pill-wrap">{chips}</div>', unsafe_allow_html=True)
    st.dataframe(df_mapped.head(20), use_container_width=True)

    # Show rows + missing values, with centered table
    st.write("Rows:", len(df_mapped))

    st.markdown("### Missing values per column:")
    col1, col2, col3 = st.columns([1,2,1])  # middle column is wider
    with col2:
        st.dataframe(df_mapped.isna().sum())

with tab_train:
    st.subheader("Preprocess & Train")

    # ---- preprocessing & split ----
    X, y, enc, scaler, label_info, df_clean = prepare_dataset(df_mapped)
    if len(df_clean) < 100:
        st.warning("Dataset too small (<100 rows) for stable training. Consider using the default dataset.")

    X_train, X_val, y_train, y_val = train_val_split_and_label(X, y)

    # ---- decide whether to retrain (auto if dataset changed) ----
    current_sig = _df_fingerprint_for_training(df_mapped)
    prev_sig = None
    if os.path.exists(DATA_SIG_FILE):
        try:
            with open(DATA_SIG_FILE, "r", encoding="utf-8") as f:
                prev_sig = f.read().strip()
        except Exception:
            prev_sig = None

    auto_retrain = (current_sig != prev_sig)
    do_retrain = retrain or auto_retrain

    # ---- train or load model ----
    history_obj, metrics, artifacts = load_or_train_model(
        X_train, y_train, X_val, y_val, enc, scaler, label_info, retrain=do_retrain
    )

    # Save dataset signature if we trained on new data
    if do_retrain:
        try:
            os.makedirs(os.path.dirname(DATA_SIG_FILE), exist_ok=True)
            with open(DATA_SIG_FILE, "w", encoding="utf-8") as f:
                f.write(current_sig)
        except Exception:
            pass

    # ---------- METRICS: modern cards ----------
    st.markdown("<div style='height:.3rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"""
        <div style="
            background:#f9fafb;padding:1rem;border-radius:12px;
            text-align:center;border:1px solid #e6e9ef;
            box-shadow:0 1px 0 rgba(0,0,0,.03);">
            <div style="font-weight:700; margin-bottom:.25rem;">Accuracy</div>
            <div style="font-size:1.4rem; font-weight:800; color:#2563eb;">
                {metrics.get('accuracy', 0):.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c2.markdown(
        f"""
        <div style="
            background:#f9fafb;padding:1rem;border-radius:12px;
            text-align:center;border:1px solid #e6e9ef;
            box-shadow:0 1px 0 rgba(0,0,0,.03);">
            <div style="font-weight:700; margin-bottom:.25rem;">Macro F1</div>
            <div style="font-size:1.4rem; font-weight:800; color:#16a34a;">
                {metrics.get('macro_f1', 0):.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c3.markdown(
        f"""
        <div style="
            background:#f9fafb;padding:1rem;border-radius:12px;
            text-align:center;border:1px solid #e6e9ef;
            box-shadow:0 1px 0 rgba(0,0,0,.03);">
            <div style="font-weight:700; margin-bottom:.25rem;">Classes</div>
            <div style="font-size:1.4rem; font-weight:800; color:#9333ea;">
                {len(metrics.get('labels', []))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- TRAINING HISTORY ----------
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Training History")

    hist_dict = metrics.get("history")
    if hist_dict:
        st.pyplot(plot_acc_loss(hist_dict), use_container_width=True, clear_figure=True)
    else:
        st.info("No saved training history available yet. Retrain once to record history.")


    # ---------- CONFUSION MATRIX ----------
    st.markdown("<div style='margin-top:.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Confusion Matrix")
    if metrics.get("confusion") is not None:
        center_left, center_mid, center_right = st.columns([1, 2.2, 1])
        with center_mid:
            st.pyplot(plot_confusion(metrics["confusion"], metrics["labels"]))
    else:
        st.info("Train (or retrain) the model to generate the confusion matrix.")


with tab_predict:
    st.subheader("Interactive Prediction")

    # --- small style for the card ---
    st.markdown("""
    <style>
      .predict-card {
        background:#f9fafb;
        border:1px solid #e6e9ef;
        border-radius:16px;
        padding:1rem 1.2rem;
        box-shadow:0 1px 0 rgba(0,0,0,.03);
      }
    </style>
    """, unsafe_allow_html=True)

    # options from data (safe fallbacks)
    dows  = sorted(df_mapped["DayOfWeek"].dropna().unique().tolist()) or ["Monday"]
    nbhds = sorted(df_mapped["Neighbourhood"].fillna("N/A").unique().tolist()) or ["N/A"]
    ctypes= sorted(df_mapped["CrimeType"].dropna().unique().tolist()) or ["Theft"]

    # center the whole form
    left, mid, right = st.columns([1, 2.2, 1])
    with mid:
        st.markdown('<div class="predict-card">', unsafe_allow_html=True)

        # ---------- FORM: prevents rerun on every input change ----------
        with st.form("predict_form", clear_on_submit=False):
            st.markdown("**Fill the 4 features to get a risk prediction**")
            c1, c2 = st.columns(2)
            with c1:
                dow  = st.selectbox("Day of Week", dows, index=0)
                hour = st.slider("Hour", 0, 23, 12, help="24‑hour clock (0 = midnight, 23 = 11pm)")
            with c2:
                nbhd = st.selectbox("Neighbourhood", nbhds)
                ctype= st.selectbox("Crime Type", ctypes)

            # ---------- CENTER the Predict button ----------
            b1, b2, b3 = st.columns([1,1,1])
            with b2:
                submitted = st.form_submit_button("Predict Risk", use_container_width=True)

        # ---------- Only runs when the form is submitted ----------
        if submitted:
            try:
                with st.spinner("Running model…"):
                    infer = get_infer()  # cached
                    label, probs = predict_single(infer, dow, hour, nbhd, ctype)

                st.success(f"**Predicted Risk: {label}**")
                st.caption("Class probabilities")
                st.bar_chart(pd.Series(probs, index=["Low","Medium","High"]))

                st.markdown(
                    f"**Why this might be {label}**: recent patterns for "
                    f"`{nbhd}` on **{dow}** around **{hour:02d}:00** with crime type "
                    f"`{ctype}` contributed to this estimate."
                )
            except Exception as e:
                st.error(f"Model artifacts not found or incompatible. Train the model first. ({e})")

        st.markdown('</div>', unsafe_allow_html=True)


# ------------------ DROP-IN for your Patrol Planner tab ------------------
with tab_route:
    st.subheader("Patrol Route Planning")

    # Style shell
    st.markdown("""
    <style>
      .route-card {
        background:#f9fafb;border:1px solid #e6e9ef;border-radius:16px;
        padding:1rem 1.2rem;box-shadow:0 1px 0 rgba(0,0,0,.03);
      }
    </style>
    """, unsafe_allow_html=True)

    # Center the form + keep Top-K inside this form to avoid sidebar-triggered reruns
    left, mid, right = st.columns([1, 2.2, 1])
    with mid:
        st.markdown('<div class="route-card">', unsafe_allow_html=True)

        with st.form("route_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                hq_lat = st.number_input("HQ Latitude", value=49.2827, format="%.6f")
                return_to_hq = st.checkbox("Return to HQ at the end", value=True)
            with c2:
                hq_lon = st.number_input("HQ Longitude", value=-123.1207, format="%.6f")
                travel_speed = st.number_input("Patrol speed (km/h)", value=25.0, min_value=1.0, max_value=120.0)

            # Move Top-K here so it doesn't cause sidebar reruns
            top_k_local = st.slider("Top‑K hotspots (for this route)", 3, 20, 7)

            # Centered submit
            b1, b2, b3 = st.columns([1,1,1])
            with b2:
                go_route = st.form_submit_button("Generate Patrol Route", use_container_width=True)

        # Keep last result in session so incidental reruns don't wipe it
        if "route_result" not in st.session_state:
            st.session_state.route_result = None
            st.session_state.route_params = None

        if go_route:
            try:
                with st.spinner("Building route…"):
                    df_sig = _df_fingerprint(df_mapped)
                    nodes_path = compute_route_nodes(df_mapped, top_k_local, hq_lat, hq_lon)
                st.session_state.route_result = {
                    "nodes_path": nodes_path,
                    "params": {
                        "hq_lat": hq_lat, "hq_lon": hq_lon,
                        "top_k": top_k_local, "return_to_hq": return_to_hq,
                        "travel_speed": travel_speed, "df_sig": df_sig
                    }
                }
            except Exception as e:
                st.error(f"Could not build route: {e}")

        # Render whatever we currently have (persisted)
        if st.session_state.route_result:
            from math import radians, sin, cos, sqrt, atan2

            def haversine_km(a, b):
                R = 6371.0
                lat1, lon1 = radians(a[0]), radians(a[1])
                lat2, lon2 = radians(b[0]), radians(b[1])
                dlat = lat2 - lat1; dlon = dlon = lon2 - lon1
                x = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                return 2 * R * atan2(sqrt(x), sqrt(1-x+1e-12))

            params = st.session_state.route_result["params"]
            nodes_path = st.session_state.route_result["nodes_path"]
            seq = nodes_path["nodes"]
            pts = [coord for _, coord in seq]

            # Distance + ETA
            total_km = 0.0
            for i in range(len(pts)-1):
                total_km += haversine_km(pts[i], pts[i+1])
            if params["return_to_hq"] and len(pts) > 1:
                total_km += haversine_km(pts[-1], pts[0])
                seq = seq + [seq[0]]  # for map/polyline closure

            eta_hours = total_km / max(params["travel_speed"], 1e-6)
            eta_min = eta_hours * 60

            # metric cards
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"<div style='background:#fff;border:1px solid #e6e9ef;border-radius:12px;"
                f"padding:1rem;text-align:center;'><div style='font-weight:700'>Stops</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#2563eb;'>{len(seq)-1}</div></div>",
                unsafe_allow_html=True)
            c2.markdown(
                f"<div style='background:#fff;border:1px solid #e6e9ef;border-radius:12px;"
                f"padding:1rem;text-align:center;'><div style='font-weight:700'>Total Distance</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#16a34a;'>{total_km:.2f} km</div></div>",
                unsafe_allow_html=True)
            c3.markdown(
                f"<div style='background:#fff;border:1px solid #e6e9ef;border-radius:12px;"
                f"padding:1rem;text-align:center;'><div style='font-weight:700'>ETA</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#9333ea;'>{eta_min:.0f} min</div></div>",
                unsafe_allow_html=True)

            # Redraw figures from nodes/path (fast)
            fig_tree, fig_route = _draw_route_figures(nodes_path)
            st.markdown("### Exploration Tree")
            st.pyplot(fig_tree)
            st.markdown("### Patrol Route (static)")
            st.pyplot(fig_route)

            # Map (if available)
            try:
                st.markdown("### Map")
                map_html = _map_html_from_nodes(seq)   # cached by nodes sequence
                st_html(map_html, height=520)          # iframe; no events back to Streamlit
            except Exception:
                st.info("Folium not available — install `folium` to view the map.")


            # Export
            export_df = pd.DataFrame(
                [{"order": i, "name": name, "lat": lat, "lon": lon}
                 for i, (name, (lat, lon)) in enumerate(seq)]
            )

        st.markdown('</div>', unsafe_allow_html=True)
        
        
with tab_viz:
    st.subheader("Visualizations")

    st.markdown("""
    <style>
      .viz-card {
        background:#f9fafb; border:1px solid #e6e9ef; border-radius:16px;
        padding:1rem 1.2rem; box-shadow:0 1px 0 rgba(0,0,0,.03); margin-bottom:1.2rem;
      }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Distribution ----------
    c1, c2, c3 = st.columns([1, 2.6, 1])
    with c2:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown("### Risk / CrimeType Distribution")
        fig_dist = plot_risk_dist(df_mapped)
        st.pyplot(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Hour × Risk ----------
    c1, c2, c3 = st.columns([1, 2.6, 1])
    with c2:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown("### Hour × Risk Heatmap")
        fig_hr = plot_hour_vs_risk(df_mapped)
        st.pyplot(fig_hr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- DayOfWeek × CrimeType ----------
    c1, c2, c3 = st.columns([1, 2.6, 1])
    with c2:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown("### DayOfWeek × CrimeType (Top 10)")
        fig_dc = plot_cooccurrence_day_crimetype(df_mapped, top_n=10)
        st.pyplot(fig_dc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        

with tab_logs:
    st.subheader("Logs & About")

    st.markdown("""
    <style>
      .about-card, .logs-card {
        background:#f9fafb;
        border:1px solid #e6e9ef;
        border-radius:16px;
        padding:1rem 1.2rem;
        box-shadow:0 1px 0 rgba(0,0,0,.03);
        margin-bottom:1.5rem;
      }
    </style>
    """, unsafe_allow_html=True)

    # --- About section ---
    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About this Project")
    st.write(
        """
        **Crime Hotspot Prediction + Patrol Path Planning**  
        This demo system combines **AI classification (ANN)** and **search algorithms (A*)**  
        to predict crime risk levels and generate efficient patrol routes.

        - 📌 **Dataset:** Vancouver Crime Data (Kaggle)  
        - 🧩 **Features used:** DayOfWeek, Hour, Neighbourhood, CrimeType  
        - 🧠 **Model:** Artificial Neural Network (Keras/TensorFlow)  
        - 🗺️ **Planner:** A* search with greedy ordering for patrol route optimization  
        - 🎯 **Impact:** Helps visualize and optimize policing strategies for safer communities
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Logs / Technical notes section ---
    st.markdown('<div class="logs-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ System Logs / Technical Notes")
    st.write(
        """
        - ✅ Safe schema validator & fallback  
        - ✅ Deterministic preprocessing & error handling  
        - ✅ Artifacts stored under `./models` and `./encoders`  
        - ✅ Interactive Streamlit GUI with session persistence  
        - ✅ Visualizations exportable as PNG for reporting  
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Credits ---
    st.caption("👨‍💻 Developed by *Team Trinary* for the Artificial Intelligence course project.")