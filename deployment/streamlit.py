import os
import requests
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="AAPL Direction Predictor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

FEATURES = ["avg_sentiment", "rolling_3d_sentiment", "rolling_7d_sentiment",
            "daily_return", "price_momentum_5d", "high_low_range",
            "ma_5", "ma_10", "volume", "day_of_week",
            "lagged_target", "sentiment_std_5d", "sentiment_vs_trend"]
DEFAULTS = dict(avg_sentiment=0.1234, rolling_3d_sentiment=0.105,
                rolling_7d_sentiment=0.082, daily_return=0.0075,
                price_momentum_5d=0.0125, high_low_range=0.018,
                ma_5=226.45, ma_10=224.10, volume=50_000_000,
                day_of_week=2, lagged_target=1,
                sentiment_std_5d=0.0412, sentiment_vs_trend=0.0414)

api_url = ""  


@st.cache_data(ttl=60, show_spinner=False)
def api_get(url, path, **params):
    r = requests.get(f"{url}{path}", params=params, timeout=20); r.raise_for_status()
    return r.json()


def api_post(path, **kw):
    r = requests.post(f"{api_url}{path}", timeout=60, **kw); r.raise_for_status()
    return r.json()


def predict_batch(df, model):
    return api_post("/predict_batch", params={"model": model},
                    files={"file": ("b.csv", df.to_csv(index=False).encode(), "text/csv")})


def style(fig):
    fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font=dict(color="#000"),
                      title=dict(font=dict(color="#000")),
                      legend=dict(font=dict(color="#000"),
                                  title=dict(font=dict(color="#000"))))
    for axes in (fig.update_xaxes, fig.update_yaxes):
        axes(linecolor="#4d4d4d", gridcolor="#d0d4db",
             tickfont=dict(color="#000"),
             title_font=dict(color="#000"))
    fig.update_annotations(font=dict(color="#000"))
    fig.update_coloraxes(colorbar_tickfont=dict(color="#000"),
                         colorbar_title_font=dict(color="#000"))
    return fig


def section(title, desc):
    st.markdown(f"<h3 style='font-weight:700;margin-bottom:.2rem'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(desc)


# Sidebar
st.sidebar.title("⚙️ Settings")
api_url = st.sidebar.text_input("FastAPI base URL",
                                value=os.environ.get("API_URL", "http://127.0.0.1:8000"))
try:
    meta = api_get(api_url, "/")
    available_models, accuracies = meta["available_models"], meta.get("test_accuracy", {})
    st.sidebar.success("✅ API connected")
except Exception as exc:
    meta, available_models, accuracies = None, ["xgboost", "logistic_regression"], {}
    st.sidebar.error(f"❌ Can't reach API: {exc}")
    st.sidebar.info("Start it with:\n\n`uvicorn fast_api:app --reload`")

model_name = st.sidebar.selectbox("Model", available_models)
if model_name in accuracies:
    st.sidebar.caption(f"Test accuracy: **{accuracies[model_name]:.3f}**")
st.sidebar.markdown("---")
st.sidebar.caption("CAP 3764 Team 3.\n\n**Target**: 1 = next-day close HIGHER, 0 = LOWER.")

st.title("📈 AAPL Next-Day Direction Predictor")
st.markdown("Predict whether Apple (AAPL) closes UP or DOWN tomorrow from today's "
            "news sentiment and price/technical signals.")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"single_{k}", v)

tab_single, tab_batch, tab_stats, tab_about = st.tabs(
    ["🎯 Single Prediction", "📊 Batch Prediction", "📚 Model Stats", "ℹ️ About"])


# Tab 1 
with tab_single:
    section("1) Feature source", "Choose manual entry, API test rows, or upload a CSV row.")
    src = st.radio("Single-row source",
                   ["Manual", "Built-in test dataset", "Upload my own CSV"], horizontal=True)

    pick_df = None
    if src == "Built-in test dataset":
        n = st.slider("How many rows to pull from the API", 20, 83, 83)
        if st.button("Fetch / refresh rows from API"):
            try:
                st.session_state["ref_df"] = pd.DataFrame(
                    api_get(api_url, "/sample_test_data", limit=n)["rows"])
                st.success(f"Loaded {len(st.session_state['ref_df'])} rows.")
            except Exception as exc:
                st.error(f"Could not load sample data: {exc}")
        pick_df = st.session_state.get("ref_df")
    elif src == "Upload my own CSV":
        up = st.file_uploader("Upload CSV for single-row pick", type=["csv"])
        if up:
            try:
                pick_df = pd.read_csv(up)
                st.dataframe(pick_df, use_container_width=True, height=220)
            except Exception as exc:
                st.error(f"Could not parse uploaded CSV: {exc}")

    if pick_df is not None and len(pick_df):
        ix = st.selectbox("Choose row (applies automatically)", range(len(pick_df)),
                          format_func=lambda i: f"{i} | {pick_df.iloc[i].get('date','n/a')}")
        key = f"{src}:{ix}:{len(pick_df)}"
        if st.session_state.get("_last") != key:
            row = pick_df.iloc[int(ix)]
            for c in FEATURES:
                if c in row.index and pd.notna(row[c]):
                    st.session_state[f"single_{c}"] = row[c]
            st.session_state["single_date"] = str(row["date"])[:10] if "date" in row.index else None
            st.session_state["single_target"] = (int(row["target"])
                if "target" in row.index and pd.notna(row["target"]) else None)
            st.session_state["_last"] = key

    section("2) Feature values and predict", "Edit fields if needed, then run the model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sentiment**")
        for f in ["avg_sentiment", "rolling_3d_sentiment", "rolling_7d_sentiment",
                  "sentiment_std_5d", "sentiment_vs_trend"]:
            st.number_input(f, key=f"single_{f}", format="%.4f")
    with c2:
        st.markdown("**Price**")
        for f in ["daily_return", "price_momentum_5d", "high_low_range"]:
            st.number_input(f, key=f"single_{f}", format="%.4f")
        for f in ["ma_5", "ma_10"]:
            st.number_input(f, key=f"single_{f}", format="%.2f")
    with c3:
        st.markdown("**Volume / context**")
        st.number_input("volume", key="single_volume", step=1_000_000)
        st.selectbox("day_of_week", [0, 1, 2, 3, 4], key="single_day_of_week",
                     format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri"][d])
        st.radio("lagged_target (yesterday)", [0, 1], key="single_lagged_target",
                 horizontal=True, format_func=lambda v: "DOWN (0)" if v == 0 else "UP (1)")

    if st.button("🎯 Predict", type="primary", use_container_width=True):
        payload = {f: st.session_state[f"single_{f}"] for f in FEATURES}
        for k in ("volume", "day_of_week", "lagged_target"):
            payload[k] = int(payload[k])
        try:
            res = api_post("/predict", params={"model": model_name}, json=payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            lbl, p_up, p_dn = res["label"], res["probability_up"], res["probability_down"]
            col, arrow = ("#1b5e20", "▲") if lbl == "UP" else ("#b71c1c", "▼")
            st.markdown(
                f"<div style='padding:24px;border-radius:14px;background:#f2f2f2;border:3px solid {col}'>"
                f"<p style='margin:0 0 8px;text-align:center;font-weight:700;color:#000'>Prediction + confidence</p>"
                f"<h1 style='margin:0;text-align:center;color:{col}'>{arrow} Next session: {lbl}</h1>"
                f"<p style='text-align:center;font-size:34px;margin:10px 0 0;color:#000'>"
                f"Winner confidence: <b style='color:#000'>{res['confidence']*100:.1f}%</b> · model "
                f"<code style='font-weight:700;color:#000;background:#e8e8e8;padding:2px 6px;border-radius:4px'>{res['model']}</code></p>"
                f"<p style='text-align:center;font-size:42px;margin:10px 0 0'>"
                f"<b style='color:#1b5e20'>P(UP) {p_up*100:.1f}%</b>&nbsp;&nbsp;&nbsp;"
                f"<b style='color:#b71c1c'>P(DOWN) {p_dn*100:.1f}%</b></p></div>",
                unsafe_allow_html=True)

            try:
                ctx = api_get(api_url, "/price_context",
                              days=20, end_date=st.session_state.get("single_date") or "")
                d = pd.DataFrame(ctx["rows"])
                d["date"] = pd.to_datetime(d["date"])
                d["close"] = pd.to_numeric(d["close"])
                d = d.dropna().sort_values("date")
                fig = go.Figure(go.Scatter(x=d["date"], y=d["close"], mode="lines+markers",
                                           line=dict(color="#263238", width=2),
                                           marker=dict(size=8, color="#607d8b")))
                lx, ly = d["date"].iloc[-1], float(d["close"].iloc[-1])
                dy = max((d["close"].max() - d["close"].min()) * 0.12, 0.5)
                fig.add_annotation(x=lx, y=ly + dy, showarrow=False,
                                   text=f"<b>Model (next day)</b> {arrow} <b>{lbl}</b>",
                                   font=dict(color=col, size=14),
                                   bgcolor="rgba(255,255,255,.92)",
                                   bordercolor=col, borderwidth=1, borderpad=4)
                tgt = st.session_state.get("single_target")
                if tgt is not None:
                    a_lbl = "UP" if tgt == 1 else "DOWN"
                    a_col, a_arr = ("#1b5e20", "▲") if a_lbl == "UP" else ("#b71c1c", "▼")
                    a_text = f"<b>Actual next session</b> {a_arr} <b>{a_lbl}</b>"
                else:
                    a_col, a_text = "#424242", "<b>Actual next session:</b> not in dataset"
                fig.add_annotation(x=lx, y=ly + 2.2 * dy, showarrow=False, text=a_text,
                                   font=dict(color=a_col, size=14),
                                   bgcolor="rgba(255,255,255,.92)",
                                   bordercolor=a_col, borderwidth=1, borderpad=4)
                fig.update_layout(title="Recent price context", xaxis_title="Date",
                                  yaxis_title="Close (USD)", height=480, showlegend=False)
                st.plotly_chart(style(fig), use_container_width=True)
            except Exception as exc:
                st.caption(f"Price context unavailable: {exc}")


# Tab 2 
with tab_batch:
    st.subheader("Batch prediction")
    st.caption("Run on the pre-loaded test dataset (Sep–Dec 2024) or upload your own CSV.")
    source = st.radio("Data source", ["Built-in test dataset", "Upload my own CSV"], horizontal=True)
    batch_df = None
    if source == "Built-in test dataset":
        limit = st.slider("Number of rows", 1, 83, 20)
        try:
            batch_df = pd.DataFrame(api_get(api_url, "/sample_test_data", limit=limit)["rows"])
        except Exception as exc:
            st.error(f"Could not fetch built-in dataset: {exc}")
    else:
        up = st.file_uploader("Choose a CSV file", type=["csv"])
        if up:
            try:
                batch_df = pd.read_csv(up)
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

    if batch_df is not None:
        st.markdown(f"**Preview ({len(batch_df)} rows):**")
        st.dataframe(batch_df, use_container_width=True, height=420)
        missing = [c for c in FEATURES if c not in batch_df.columns]
        if missing:
            st.error(f"Data is missing required columns: {missing}")
        elif st.button("🚀 Run batch prediction", type="primary", use_container_width=True):
            try:
                res = predict_batch(batch_df, model_name)
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
            else:
                preds = pd.DataFrame(res["predictions"])
                merged = pd.concat([batch_df.dropna(subset=FEATURES).reset_index(drop=True), preds], axis=1)
                cols = st.columns(4)
                for c, label, val in zip(cols,
                        ["Rows scored", "Predicted UP", "Predicted DOWN", "Mean P(UP)"],
                        [res["n_rows"], res["up_count"], res["down_count"],
                         f"{res['mean_probability_up']*100:.1f}%"]):
                    c.metric(label, val)

                if {"date", "close"}.issubset(merged.columns):
                    section("📈 1. Price chart with prediction overlay",
                            "**Line:** realized close. **Markers:** model prediction "
                            "(green ▲= UP, red ▼= DOWN).")
                    d = merged.copy()
                    d["date"] = pd.to_datetime(d["date"])
                    d["close"] = pd.to_numeric(d["close"])
                    d = d.dropna(subset=["date", "close"]).sort_values("date")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines",
                                             name="Close", line=dict(color="#263238", width=2.5)))
                    for v, name, c, sym in [(1, "Predicted UP", "#2e7d32", "triangle-up"),
                                            (0, "Predicted DOWN", "#c62828", "triangle-down")]:
                        sub = d[d["prediction"].astype(int) == v]
                        if len(sub):
                            fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"],
                                                     mode="markers", name=name,
                                                     marker=dict(symbol=sym, size=11, color=c)))
                    fig.update_layout(title="AAPL close with predicted next-day direction",
                                      xaxis_title="Date", yaxis_title="Close (USD)",
                                      height=480, hovermode="x unified")
                    st.plotly_chart(style(fig), use_container_width=True)

                if "target" in merged.columns and merged["target"].notna().any():
                    yt = merged["target"].astype(int).values
                    yp = merged["prediction"].astype(int).values
                    ok = yt == yp
                    section("2. Prediction vs actual",
                            "Each bar is one row. Green = match, Red = mismatch.")
                    st.metric("Accuracy on this batch", f"{ok.mean():.1%}")
                    fig = go.Figure(go.Bar(
                        x=pd.to_datetime(merged["date"]), y=np.ones(len(merged)),
                        marker_color=["#43a047" if o else "#e53935" for o in ok],
                        text=["Correct" if o else "Wrong" for o in ok],
                        textposition="inside", insidetextfont=dict(color="#000", size=11)))
                    fig.update_layout(yaxis=dict(showticklabels=False, range=[0, 1.15]),
                                      xaxis_title="Date", height=280, showlegend=False)
                    st.plotly_chart(style(fig), use_container_width=True)

                    section("Evaluation table", "Same info as the bar strip — scan dates and labels side by side.")
                    tbl = pd.DataFrame({
                        "Date": pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d"),
                        "Actual": pd.Series(yt).map({0: "DOWN", 1: "UP"}),
                        "Predicted": pd.Series(yp).map({0: "DOWN", 1: "UP"}),
                        "P(UP)": merged["probability_up"].astype(float),
                        "Match": np.where(ok, "Correct", "Wrong"),
                    })
                    st.dataframe(
                        tbl.style.apply(
                            lambda r: [f"background-color:{'#689f38' if r['Match']=='Correct' else '#d84315'};"
                                       "color:#fff;font-weight:700;"] * len(r), axis=1),
                        use_container_width=True, hide_index=True)

                st.download_button("⬇️ Download predictions as CSV",
                                   data=merged.to_csv(index=False).encode(),
                                   file_name="batch_predictions.csv", mime="text/csv")
                with st.expander("Per-row model output (raw)"):
                    st.dataframe(preds, use_container_width=True)


# Tab 3
with tab_stats:
    st.subheader("Model stats")
    st.caption("Side-by-side diagnostics on the API test slice for both models.")
    try:
        clean = pd.DataFrame(api_get(api_url, "/sample_test_data", limit=83)["rows"]) \
                  .dropna(subset=FEATURES).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not load sample test data for stats: {exc}")
        clean = None

    if clean is not None and len(clean):
        for mdl in [m for m in ["xgboost", "logistic_regression"] if m in available_models]:
            st.markdown("---")
            section("XGBoost" if mdl == "xgboost" else "Logistic Regression",
                    "Feature importance, classification report, and confusion matrix.")

            section("1) Feature importance", "Higher bars = stronger global contribution.")
            try:
                imp = api_get(api_url, "/feature_importance", model=mdl)["importance"]
                d = pd.DataFrame({"f": list(imp), "v": list(imp.values())}).sort_values("v", ascending=False)
                fig = go.Figure(go.Bar(x=d["v"], y=d["f"], orientation="h",
                                       marker_color="#1565c0",
                                       text=[f"{v:.3f}" for v in d["v"]],
                                       textposition="outside"))
                fig.update_layout(title=f"{mdl}: feature importance",
                                  xaxis_title="Relative importance",
                                  height=420, margin=dict(l=180, r=90, t=70, b=50))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style(fig), use_container_width=True)
            except Exception as exc:
                st.warning(f"Feature importance unavailable for {mdl}: {exc}")

            try:
                preds_df = pd.DataFrame(predict_batch(clean, mdl)["predictions"])
            except Exception as exc:
                st.warning(f"Could not score test slice for {mdl}: {exc}"); continue

            merged = pd.concat([clean, preds_df], axis=1)
            if "target" in merged.columns and merged["target"].notna().any():
                yt = merged["target"].astype(int).values
                yp = merged["prediction"].astype(int).values
                section("2) Classification report", "Precision, recall, F1, support on the same test rows.")
                st.metric(f"{mdl} accuracy on test slice", f"{(yt == yp).mean():.1%}")
                rep = classification_report(yt, yp, target_names=["DOWN (0)", "UP (1)"],
                                            output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rep).T.round(3).reset_index().rename(columns={"index": "label"}),
                             use_container_width=True, hide_index=True)

                section("3) Confusion matrix", "Rows = actual classes, columns = predicted classes.")
                cm = confusion_matrix(yt, yp, labels=[0, 1])
                fig = go.Figure(go.Heatmap(z=cm, x=["Pred DOWN (0)", "Pred UP (1)"],
                                           y=["Actual DOWN (0)", "Actual UP (1)"],
                                           text=cm.astype(str), texttemplate="%{text}",
                                           textfont=dict(size=14), colorscale="Blues"))
                fig.update_layout(title=f"{mdl}: confusion matrix", height=360)
                st.plotly_chart(style(fig), use_container_width=True)
            else:
                st.warning(f"No target labels in test rows for {mdl}.")


# Tab 4
with tab_about:
    st.subheader("About this app")
    st.markdown("""
Frontend for a FastAPI service that serves the two models trained in
`notebooks/04_models.ipynb`:

- **Logistic Regression** (StandardScaler → LR, C=0.1)
- **XGBoost** (300 trees, depth 4, lr 0.05)

Binary classification of **AAPL next-day direction** (UP vs DOWN) from
13 features combining news sentiment signals with price/volume technicals.

### Feature list
""")
    st.code("\n".join(FEATURES))
    if meta is not None:
        st.markdown("### Live API metadata"); st.json(meta)
    else:
        st.info("Start the FastAPI service to see live metadata: `uvicorn fast_api:app --reload`")