from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEATURES = ["Daily_Usage_Hours", "Sleep_Hours", "Study_Hours", "Social_Media_Hours",
            "Notifications_Per_Day", "Pickups_Per_Day"]
DEMO_DATASETS = {
    "balanced": ("demo_balanced_240.csv", "Balanced demo dataset · 240 records"),
    "high": ("demo_high_usage_180.csv", "High-usage demo dataset · 180 records"),
    "healthy": ("demo_healthy_habits_180.csv", "Healthy-habits demo dataset · 180 records"),
}
ACTIVE_DATA = None
ACTIVE_NAME = None

def prepare_data(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    if len(df) < 12:
        raise ValueError("Please provide at least 12 rows.")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).copy()
    if len(df) < 12:
        raise ValueError("At least 12 rows must contain valid numeric values.")
    if "Addiction_Level" not in df.columns:
        score = ((df["Daily_Usage_Hours"] >= 7).astype(int) +
                 (df["Sleep_Hours"] < 6).astype(int) +
                 (df["Study_Hours"] < 2).astype(int) +
                 (df["Social_Media_Hours"] >= 3).astype(int) +
                 (df["Pickups_Per_Day"] >= 100).astype(int))
        df["Addiction_Level"] = np.select([score >= 4, score >= 2], ["High", "Moderate"], default="Low")
    else:
        allowed = {"Low", "Moderate", "High"}
        df["Addiction_Level"] = df["Addiction_Level"].astype(str).str.title()
        invalid = sorted(set(df["Addiction_Level"]) - allowed)
        if invalid:
            raise ValueError("Addiction_Level must contain only Low, Moderate, or High.")
    return df

def analyze(df):
    df = prepare_data(df)
    X = df[FEATURES]
    scaled = StandardScaler().fit_transform(X)
    df["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    means = df.groupby("Cluster")["Daily_Usage_Hours"].mean().sort_values()
    names = {int(means.index[0]): "Low usage", int(means.index[1]): "Moderate usage", int(means.index[2]): "High usage"}
    df["Usage_Segment"] = df["Cluster"].map(names)

    y = df["Addiction_Level"]
    if y.nunique() > 1 and len(df) >= 16:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42,
                                                            stratify=y if y.value_counts().min() >= 2 else None)
        tree = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
        tree.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, tree.predict(X_test)) * 100
    else:
        tree = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
        tree.fit(X, y)
        accuracy = None
    importance = sorted([{"feature":f.replace("_"," "), "importance":round(float(v),3)}
                         for f,v in zip(FEATURES, tree.feature_importances_)],
                        key=lambda x:x["importance"], reverse=True)

    basket = pd.DataFrame({
        "High daily use (7h+)": df["Daily_Usage_Hours"] >= 7,
        "Short sleep (<6h)": df["Sleep_Hours"] < 6,
        "Low study time (<2h)": df["Study_Hours"] < 2,
        "High social media (3h+)": df["Social_Media_Hours"] >= 3,
        "Many pickups (100+)": df["Pickups_Per_Day"] >= 100,
        "High notifications (150+)": df["Notifications_Per_Day"] >= 150,
        "High addiction label": df["Addiction_Level"] == "High",
        "Moderate addiction label": df["Addiction_Level"] == "Moderate",
        "Low addiction label": df["Addiction_Level"] == "Low",
    }).astype(bool)
    frequent = apriori(basket, min_support=.12, use_colnames=True)
    rules_out=[]
    if len(frequent):
        rules=association_rules(frequent, metric="confidence", min_threshold=.55)
        if len(rules):
            rules=rules.sort_values(["lift","confidence"],ascending=False).head(8)
            for _,r in rules.iterrows():
                rules_out.append({"antecedents":", ".join(sorted(list(r["antecedents"]))),
                                  "consequents":", ".join(sorted(list(r["consequents"]))),
                                  "support":round(float(r["support"]),3),
                                  "confidence":round(float(r["confidence"]),3),
                                  "lift":round(float(r["lift"]),2)})
    counts=df["Addiction_Level"].value_counts().reindex(["Low","Moderate","High"],fill_value=0)
    seg=df["Usage_Segment"].value_counts().reindex(["Low usage","Moderate usage","High usage"],fill_value=0)
    return {"total":len(df),"avg_usage":round(float(df["Daily_Usage_Hours"].mean()),1),
            "avg_sleep":round(float(df["Sleep_Hours"].mean()),1),
            "high_pct":round(float((df["Addiction_Level"]=="High").mean()*100),1),
            "labels":list(counts.index),"label_values":[int(v) for v in counts.values],
            "segments":list(seg.index),"segment_values":[int(v) for v in seg.values],
            "importance":importance,"accuracy":round(float(accuracy),1) if accuracy is not None else None,
            "rules":rules_out,"preview":df.head(8).round(2).to_dict(orient="records"),
            "dataset_name":ACTIVE_NAME or "Uploaded dataset"}

@app.route("/", methods=["GET","POST"])
def index():
    global ACTIVE_DATA, ACTIVE_NAME
    error=None
    if request.method=="POST":
        mode=request.form.get("mode")
        if mode=="demo":
            key=request.form.get("demo_key","balanced")
            if key not in DEMO_DATASETS:
                error="Please choose one of the available demo datasets."
            else:
                filename,label=DEMO_DATASETS[key]
                try:
                    ACTIVE_DATA=pd.read_csv(DATA_DIR/filename)
                    ACTIVE_NAME=label
                except Exception as exc:
                    error="Could not load demo dataset: "+str(exc)
        elif mode=="upload":
            upload=request.files.get("dataset")
            if not upload or not upload.filename.lower().endswith(".csv"):
                error="Choose a valid .csv file first."
            else:
                try:
                    df=pd.read_csv(upload)
                    df=prepare_data(df)
                    ACTIVE_DATA=df
                    ACTIVE_NAME=secure_filename(upload.filename)
                except Exception as exc:
                    error=str(exc)
        return redirect(url_for("index", error=error) if error else url_for("index"))
    error=request.args.get("error")
    analysis=None
    if ACTIVE_DATA is not None:
        try:
            analysis=analyze(ACTIVE_DATA.copy())
        except Exception as exc:
            error=str(exc)
    return render_template("index.html", a=analysis, error=error, demo_datasets=DEMO_DATASETS)

@app.route("/reset")
def reset():
    global ACTIVE_DATA, ACTIVE_NAME
    ACTIVE_DATA=None
    ACTIVE_NAME=None
    return redirect(url_for("index"))

if __name__=="__main__":
    app.run(debug=True)
