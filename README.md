# MindMeter — Smartphone Addiction Analysis

A Python + Flask data-mining project demonstrating **Apriori**, **K-Means clustering**, and a **Decision Tree** with a browser-based dashboard.

## Features
- Dashboard with dataset KPIs and charts
- Apriori association rules with support, confidence, and lift
- K-Means clustering into three usage segments
- Decision Tree classification and feature importance
- Upload a CSV dataset and analyze it
- Three bundled synthetic demo datasets: balanced, high-usage, and healthy-habits
- Dataset-first workflow: no algorithms run until a demo or uploaded CSV is selected
- Responsive UI; no Firebase or database server required

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt in the extracted `Smartphone_Addiction_Analysis` folder.
4. Create and activate a virtual environment (recommended):
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages:
   ```bat
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```bat
   python app.py
   ```
7. Open **http://127.0.0.1:5000** in your browser.

## Dataset format
Required CSV columns (exact spelling):
- `Daily_Usage_Hours`
- `Sleep_Hours`
- `Study_Hours`
- `Social_Media_Hours`
- `Notifications_Per_Day`
- `Pickups_Per_Day`

Optional: `Addiction_Level` with values such as `Low`, `Moderate`, `High`. If omitted, the app creates a simple rule-based educational label. This label is a demonstration target, not a validated clinical assessment.

The bundled CSV is **synthetic sample data** created for demonstrating the algorithms; it is not a real survey or medical dataset. Use ethically collected, anonymized data for research claims.

## Algorithm notes
- **Apriori:** discretizes usage variables into boolean items and mines association rules (minimum support 0.12; confidence 0.55).
- **K-Means:** standardizes numeric features and forms three clusters, named by their average daily usage.
- **Decision Tree:** uses a stratified holdout when possible and reports test-set accuracy and feature importance. If the label was generated from thresholds, the accuracy reflects how well the tree approximates those thresholds.

## Project structure
```
Smartphone_Addiction_Analysis/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── smartphone_addiction.csv
│   └── smartphone_addiction_sample.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Troubleshooting
- If `py` is not recognized, install Python from https://www.python.org/downloads/ and enable **Add Python to PATH**.
- If pip is missing, try `python -m ensurepip --upgrade`, then `python -m pip install -r requirements.txt`.
- If port 5000 is busy, change the last line of `app.py` to `app.run(debug=True, port=5001)` and visit port 5001.

## Dataset-first workflow
When you open the site, no analysis is run automatically. Choose one of the three included demo CSV files or upload your own CSV. Only then do Apriori, K-Means, and Decision Tree run against the selected data. Use **Clear dataset & start over** to return to the selection screen.

Demo files:
- `data/demo_balanced_240.csv`
- `data/demo_high_usage_180.csv`
- `data/demo_healthy_habits_180.csv`

All bundled demo data is synthetic and intended for educational testing.
