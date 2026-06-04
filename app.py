from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# GRAPH
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NEW MEDICAL REPORT GENERATOR
from report_generator import generate_medical_report

app = Flask(__name__)

history_file = "history.csv"
last_report = {}

# Create folders
if not os.path.exists("static"):
    os.makedirs("static")

# ================= MODEL =================
X_dummy = np.random.randint(0, 10, (200, 10))
y_dummy = np.array([0 if np.mean(x) < 4 else 1 if np.mean(x) < 7 else 2 for x in X_dummy])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)

log_model = LogisticRegression().fit(X_scaled, y_dummy)
rf_model = RandomForestClassifier().fit(X_scaled, y_dummy)
kmeans = KMeans(n_clusters=3).fit(X_scaled)
hier = AgglomerativeClustering(n_clusters=3)

# ================= CATEGORY =================
def get_categories(values):
    return {
        "Respiratory": round(np.mean([values[1], values[2]]), 2),
        "Neurological": round(np.mean([values[3], values[5], values[7]]), 2),
        "Fatigue":      round(np.mean([values[0], values[9]]), 2),
        "Mental":       round(np.mean([values[4], values[6], values[8]]), 2)
    }

# ================= GRAPHS =================
def generate_cluster_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_vis = X_scaled[:, :3]
    ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=kmeans.labels_)

    user = scaler.transform(np.array(last_report["symptoms"]).reshape(1, -1))
    ax.scatter(user[0, 0], user[0, 1], user[0, 2], c='red', s=200, marker='x')

    plt.savefig("static/cluster.png")
    plt.close()

def generate_category_graph(categories):
    names = list(categories.keys())
    values = list(categories.values())

    plt.figure()
    plt.bar(names, values)
    plt.title("Category Severity")
    plt.savefig("static/category.png")
    plt.close()

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    symptoms = [5] * 10
    ml = None
    categories = {}

    if request.method == "POST":

        name   = request.form.get("name")
        age    = request.form.get("age")
        gender = request.form.get("gender")

        values   = [float(request.form[f"f{i}"]) for i in range(1, 11)]
        symptoms = values

        # WEIGHTED MEAN
        weights    = np.array([1.2, 1.5, 2.0, 1.2, 1.0, 1.1, 1.8, 1.3, 1.4, 1.2])
        values_arr = np.array(values)

        score   = np.sum(values_arr * weights) / np.sum(weights)
        percent = round((score / 10) * 100, 2)

        if percent < 40:
            level = "Low Risk"
        elif percent < 70:
            level = "Medium Risk"
        else:
            level = "High Risk"

        result = (level, percent, "Follow healthy lifestyle")

        # ML
        arr        = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        ml = {
            "logistic":      int(log_model.predict(arr_scaled)[0]),
            "random_forest": int(rf_model.predict(arr_scaled)[0]),
            "kmeans":        int(kmeans.predict(arr_scaled)[0]),
            "hierarchical":  int(AgglomerativeClustering(n_clusters=3)
                                 .fit_predict(np.vstack([X_scaled, arr_scaled]))[-1])
        }

        # Categories
        categories = get_categories(values)

        # Save history
        df = pd.DataFrame([{
            "Name": name,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Risk": percent
        }])

        if not os.path.exists(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)

        global last_report
        last_report = {
            "patient":    {"name": name, "age": age, "gender": gender},
            "symptoms":   values,
            "result":     result,
            "ml":         ml,
            "categories": categories
        }

    try:
        history = pd.read_csv(history_file).to_dict(orient="records")
    except:
        history = []

    return render_template("index.html",
                           result=result,
                           symptoms=symptoms,
                           history=history,
                           ml=ml,
                           categories=categories)

# ================= GRAPH ROUTES =================
@app.route("/cluster_graph")
def cluster_graph():
    generate_cluster_graph()
    return send_file("static/cluster.png", mimetype='image/png')

@app.route("/category_graph")
def category_graph():
    generate_category_graph(last_report["categories"])
    return send_file("static/category.png", mimetype='image/png')

# ================= PDF (NEW MEDICAL REPORT) =================
@app.route("/download_report")
def download_report():
    path = generate_medical_report(last_report, "report.pdf")
    return send_file(path, as_attachment=True)

# ================= CHATBOT =================
@app.route("/chatbot", methods=["POST"])
def chatbot():

    msg = request.json["message"].lower()

    data = {
        "fatigue": {
            "reply": "Take proper rest and stay hydrated.",
            "steps": [
                "Lie down in a comfortable position",
                "Relax your body and mind",
                "Drink plenty of water",
                "Avoid heavy work",
                "Sleep for at least 7-8 hours"
            ],
            "image": "https://images.pexels.com/photos/3822622/pexels-photo-3822622.jpeg"
        },
        "cough": {
            "reply": "Drink warm fluids and avoid cold exposure.",
            "steps": [
                "Drink warm water frequently",
                "Take steam inhalation",
                "Avoid cold drinks",
                "Use honey with warm water",
                "Rest your throat"
            ],
            "image": "https://images.pexels.com/photos/3768911/pexels-photo-3768911.jpeg"
        },
        "breath": {
            "reply": "Practice breathing exercise daily.",
            "steps": [
                "Sit comfortably",
                "Close right nostril",
                "Inhale through left",
                "Exhale through right",
                "Repeat for 5-10 minutes"
            ],
            "image": "https://images.pexels.com/photos/4056723/pexels-photo-4056723.jpeg"
        },
        "headache": {
            "reply": "Take rest and reduce screen time.",
            "steps": [
                "Rest in a quiet room",
                "Close your eyes",
                "Avoid bright light",
                "Drink water",
                "Apply cold compress"
            ],
            "image": "https://images.pexels.com/photos/3760810/pexels-photo-3760810.jpeg"
        },
        "sleep": {
            "reply": "Improve your sleep routine.",
            "steps": [
                "Sleep at fixed time",
                "Avoid mobile before sleep",
                "Keep room dark",
                "Relax your mind",
                "Do meditation"
            ],
            "image": "https://images.pexels.com/photos/3771118/pexels-photo-3771118.jpeg"
        },
        "memory": {
            "reply": "Practice brain exercises.",
            "steps": [
                "Do puzzles daily",
                "Read books",
                "Meditate regularly",
                "Stay mentally active",
                "Get proper sleep"
            ],
            "image": "https://images.pexels.com/photos/5905709/pexels-photo-5905709.jpeg"
        },
        "chest": {
            "reply": "Consult doctor if pain persists.",
            "steps": [
                "Avoid heavy activity",
                "Take deep breaths slowly",
                "Rest properly",
                "Monitor pain level",
                "Seek medical help if severe"
            ],
            "image": "https://images.pexels.com/photos/4482900/pexels-photo-4482900.jpeg"
        },
        "smell": {
            "reply": "Smell training helps recovery.",
            "steps": [
                "Smell strong scents like lemon",
                "Practice daily twice",
                "Be consistent",
                "Avoid strong chemicals",
                "Stay patient"
            ],
            "image": "https://images.pexels.com/photos/6621462/pexels-photo-6621462.jpeg"
        },
        "fever": {
            "reply": "Monitor temperature regularly.",
            "steps": [
                "Drink fluids",
                "Take rest",
                "Use light clothes",
                "Check temperature frequently",
                "Take medicine if needed"
            ],
            "image": "https://images.pexels.com/photos/3985163/pexels-photo-3985163.jpeg"
        },
        "weakness": {
            "reply": "Eat nutritious food and rest.",
            "steps": [
                "Eat protein-rich diet",
                "Drink fluids",
                "Take proper rest",
                "Avoid stress",
                "Do light exercise"
            ],
            "image": "https://images.pexels.com/photos/414029/pexels-photo-414029.jpeg"
        }
    }

    response = data.get(msg, {
        "reply": "Stay healthy",
        "steps": [],
        "image": ""
    })

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)