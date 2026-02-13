from flask import Flask, render_template
from flask import request
import folium
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
ml_model = joblib.load("models/ml_risk_model.pkl")
dl_model = tf.keras.models.load_model("models/dl_retreat_model.h5")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data/glacier_climate.csv")

# -----------------------------
# ML RISK PREDICTION
# -----------------------------
def predict_ml_risk(glacier_name):

    glacier = df[df["glacier_name"] == glacier_name]

    if glacier.empty:
        return "UNKNOWN"

    features = glacier[[
        "length_km",
        "area_km2",
        "elevation_m",
        "retreat_rate_m_per_year",
        "temperature_increase"
    ]]

    prediction = ml_model.predict(features)

    return "HIGH" if prediction[0] == 1 else "LOW"


# -----------------------------
# DL FUTURE RETREAT PREDICTION
# -----------------------------
def predict_future_retreat(glacier_name):

    glacier = df[df["glacier_name"] == glacier_name]

    retreat_value = glacier["retreat_rate_m_per_year"].values.reshape(-1, 1)

    scaled = scaler.transform(retreat_value)
    scaled = scaled.reshape((1, 1, 1))

    future_scaled = dl_model.predict(scaled, verbose=0)
    future = scaler.inverse_transform(future_scaled)

    return round(float(future[0][0]), 2)


# -----------------------------
# HOME ROUTE - SHOW ALL 50 GLACIERS
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    selected_glacier = None

    # Create base map (no markers)
    map_glaciers = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles=None
    )

    # Satellite Layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(map_glaciers)

    folium.TileLayer("OpenStreetMap").add_to(map_glaciers)

    if request.method == "POST":
        glacier_name = request.form.get("glacier_name")
        selected_glacier = glacier_name

        glacier = df[df["glacier_name"] == glacier_name]

        if not glacier.empty:
            lat = glacier.iloc[0]["latitude"]
            lon = glacier.iloc[0]["longitude"]

            # Add marker only for selected glacier
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{glacier_name}</b><br><a href='/glacier/{glacier_name}'>View Detailed Analysis</a>"
            ).add_to(map_glaciers)
            

            map_glaciers.location = [lat, lon]
            map_glaciers.zoom_start = 6

    folium.LayerControl().add_to(map_glaciers)

    os.makedirs("static", exist_ok=True)
    map_file = "static/map.html"
    map_glaciers.save(map_file)

    glacier_list = sorted(df["glacier_name"].unique())

    return render_template(
        "index.html",
        map_file=map_file,
        glacier_list=glacier_list,
        selected_glacier=selected_glacier
    )


# -----------------------------
# GLACIER DETAILS PAGE
# -----------------------------
@app.route("/glacier/<name>")
def glacier_detail(name):

    glacier = df[df["glacier_name"] == name]

    if glacier.empty:
        return "Glacier Not Found"

    data = glacier.iloc[0].to_dict()

    risk = predict_ml_risk(name)
    future_retreat = predict_future_retreat(name)

    return render_template(
        "glacier_detail.html",
        name=name,
        data=data,
        risk=risk,
        future_retreat=future_retreat
    )

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)



