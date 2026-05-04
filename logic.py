import os
import time
import socket
import json
import threading
import numpy as np
import pandas as pd
from collections import deque, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import paho.mqtt.client as mqtt

# ──────────────────────────────────────────────────────────────
#  CONSTANTS & CONFIG
# ──────────────────────────────────────────────────────────────
ACTIVITY_ORDER = ["Sitting", "Standing", "Walking", "Jogging"]

ACTIVITY_META = {
    "Sitting":    {"icon": "🪑", "color": "#4C9BE8", "energy": "Very Low"},
    "Standing":   {"icon": "🧍", "color": "#5DBB6B", "energy": "Low"},
    "Walking":    {"icon": "🚶", "color": "#F5A623", "energy": "High"},
    "Jogging":    {"icon": "🏃", "color": "#E84C4C", "energy": "Very High"},
}

# ──────────────────────────────────────────────────────────────
#  NETWORKING & TELEMETRY
# ──────────────────────────────────────────────────────────────

def get_local_ip():
    """Best-effort local LAN IP for phone configuration hints."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def process_sensor_chunk_shared(ax, ay, az, buffer, lock):
    """Common logic for appending data to buffer with G-force normalization."""
    try:
        vals = [float(ax), float(ay), float(az)]
        # Auto-convert if data is in Gs instead of m/s^2
        is_m_s2 = any(abs(v) > 2.5 for v in vals)
        if not is_m_s2:
            vals = [v * 9.80665 for v in vals]
        ax, ay, az = vals
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        with lock:
            buffer.append({
                "acc_x": ax, "acc_y": ay, "acc_z": az,
                "magnitude": mag, "ts": time.time()
            })
    except Exception:
        pass

def run_mqtt_client_thread(buffer, lock):
    """Background MQTT client for low-latency telemetry from cloud broker."""
    broker = 'broker.hivemq.com'
    port = 1883
    topic = "antigravity_har_project_123/sensors"
    
    print(f"📡 [LOGIC] Connecting to MQTT Broker {broker}:{port}")
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ [MQTT] Connected successfully to {broker}")
            client.subscribe(topic)
            print(f"📡 [MQTT] Subscribed to topic: {topic}")
        else:
            print(f"❌ [MQTT] Connection failed with code {rc}")

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            obj = json.loads(payload)
            accel = None
            
            # Try common keys
            for key in ['accelerometer', 'accel', 'linear', 'sensor']:
                if key in obj and isinstance(obj[key], list) and len(obj[key]) >= 3:
                    accel = obj[key]; break
            
            if not accel:
                # Fallback: find any 3-element numeric list
                for k, v in obj.items():
                    if isinstance(v, list) and len(v) >= 3:
                        if all(isinstance(x, (int, float)) for x in v[:3]):
                            accel = v; break
            
            if accel:
                process_sensor_chunk_shared(accel[0], accel[1], accel[2], buffer, lock)
                if len(buffer) % 10 == 0:
                    pass # Optional debug print
            else:
                print(f"⚠️ [MQTT] Unknown data format: {list(obj.keys())}")
        except Exception as e:
            pass # Ignore malformed messages

    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker, port, 60)
        client.loop_forever()
    except Exception as e:
        print(f"❌ [LOGIC] MQTT Client Error: {e}")

# ──────────────────────────────────────────────────────────────
#  CORE ML LOGIC
# ──────────────────────────────────────────────────────────────

def extract_features(window: pd.DataFrame) -> dict:
    """
    Enhanced feature extraction:
    - Original statistical features
    - Gravity + body separation
    - Pitch & Roll (orientation)
    - Jerk (motion change)
    - Zero-crossing rate (signal dynamics)
    """

    # -----------------------------
    # Ensure magnitude exists
    # -----------------------------
    if "magnitude" not in window.columns:
        w_vals = window[["acc_x", "acc_y", "acc_z"]].values
        mag = np.sqrt(np.sum(w_vals**2, axis=1))
    else:
        mag = window["magnitude"].values

    acc_x = window["acc_x"].values
    acc_y = window["acc_y"].values
    acc_z = window["acc_z"].values

    features = {}

    # ============================================================
    # 🔴 1. GRAVITY + BODY SPLIT
    # ============================================================
    gravity_x = pd.Series(acc_x).rolling(10, min_periods=1).mean().values
    gravity_y = pd.Series(acc_y).rolling(10, min_periods=1).mean().values
    gravity_z = pd.Series(acc_z).rolling(10, min_periods=1).mean().values

    body_x = acc_x - gravity_x
    body_y = acc_y - gravity_y
    body_z = acc_z - gravity_z

    # ============================================================
    # 🔴 2. PITCH & ROLL (ORIENTATION)
    # ============================================================
    pitch = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))
    roll  = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))

    features["pitch_mean"] = np.mean(pitch)
    features["pitch_std"]  = np.std(pitch)
    features["roll_mean"]  = np.mean(roll)
    features["roll_std"]   = np.std(roll)

    # ============================================================
    # 🔴 3. JERK (DERIVATIVE OF MOTION)
    # ============================================================
    jerk_x = np.diff(acc_x)
    jerk_y = np.diff(acc_y)
    jerk_z = np.diff(acc_z)

    features["jerk_x_mean"] = np.mean(jerk_x) if len(jerk_x) > 0 else 0
    features["jerk_y_mean"] = np.mean(jerk_y) if len(jerk_y) > 0 else 0
    features["jerk_z_mean"] = np.mean(jerk_z) if len(jerk_z) > 0 else 0

    features["jerk_x_std"] = np.std(jerk_x) if len(jerk_x) > 0 else 0
    features["jerk_y_std"] = np.std(jerk_y) if len(jerk_y) > 0 else 0
    features["jerk_z_std"] = np.std(jerk_z) if len(jerk_z) > 0 else 0

    # ============================================================
    # 🔴 4. ZERO-CROSSING RATE
    # ============================================================
    def zero_crossing_rate(signal):
        return ((signal[:-1] * signal[1:]) < 0).sum()

    features["zcr_x"] = zero_crossing_rate(acc_x)
    features["zcr_y"] = zero_crossing_rate(acc_y)
    features["zcr_z"] = zero_crossing_rate(acc_z)

    # ============================================================
    # 🔵 ORIGINAL FEATURES (UNCHANGED)
    # ============================================================
    for col, vals in {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "magnitude": mag
    }.items():
        features[f"{col}_mean"]   = np.mean(vals)
        features[f"{col}_std"]    = np.std(vals)
        features[f"{col}_max"]    = np.max(vals)
        features[f"{col}_min"]    = np.min(vals)
        features[f"{col}_energy"] = np.mean(vals**2)

    # ============================================================
    # 🔵 BODY FEATURES (MOVEMENT)
    # ============================================================
    for col, vals in {
        "body_x": body_x,
        "body_y": body_y,
        "body_z": body_z
    }.items():
        features[f"{col}_std"] = np.std(vals)
        features[f"{col}_energy"] = np.mean(vals**2)

    # ============================================================
    # 🔵 GRAVITY FEATURES (POSTURE)
    # ============================================================
    for col, vals in {
        "gravity_x": gravity_x,
        "gravity_y": gravity_y,
        "gravity_z": gravity_z
    }.items():
        features[f"{col}_mean"] = np.mean(vals)
        features[f"{col}_std"]  = np.std(vals)

    return features
def load_and_train_model(filepath="WISDM_ar_v1.1_raw.txt"):
    """Loads WISDM dataset, extracts features, and trains KMeans clustering with improved mapping."""
    if not os.path.exists(filepath):
        return None, None, None, None, None, None

    col_names = ["user", "activity", "timestamp", "acc_x", "acc_y", "acc_z"]
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().rstrip(";")
            parts = line.split(",")
            if len(parts) == 6:
                rows.append(parts)

    df = pd.DataFrame(rows, columns=col_names)

    for col in ["acc_x", "acc_y", "acc_z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["acc_x", "acc_y", "acc_z"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Magnitude
    df["magnitude"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    feature_rows = []
    for start in range(0, len(df) - 40 + 1, 20):
        w = df.iloc[start: start + 40]
        feature_rows.append(extract_features(w))

    feature_df = pd.DataFrame(feature_rows)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)

    # -----------------------------
    # KMeans Clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=6, n_init=20, random_state=42, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # ============================================================
    # 🔥 NEW: SMART CLUSTER ANALYSIS (FIX FOR STANDING ISSUE)
    # ============================================================
    cluster_stats = {}

    for c in range(6):
        cluster_data = feature_df.loc[cluster_labels == c]

        if len(cluster_data) == 0:
            continue

        cluster_stats[c] = {
            "energy": cluster_data["magnitude_energy"].mean(),
            "pitch": cluster_data["pitch_mean"].mean(),
            "roll": cluster_data["roll_mean"].mean(),
            "jerk": cluster_data["jerk_x_std"].mean()
        }

    # Sort clusters by energy
    sorted_clusters = sorted(cluster_stats, key=lambda x: cluster_stats[x]["energy"])

    cluster_to_activity = {}

    # -----------------------------
    # 🔴 LOW ENERGY → Sitting & Standing
    # -----------------------------
    low1 = sorted_clusters[0]
    low2 = sorted_clusters[1]

    # Use pitch (orientation) to separate
    if abs(cluster_stats[low1]["pitch"]) > abs(cluster_stats[low2]["pitch"]):
        cluster_to_activity[low1] = "Standing"
        cluster_to_activity[low2] = "Sitting"
    else:
        cluster_to_activity[low1] = "Sitting"
        cluster_to_activity[low2] = "Standing"

    # -----------------------------
    # 🟡 MID ENERGY → Walking
    # -----------------------------
    for c in sorted_clusters[2:5]:
        cluster_to_activity[c] = "Walking"

    # -----------------------------
    # 🔴 HIGH ENERGY → Jogging
    # -----------------------------
    cluster_to_activity[sorted_clusters[-1]] = "Jogging"

    # -----------------------------
    # Debug print (IMPORTANT)
    # -----------------------------
    print("\n🔍 Cluster Analysis:")
    for c in cluster_stats:
        print(f"Cluster {c}: {cluster_stats[c]} → {cluster_to_activity.get(c, 'Unassigned')}")

    return kmeans, scaler, cluster_to_activity, X_scaled, cluster_labels, feature_df

def detect_and_normalize_columns(df: pd.DataFrame):
    """Automatically detects accelerometer columns in a CSV file."""
    X_CANDS = ["ax (m/s^2)", "ax", "accel_x", "acc_x", "gfx", "x", "accelerationx"]
    Y_CANDS = ["ay (m/s^2)", "ay", "accel_y", "acc_y", "gfy", "y", "accelerationy"]
    Z_CANDS = ["az (m/s^2)", "az", "accel_z", "acc_z", "gfz", "z", "accelerationz"]

    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    for target, candidates in [("acc_x", X_CANDS), ("acc_y", Y_CANDS), ("acc_z", Z_CANDS)]:
        for cand in candidates:
            if cand.lower() in cols_lower:
                rename_map[cols_lower[cand.lower()]] = target; break

    if len(rename_map) < 3: return None, None

    df = df.rename(columns=rename_map)
    for col in ["acc_x", "acc_y", "acc_z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["acc_x", "acc_y", "acc_z"], inplace=True)

    if any("gf" in k.lower() for k in rename_map.keys()):
        for col in ["acc_x", "acc_y", "acc_z"]: df[col] *= 9.81

    df["magnitude"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    return df.reset_index(drop=True), rename_map

def predict_from_dataframe(df, kmeans, scaler, cluster_to_activity, window_size=40):
    """Vectorized prediction from a dataframe for batch analysis."""
    results = []
    for start in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[start: start + window_size]
        features = extract_features(window)
        feat_vec = np.array(list(features.values())).reshape(1, -1)
        if feat_vec.shape[1] != scaler.n_features_in_: continue
        
        feat_scaled = scaler.transform(feat_vec)
        cluster = kmeans.predict(feat_scaled)[0]
        activity = cluster_to_activity[cluster]
        
        # Real confidence
        dists = kmeans.transform(feat_scaled)[0]
        conf = (1 - dists[cluster] / (sum(dists) + 1e-9)) * 100
        
        results.append({"activity": activity, "confidence": conf})
    return results
