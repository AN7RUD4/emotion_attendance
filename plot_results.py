import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
with open("health_history.json", "r") as f:
    data = json.load(f)["entries"]

# Convert to DataFrame
df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Extract date for daily averages
df["date"] = df["timestamp"].dt.date

# Create result directories
os.makedirs("results/raw", exist_ok=True)
os.makedirs("results/avg", exist_ok=True)

# --- Plot helper ---
def plot_and_save(x, y, kind, title, xlabel, ylabel, filename, folder="raw"):
    plt.figure(figsize=(10, 5))
    if kind == "line":
        plt.plot(x, y, marker="", linewidth=1.5)
    elif kind == "bar":
        plt.bar(x, y)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{folder}/{filename}", dpi=300)
    plt.close()

# ================== RAW PLOTS ==================
plot_and_save(df["timestamp"], df["avg_ear"], "line",
              "Trend of Eye Aspect Ratio (EAR)", "Timestamp", "EAR", "ear_trend.png", "raw")

plot_and_save(df["timestamp"], df["drowsy_percent"], "line",
              "Drowsiness Percentage Over Time", "Timestamp", "Drowsy %", "drowsiness_trend.png", "raw")

plot_and_save(df["timestamp"], df["stress_level"], "line",
              "Stress Level Trend", "Timestamp", "Stress Level", "stress_trend.png", "raw")

plot_and_save(df["timestamp"], df["fatigue_level"], "line",
              "Fatigue Level Trend", "Timestamp", "Fatigue Level", "fatigue_trend.png", "raw")

plot_and_save(df["timestamp"], df["health_risk_score"], "line",
              "Health Risk Score Trend", "Timestamp", "Risk Score", "health_risk_score.png", "raw")

plot_and_save(df["timestamp"], df["pallor_score"], "line",
              "Pallor Score Trend", "Timestamp", "Pallor Score", "pallor_trend.png", "raw")

plot_and_save(df["timestamp"], df["yellowing_score"], "line",
              "Yellowing Score Trend", "Timestamp", "Yellowing Score", "yellowing_trend.png", "raw")

# Distribution plots
plot_and_save(df["health_state"].value_counts().index,
              df["health_state"].value_counts().values, "bar",
              "Distribution of Health States", "Health State", "Count", "health_state_distribution.png", "raw")

plot_and_save(df["dominant_emotion"].value_counts().index,
              df["dominant_emotion"].value_counts().values, "bar",
              "Distribution of Emotions", "Emotion", "Count", "emotions_distribution.png", "raw")

# Fatigue vs Stress comparison
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["fatigue_level"], label="Fatigue", linewidth=1.5)
plt.plot(df["timestamp"], df["stress_level"], label="Stress", linewidth=1.5)
plt.title("Comparison: Fatigue vs Stress", fontsize=14, fontweight="bold")
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Level", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/raw/fatigue_stress_comparison.png", dpi=300)
plt.close()

# ================== DAILY AVERAGES ==================
daily_avg = df.groupby("date").mean(numeric_only=True).reset_index()

plot_and_save(daily_avg["date"], daily_avg["avg_ear"], "line",
              "Daily Average EAR", "Date", "Avg EAR", "ear_avg.png", "avg")

plot_and_save(daily_avg["date"], daily_avg["drowsy_percent"], "line",
              "Daily Average Drowsiness %", "Date", "Drowsy %", "drowsiness_avg.png", "avg")

plot_and_save(daily_avg["date"], daily_avg["stress_level"], "line",
              "Daily Average Stress Level", "Date", "Stress Level", "stress_avg.png", "avg")

plot_and_save(daily_avg["date"], daily_avg["fatigue_level"], "line",
              "Daily Average Fatigue Level", "Date", "Fatigue Level", "fatigue_avg.png", "avg")

plot_and_save(daily_avg["date"], daily_avg["health_risk_score"], "line",
              "Daily Average Health Risk Score", "Date", "Risk Score", "health_risk_avg.png", "avg")
