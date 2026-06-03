"""
ml_engine.py
ML pipeline: TF-IDF embedding + Cosine Similarity + KNN logika + What-if risk model.
Simulira Sentence-Transformers + FAISS pristup bez heavy dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")


class MediConnectEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.patients_df = None
        self.doctors_df = None
        self.specialty_symptoms = None
        self.vectorizer = None
        self.patient_vectors = None
        self.knn_model = None
        self.label_encoder = None

    # ── Učitavanje podataka ──────────────────────────────────────────────────
    def load_data(self):
        self.patients_df = pd.read_csv(os.path.join(self.data_dir, "patients.csv"))
        self.doctors_df = pd.read_csv(os.path.join(self.data_dir, "doctors.csv"))
        with open(os.path.join(self.data_dir, "specialty_symptoms.json"), encoding="utf-8") as f:
            self.specialty_symptoms = json.load(f)
        return self

    # ── TF-IDF Embedding (simulira Sentence-Transformers) ────────────────────
    def build_embeddings(self):
        """
        Kreira TF-IDF vektore simptoma. U produkciji bi se koristili
        Sentence-Transformers embeddings + FAISS index.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=500,
            analyzer='word',
            token_pattern=r"[^;,]+",
        )
        corpus = self.patients_df["symptoms"].fillna("").tolist()
        self.patient_vectors = self.vectorizer.fit_transform(corpus)
        return self

    # ── KNN Klasifikator za specijalizaciju ──────────────────────────────────
    def train_knn(self, n_neighbors=5):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.patients_df["specialty"])
        self.knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            weights="distance"
        )
        self.knn_model.fit(self.patient_vectors, y)
        return self

    def fit(self):
        """Cijeli pipeline."""
        self.load_data()
        self.build_embeddings()
        self.train_knn()
        return self

    # ── Pretraga sličnih pacijenata (FAISS-stil) ─────────────────────────────
    def find_similar_patients(self, query_symptoms: str, top_k=5):
        query_vec = self.vectorizer.transform([query_symptoms])
        sims = cosine_similarity(query_vec, self.patient_vectors).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        similar = self.patients_df.iloc[top_indices].copy()
        similar["similarity_score"] = (sims[top_indices] * 100).round(1)
        return similar

    # ── Preporuka doktora ────────────────────────────────────────────────────
    def recommend_doctors(self, query_symptoms: str, urgency: str = "umereno",
                          age: int = 40, top_k=3):
        """
        Scoring formula:
          score = 0.45 × sim_score
                + 0.25 × success_rate_norm
                + 0.15 × wait_penalty
                + 0.10 × experience_norm
                + 0.05 × urgency_bonus
        """
        query_vec = self.vectorizer.transform([query_symptoms])
        sims = cosine_similarity(query_vec, self.patient_vectors).flatten()

        # Agregiraj prosječnu sličnost po specijalizaciji
        df = self.patients_df.copy()
        df["sim"] = sims
        specialty_sim = df.groupby("specialty")["sim"].mean().reset_index()
        specialty_sim.columns = ["specialty", "avg_sim"]

        # Mergeuj s doktorima
        doc_df = self.doctors_df.merge(specialty_sim, on="specialty", how="left")
        doc_df["avg_sim"] = doc_df["avg_sim"].fillna(0)

        # Normalizacija
        doc_df["sim_norm"] = doc_df["avg_sim"] / (doc_df["avg_sim"].max() + 1e-9)
        doc_df["success_norm"] = doc_df["success_rate"] / 100
        doc_df["wait_norm"] = 1 - (doc_df["avg_wait_days"] / doc_df["avg_wait_days"].max())
        doc_df["exp_norm"] = doc_df["experience_years"] / doc_df["experience_years"].max()

        # Urgency bonus
        urgency_map = {"hitno": 0.3, "umereno": 0.0, "blago": -0.1}
        ub = urgency_map.get(urgency, 0.0)
        # Za hitne slučajeve penalizujemo dugo čekanje
        wait_weight = 0.25 if urgency == "hitno" else 0.15

        doc_df["final_score"] = (
            0.45 * doc_df["sim_norm"] +
            0.25 * doc_df["success_norm"] +
            wait_weight * doc_df["wait_norm"] +
            0.10 * doc_df["exp_norm"] +
            0.05 * (1 + ub)
        )
        doc_df["final_score"] = (doc_df["final_score"] * 100).round(1)
        doc_df["similarity_pct"] = (doc_df["avg_sim"] * 100).round(1)

        result = doc_df.sort_values("final_score", ascending=False).head(top_k)
        return result.reset_index(drop=True)

    # ── Predikcija čekanja ────────────────────────────────────────────────────
    def predict_wait_time(self, specialty: str, urgency: str):
        row = self.doctors_df[self.doctors_df["specialty"] == specialty]
        if row.empty:
            return 5
        base = float(row["avg_wait_days"].values[0])
        multiplier = {"hitno": 0.5, "umereno": 1.0, "blago": 1.3}.get(urgency, 1.0)
        return max(1, round(base * multiplier))

    # ── What-if Risk Engine ──────────────────────────────────────────────────
    def whatif_risk(self, symptoms: str, urgency: str, days_delay: int, age: int = 40):
        """
        Model rizika pogoršanja:
          risk(t) = base_risk + symptom_factor + age_factor + delay_factor
        """
        n_symptoms = len([s for s in symptoms.split(";") if s.strip()])
        base = {"hitno": 42, "umereno": 18, "blago": 7}.get(urgency, 18)
        symptom_factor = n_symptoms * 2.8
        age_factor = max(0, (age - 35) * 0.25)
        delay_factor = days_delay * ({"hitno": 3.5, "umereno": 1.8, "blago": 0.9}.get(urgency, 1.8))
        risk = base + symptom_factor + age_factor + delay_factor
        return min(97, max(3, round(risk, 1)))

    def whatif_scenarios(self, symptoms: str, urgency: str, age: int = 40):
        """Generiše 4 scenarija: danas, +3d, +7d, +14d."""
        scenarios = []
        for days in [0, 3, 7, 14]:
            risk = self.whatif_risk(symptoms, urgency, days, age)
            label = {0: "Danas", 3: "+3 dana", 7: "+7 dana", 14: "+14 dana"}[days]
            scenarios.append({"days": days, "label": label, "risk": risk})
        return scenarios

    # ── Dashboard statistike ─────────────────────────────────────────────────
    def dashboard_stats(self):
        df = self.patients_df
        return {
            "total_patients": len(df),
            "avg_success_rate": round(self.doctors_df["success_rate"].mean(), 1),
            "avg_wait_days": round(self.doctors_df["avg_wait_days"].mean(), 1),
            "specialties_count": df["specialty"].nunique(),
            "urgency_dist": df["urgency"].value_counts().to_dict(),
            "specialty_dist": df["specialty"].value_counts().to_dict(),
            "avg_risk_by_urgency": df.groupby("urgency")["risk_score"].mean().round(1).to_dict(),
            "success_by_specialty": df.groupby("specialty")["treatment_success"].mean().mul(100).round(1).to_dict(),
        }

    # ── Sačuvaj / učitaj model ────────────────────────────────────────────────
    def save(self, path="models/engine.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="models/engine.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    engine = MediConnectEngine(data_dir="data")
    engine.fit()
    engine.save("models/engine.pkl")
    print("✅ Model treniran i sačuvan")

    # Test
    query = "bol u grudima; otežano disanje; palpitacije"
    recs = engine.recommend_doctors(query, urgency="hitno", age=55)
    print("\n🩺 Top 3 preporuke za:", query)
    for _, row in recs.iterrows():
        print(f"  {row['name']} ({row['specialty']}) — Score: {row['final_score']}%")

    scenarios = engine.whatif_scenarios(query, "hitno", age=55)
    print("\n⚡ What-if scenariji:")
    for s in scenarios:
        print(f"  {s['label']}: rizik {s['risk']}%")
