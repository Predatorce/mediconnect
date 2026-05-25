"""
generate_dataset.py
Generiše simulirani medicinski dataset sa 500 pacijenata.
"""

import pandas as pd
import numpy as np
import random
import json
import os

random.seed(42)
np.random.seed(42)

# ─── Doktori i specijalizacije ───────────────────────────────────────────────
DOCTORS = [
    {"doctor_id": 1, "name": "Dr. Amira Kovačević", "specialty": "Kardiologija",
     "success_rate": 94, "avg_wait_days": 3, "experience_years": 18, "rating": 4.8},
    {"doctor_id": 2, "name": "Dr. Nemanja Perić",   "specialty": "Neurologija",
     "success_rate": 91, "avg_wait_days": 5, "experience_years": 14, "rating": 4.6},
    {"doctor_id": 3, "name": "Dr. Jelena Marić",    "specialty": "Pulmologija",
     "success_rate": 88, "avg_wait_days": 2, "experience_years": 11, "rating": 4.5},
    {"doctor_id": 4, "name": "Dr. Stefan Đorđević", "specialty": "Gastroenterologija",
     "success_rate": 92, "avg_wait_days": 4, "experience_years": 16, "rating": 4.7},
    {"doctor_id": 5, "name": "Dr. Milica Stanić",   "specialty": "Endokrinologija",
     "success_rate": 89, "avg_wait_days": 7, "experience_years": 12, "rating": 4.4},
    {"doctor_id": 6, "name": "Dr. Aleksandar Tomić","specialty": "Ortopedija",
     "success_rate": 96, "avg_wait_days": 6, "experience_years": 22, "rating": 4.9},
    {"doctor_id": 7, "name": "Dr. Ivana Lazović",   "specialty": "Dermatologija",
     "success_rate": 87, "avg_wait_days": 1, "experience_years": 9,  "rating": 4.3},
    {"doctor_id": 8, "name": "Dr. Bojan Nikolić",   "specialty": "Psihijatrija",
     "success_rate": 85, "avg_wait_days": 8, "experience_years": 15, "rating": 4.2},
]

# ─── Simptomi po specijalizaciji ─────────────────────────────────────────────
SPECIALTY_SYMPTOMS = {
    "Kardiologija":        ["bol u grudima", "palpitacije", "otežano disanje", "visok pritisak",
                            "umor", "vrtoglavica", "bol u lijevoj ruci", "znojenje", "ubrzan puls"],
    "Neurologija":         ["jaka glavobolja", "migrena", "vrtoglavica", "trnci u rukama",
                            "gubitak pamćenja", "nesanica", "tremor ruku", "smetnje govora", "slabost mišića"],
    "Pulmologija":         ["suhi kašalj", "otežano disanje", "piskanje pri disanju", "bol u grudima",
                            "iskašljavanje", "bronhitis", "astma napadi", "kratkoća daha", "hronični kašalj"],
    "Gastroenterologija":  ["bol u stomaku", "mučnina", "povraćanje", "proljev", "zatvor",
                            "nadutost", "žgaravica", "gubitak apetita", "krv u stolici"],
    "Endokrinologija":     ["umor", "debljanje", "mrsavljenje", "prekomjerna žeđ", "učestalo mokrenje",
                            "prekomjerno znojenje", "intolerancijahlada", "gubitak kose", "suha koža"],
    "Ortopedija":          ["bol u leđima", "bol u zglobovima", "otok zgloba", "ukočenost",
                            "bol u kolenu", "bol u kuku", "bol u ramenu", "smanjeni obim pokreta", "krčanje u zglobu"],
    "Dermatologija":       ["osip na koži", "svrbež", "akne", "crvena koža", "plikovi",
                            "gubitak kose", "suha koža", "ekcem", "promjena boje kože"],
    "Psihijatrija":        ["anksioznost", "depresija", "nesanica", "napadi panike",
                            "hronični stres", "promjene raspoloženja", "socijalna izolacija", "gubitak motivacije", "razdražljivost"],
}

URGENCY_LEVELS = ["hitno", "umereno", "blago"]
GENDERS = ["M", "Ž"]

def generate_patient(pid):
    specialty = random.choice(list(SPECIALTY_SYMPTOMS.keys()))
    doctor = next(d for d in DOCTORS if d["specialty"] == specialty)

    all_symptoms = SPECIALTY_SYMPTOMS[specialty]
    n_symptoms = random.randint(2, 5)
    patient_symptoms = random.sample(all_symptoms, n_symptoms)

    # Dodaj ponekad simptome iz susjednih specijalizacija (realistično)
    if random.random() < 0.3:
        other_spec = random.choice(list(SPECIALTY_SYMPTOMS.keys()))
        extra = random.choice(SPECIALTY_SYMPTOMS[other_spec])
        if extra not in patient_symptoms:
            patient_symptoms.append(extra)

    age = random.randint(18, 80)
    urgency = random.choices(URGENCY_LEVELS, weights=[0.2, 0.5, 0.3])[0]
    wait_days = max(1, int(np.random.normal(doctor["avg_wait_days"], 1.5)))
    
    # Rizik pogoršanja: baziran na urgentnosti + broju simptoma + starosti
    base_risk = {"hitno": 45, "umereno": 20, "blago": 8}[urgency]
    risk = min(95, base_risk + len(patient_symptoms) * 3 + (age - 40) * 0.3 + random.gauss(0, 5))
    risk = max(5, round(risk, 1))
    
    treatment_success = random.random() < (doctor["success_rate"] / 100)
    
    return {
        "patient_id": pid,
        "age": age,
        "gender": random.choice(GENDERS),
        "symptoms": "; ".join(patient_symptoms),
        "symptoms_text": ", ".join(patient_symptoms),
        "urgency": urgency,
        "specialty": specialty,
        "doctor_id": doctor["doctor_id"],
        "doctor_name": doctor["name"],
        "wait_days": wait_days,
        "risk_score": risk,
        "treatment_success": int(treatment_success),
        "num_symptoms": len(patient_symptoms),
    }

# Generišemo dataset
patients = [generate_patient(i+1) for i in range(500)]
df = pd.DataFrame(patients)

os.makedirs("data", exist_ok=True)
df.to_csv("data/patients.csv", index=False)
pd.DataFrame(DOCTORS).to_csv("data/doctors.csv", index=False)

# Sačuvamo i JSON za laku upotrebu
with open("data/specialty_symptoms.json", "w", encoding="utf-8") as f:
    json.dump(SPECIALTY_SYMPTOMS, f, ensure_ascii=False, indent=2)

print(f"✅ Generisano {len(df)} pacijenata")
print(f"✅ {df['specialty'].nunique()} specijalizacija")
print(f"✅ Distribucija urgentnosti:\n{df['urgency'].value_counts()}")
print(df.head(3).to_string())
