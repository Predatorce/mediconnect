# ⚕️ MediConnect AI — Sistem za preporuku doktora

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Inteligentni sistem koji pomaže pacijentima da pronađu pravog specijaliste u pravo vrijeme — koristeći mašinsko učenje, analizu simptoma i prediktivni model rizika.**

---

## 🎯 Pregled projekta

MediConnect AI rješava realan problem: pacijent ima simptome, ali ne zna kod kojeg specijaliste da ide, koliko će čekati, niti koji doktor ima najbolji "fit" za njegov slučaj.

Sistem kombinuje:
- **KNN pretragu** nad bazom 500 simuliranih pacijenata
- **TF-IDF embedding** simptoma (produkciona verzija: Sentence-Transformers + FAISS)
- **Cosine Similarity** za rangiranje sličnih slučajeva
- **Prediktivni What-if model** za procjenu rizika pogoršanja

---

## ✨ Funkcionalnosti

### 🔍 Pretraga i preporuka
- Unos simptoma slobodnim tekstom **ili** checkboxovima
- Top 3 preporuke doktora sa detaljnim objašnjenjem scoring-a
- Prikaz 5 najsličnijih pacijenata iz baze (KNN)
- Predviđanje vremena čekanja na termin

### 📊 Analytics Dashboard
- KPI metrike: ukupni pacijenti, stopa uspjeha, prosječno čekanje
- Interaktivni Plotly grafikoni:
  - Distribucija pacijenata po specijalizaciji
  - Distribucija urgentnosti (pie chart)
  - Stopa uspjeha po specijalizaciji
  - Prosječni rizik po urgentnosti
- Kompletna tabela doktora s performansama

### ⚡ What-if Simulator
- Gauge indikator trenutnog rizika
- Scenariji: danas / +3d / +7d / +14d
- Interaktivni slider (0–30 dana)
- Automatska klasifikacija preporuke: HITNO / USKORO / PLANIRANO

### 📋 Baza pacijenata
- Pretraživanje i filtriranje 500 pacijenata
- Filteri: specijalizacija, urgentnost, raspon starosti

---

## 🧠 ML Pipeline

```
Korisnik unese simptome
        │
        ▼
  TF-IDF Vektorizacija
  (ngram_range=(1,2), max_features=500)
        │
        ▼
  Cosine Similarity
  vs. baza 500 pacijenata
        │
        ├──► Top-K slični pacijenti (KNN, k=5)
        │
        ▼
  Scoring po specijalizaciji:
  score = 0.45×sim + 0.25×success_rate
        + 0.15×wait_penalty + 0.10×experience
        + 0.05×urgency_bonus
        │
        ▼
  Rangiranje doktora
  + What-if Risk Model
```

### Scoring formula (detaljna)

| Komponenta | Težina | Opis |
|-----------|--------|------|
| Cosine Similarity | 45% | TF-IDF sličnost simptoma vs. baza |
| Success Rate | 25% | Stopa uspješnog liječenja doktora |
| Wait Penalty | 15% | Inverzno proporcionalno čekanju (hitni slučajevi: 25%) |
| Experience | 10% | Normalizovane godine iskustva |
| Urgency Bonus | 5% | Korekcija za hitnost slučaja |

### What-if Risk Model

```python
risk(t) = base_urgency + symptom_factor + age_factor + t × delay_rate

gdje je:
  base_urgency  = {hitno: 42, umereno: 18, blago: 7}
  symptom_factor = n_symptoms × 2.8
  age_factor    = max(0, (age - 35) × 0.25)
  delay_rate    = {hitno: 3.5, umereno: 1.8, blago: 0.9}
```

---

## 🛠️ Tehnički stack

| Tehnologija | Uloga |
|-------------|-------|
| **Python 3.9+** | Jezik |
| **Streamlit** | Web aplikacija / UI |
| **scikit-learn** | TF-IDF, KNN, Cosine Similarity |
| **pandas / numpy** | Obrada podataka |
| **Plotly** | Interaktivni grafikoni |
| **pickle** | Serijalizacija modela |

> **Produkciona ekstenzija** bi koristila: `sentence-transformers` (multilingual-MiniLM-L6-v2) + `FAISS` (IndexFlatIP) za embedding pretragu u realnom vremenu.





## 📁 Struktura projekta

```
mediconnect-ai/
├── app.py                    # Streamlit web aplikacija
├── ml_engine.py              # ML pipeline (embeddings + KNN + scoring)
├── requirements.txt          # Python zavisnosti
├── README.md                 # Dokumentacija
├── data/
│   ├── generate_dataset.py   # Generator simuliranog dataseta
│   ├── patients.csv          # 500 simuliranih pacijenata
│   ├── doctors.csv           # 8 doktora sa karakteristikama
│   └── specialty_symptoms.json  # Mapiranje simptoma po specijalizaciji
└── models/
    └── engine.pkl            # Trenirani ML model (serijalizovan)
```

---

## 📊 Dataset

Simulirani dataset generisan za akademske svrhe (u skladu sa opisom projekta: *"realni ili simulirani dataset"*).

| Parametar | Vrijednost |
|-----------|------------|
| Broj pacijenata | 500 |
| Broj specijalizacija | 8 |
| Broj doktora | 8 |
| Broj simptoma | 72 (po specijalizaciji) |
| Urgentnost distribucija | 22% hitno, 48% umereno, 30% blago |

**Specijalizacije:** Kardiologija, Neurologija, Pulmologija, Gastroenterologija, Endokrinologija, Ortopedija, Dermatologija, Psihijatrija

---

## 🔭 Moguća proširenja (produkcija)

- [ ] **Sentence-Transformers** (`paraphrase-multilingual-MiniLM-L6-v2`) za semantičke embeddings
- [ ] **FAISS IndexFlatIP** za skalabilnu pretragu na milionima pacijenata
- [ ] **Integracija sa EHR sistemima** (FHIR API)
- [ ] **Autentifikacija pacijenata** (OAuth2)
- [ ] **Feedback loop** — doktori ocjenjuju preporuke, model se unapređuje
- [ ] **Multilingual NLP** za dijalekatske varijante simptoma
- [ ] **Explainable AI** (SHAP vrijednosti) za transparentnost preporuka

---

## 👤 Autor
Andrej Zajović 22/096 FIST

---

## 📄 Licenca

MIT License — slobodno korišćenje za akademske i istraživačke svrhe.
