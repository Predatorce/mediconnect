"""
app.py — MediConnect AI v1.1
Sistem za preporuku doktora i termina baziran na masinskom ucenju.
Verzija 1.1 - popravljen UI, navigacija uvijek vidljiva
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediConnect AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS v1.1 — svjetliji, cistiji UI ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Pozadina — malo svjetlija od v1.0 */
.stApp { background-color: #111827; }

/* Sidebar svjetliji */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2236 0%, #1e2a3a 100%) !important;
    border-right: 1px solid rgba(99,202,183,0.25) !important;
}

/* Radio buttons u sidebaru */
[data-testid="stSidebar"] .stRadio label {
    color: #c9d4e8 !important;
    font-size: 0.95rem !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    display: block !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,202,183,0.1) !important;
}

/* Glavni sadrzaj svjetliji */
.main .block-container {
    background: transparent;
    padding-top: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2d3d 0%, #1a2535 100%);
    border: 1px solid rgba(99,202,183,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #5ecfba;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #8899b0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Doc cards */
.doc-card-top {
    background: linear-gradient(135deg, #1e2d3d, #1a2a38);
    border: 1px solid rgba(99,202,183,0.5);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 0 25px rgba(99,202,183,0.1);
}
.doc-name { font-size: 1.1rem; font-weight: 600; color: #e8f0ff; }
.doc-spec { font-size: 0.85rem; color: #8899b0; margin-top: 2px; }
.badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 20px;
    background: rgba(99,202,183,0.2);
    color: #5ecfba;
    border: 1px solid rgba(99,202,183,0.4);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8f0ff;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.9rem;
    color: #8899b0;
    margin-bottom: 24px;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #1a2d3d, #162535);
    border: 1px solid rgba(99,202,183,0.2);
    border-left: 3px solid #5ecfba;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #c9d4e8;
    font-size: 0.88rem;
}

/* Input fields svjetliji */
.stTextArea textarea, .stTextInput input {
    background: #1e2d3d !important;
    color: #e8f0ff !important;
    border: 1px solid rgba(99,202,183,0.25) !important;
    border-radius: 10px !important;
}
.stMultiSelect > div {
    background: #1e2d3d !important;
    border: 1px solid rgba(99,202,183,0.25) !important;
    border-radius: 10px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(99,202,183,0.2) !important;
    color: #5ecfba !important;
    border: 1px solid rgba(99,202,183,0.4) !important;
}

/* Dugme */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2ea89a, #2d7ec4) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: transform 0.1s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(46,168,154,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(46,168,154,0.4) !important;
}

/* Divider svjetliji */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Sakrij Streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load engine ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    from ml_engine import MediConnectEngine
    e = MediConnectEngine(data_dir="data")
    e.fit()
    return e

with st.spinner("⟳ Učitavanje AI sistema..."):
    engine = load_engine()

stats = engine.dashboard_stats()

# Ucitaj simptome (jednom, globalno)
with open("data/specialty_symptoms.json", encoding="utf-8") as f:
    specialty_symptoms = json.load(f)
all_symptoms = sorted(set(s for lst in specialty_symptoms.values() for s in lst))


# ─── Sidebar — uvijek vidljiv, navigacija na vrhu ─────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 20px;">
        <div style="font-size:2.8rem; line-height:1;">⚕️</div>
        <div style="font-size:1.25rem; font-weight:700; color:#e8f0ff; margin-top:8px;">MediConnect AI</div>
        <div style="font-size:0.68rem; color:#5ecfba; letter-spacing:0.12em; margin-top:4px; text-transform:uppercase;">
            Sistem za preporuku doktora
        </div>
    </div>
    <hr style="border-color:rgba(99,202,183,0.2); margin:0 0 16px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigacija",
        ["🔍 Pretraga i preporuka", "📊 Dashboard", "⚡ What-if Simulator", "📋 Baza pacijenata"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.08); margin:16px 0;">
    <div style="font-size:0.75rem; color:#5a6a80; line-height:1.9;">
        <div style="color:#5ecfba; font-weight:600; margin-bottom:6px;">🧠 ML Tehnologije</div>
        • TF-IDF Embeddings<br>
        • KNN (k=5, cosine)<br>
        • Cosine Similarity Search<br>
        • What-if Risk Model<br>
        • Plotly vizualizacije<br>
        <div style="color:#5ecfba; font-weight:600; margin:10px 0 6px;">📦 Dataset</div>
        • 500 simuliranih pacijenata<br>
        • 8 specijalizacija<br>
        • 72 simptoma
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin:16px 0;">
    <div style="font-size:0.7rem; color:#3a4a5a; text-align:center;">
        v1.1 · MediConnect AI
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PRETRAGA I PREPORUKA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Pretraga i preporuka":

    st.markdown('<div class="section-header">🔍 Preporuka doktora i termina</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Unesite simptome — sistem pronalazi najboljeg specijaliste koristeći KNN + Cosine Similarity pretragu nad bazom od 500 pacijenata.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("**📋 Odaberite simptome**")
        selected_symptoms = st.multiselect(
            "Simptomi",
            options=all_symptoms,
            placeholder="Počnite kucati simptom...",
            label_visibility="collapsed"
        )

        st.markdown("**✏️ Ili opišite slobodnim tekstom**")
        free_text = st.text_area(
            "Opis",
            placeholder="npr. Imam jak bol u grudima i otežano disanje već 3 dana...",
            height=100,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**👤 Profil pacijenta**")
        age = st.slider("Starost", 18, 85, 45)
        urgency = st.select_slider(
            "Urgentnost stanja",
            options=["blago", "umereno", "hitno"],
            value="umereno"
        )
        urgency_info = {
            "blago": ("🟢", "#4ade80", "Stanje nije hitno"),
            "umereno": ("🟡", "#facc15", "Preporučuje se pregled"),
            "hitno": ("🔴", "#f87171", "Hitno tražiti pomoć")
        }
        icon, col_u, txt = urgency_info[urgency]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 14px;margin-top:4px;
                    border-left:3px solid {col_u};">
            <span style="font-size:1.1rem;">{icon}</span>
            <span style="color:{col_u};font-weight:600;margin-left:8px;">{urgency.upper()}</span>
            <div style="color:#8899b0;font-size:0.8rem;margin-top:2px;">{txt}</div>
        </div>
        """, unsafe_allow_html=True)

    # Kombinuj simptome
    all_selected = list(selected_symptoms)
    if free_text.strip():
        for symptom in all_symptoms:
            if symptom.lower() in free_text.lower() and symptom not in all_selected:
                all_selected.append(symptom)
        if not all_selected and free_text.strip():
            all_selected = [free_text.strip()]

    # Prikazi odabrane
    if all_selected:
        tags_html = " ".join([
            f'<span style="display:inline-block;margin:3px;padding:5px 13px;border-radius:20px;'
            f'background:rgba(94,207,186,0.15);border:1px solid rgba(94,207,186,0.4);'
            f'color:#5ecfba;font-size:0.82rem;">{s}</span>'
            for s in all_selected
        ])
        st.markdown(f"**Odabrani simptomi ({len(all_selected)}):** {tags_html}", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    search_btn = st.button("🩺 Pronađi najboljeg doktora", type="primary", use_container_width=True)

    # ── Rezultati ─────────────────────────────────────────────────────────────
    if search_btn:
        if not all_selected:
            st.warning("⚠️ Molimo unesite bar jedan simptom.")
        else:
            query_str = "; ".join(all_selected)
            with st.spinner("🔍 Pretražujem bazu pacijenata..."):
                recommendations = engine.recommend_doctors(query_str, urgency=urgency, age=age, top_k=3)
                similar_patients = engine.find_similar_patients(query_str, top_k=5)

            st.markdown("---")
            st.markdown("### 🏥 Top 3 preporuke doktora")

            for i, (_, row) in enumerate(recommendations.iterrows()):
                score = row["final_score"]
                score_color = "#4ade80" if score > 75 else "#facc15" if score > 50 else "#f87171"
                wait = engine.predict_wait_time(row["specialty"], urgency)
                badge = '<span class="badge">✦ BEST MATCH</span>' if i == 0 else ""
                avatar = ["♥", "🧠", "🫁", "🫀", "⚗️", "🦴", "✨", "🧩"][row["doctor_id"] - 1]
                border_col = "rgba(94,207,186,0.5)" if i == 0 else "rgba(255,255,255,0.08)"
                bg = "linear-gradient(135deg,#1e2d3d,#1a2a38)" if i == 0 else "#1a2030"

                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border_col};border-radius:16px;
                            padding:20px 24px;margin-bottom:14px;
                            box-shadow:{'0 0 25px rgba(94,207,186,0.08)' if i==0 else 'none'}">
                    <div style="display:flex;align-items:center;gap:16px;">
                        <div style="font-size:2.4rem;min-width:50px;text-align:center;">{avatar}</div>
                        <div style="flex:1;">
                            <div style="font-size:1.1rem;font-weight:700;color:#e8f0ff;">
                                {row['name']} {badge}
                            </div>
                            <div style="color:#8899b0;font-size:0.88rem;margin-top:2px;">{row['specialty']}</div>
                            <div style="margin-top:8px;font-size:0.8rem;color:#6a7a90;">
                                ⭐ {row['rating']} ocjena &nbsp;·&nbsp;
                                📅 {row['experience_years']} god. iskustva &nbsp;·&nbsp;
                                ✅ {row['success_rate']}% uspjeh
                            </div>
                        </div>
                        <div style="text-align:center;min-width:90px;">
                            <div style="font-size:0.65rem;color:#8899b0;text-transform:uppercase;letter-spacing:0.08em;">Match Score</div>
                            <div style="font-size:2.4rem;font-weight:800;color:{score_color};font-family:monospace;line-height:1.1;">{score}%</div>
                        </div>
                        <div style="text-align:center;min-width:80px;">
                            <div style="font-size:0.65rem;color:#8899b0;text-transform:uppercase;letter-spacing:0.08em;">Čekanje</div>
                            <div style="font-size:2.4rem;font-weight:800;color:#60a5fa;font-family:monospace;line-height:1.1;">{wait}d</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))

            # Slicni pacijenti
            st.markdown("---")
            st.markdown("### 👥 Najsličniji pacijenti iz baze")
            st.caption("KNN pretraga — 5 pacijenata sa najsličnijim simptomima i ishodima liječenja")

            display_df = similar_patients[[
                "patient_id", "age", "gender", "symptoms_text",
                "specialty", "doctor_name", "wait_days", "risk_score", "similarity_score"
            ]].copy()
            display_df.columns = ["ID", "Starost", "Pol", "Simptomi",
                                   "Specijalizacija", "Doktor", "Čekanje (d)", "Rizik %", "Sličnost %"]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Sličnost %": st.column_config.ProgressColumn("Sličnost %", min_value=0, max_value=100),
                    "Rizik %": st.column_config.ProgressColumn("Rizik %", min_value=0, max_value=100),
                }
            )

            with st.expander("📖 Kako je izračunat Match Score?"):
                st.markdown("""
                **Scoring formula (KNN + ML pipeline):**
                ```
                score = 0.45 × similarity_score    (TF-IDF Cosine Similarity vs. baza)
                      + 0.25 × success_rate         (stopa uspješnog liječenja)
                      + 0.15 × wait_penalty         (inverzno proporcionalno čekanju)
                      + 0.10 × experience_norm      (godine iskustva doktora)
                      + 0.05 × urgency_bonus        (korekcija za hitnost)
                ```
                U produkcijskom sistemu, TF-IDF vektori bi bili zamijenjeni sa **Sentence-Transformers**
                embeddings (multilingual-MiniLM) i **FAISS** indexom za brzu pretragu.
                """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":

    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pregled statistika sistema, distribucija i performansi modela.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in zip(
        [k1, k2, k3, k4],
        [stats['total_patients'], f"{stats['avg_success_rate']}%", f"{stats['avg_wait_days']}d", stats['specialties_count']],
        ["Pacijenata u bazi", "Prosj. stopa uspjeha", "Prosj. čekanje", "Specijalizacija"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Distribucija pacijenata po specijalizaciji**")
        spec_data = stats["specialty_dist"]
        fig = go.Figure(go.Bar(
            x=list(spec_data.values()), y=list(spec_data.keys()), orientation='h',
            marker=dict(color=list(spec_data.values()),
                        colorscale=[[0,'#1a3a35'],[0.5,'#1a6b60'],[1,'#5ecfba']]),
            text=list(spec_data.values()), textposition='outside',
            textfont=dict(color='#c9d4e8', size=11)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8899b0'), height=320,
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(color='#c9d4e8', size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Distribucija urgentnosti**")
        urg_data = stats["urgency_dist"]
        fig2 = go.Figure(go.Pie(
            labels=list(urg_data.keys()), values=list(urg_data.values()), hole=0.55,
            marker=dict(colors=["#f87171","#facc15","#4ade80"]),
            textfont=dict(color='#e8f0ff', size=12), textinfo='percent+label'
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8899b0'), height=320,
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            annotations=[dict(text='Urgentnost', x=0.5, y=0.5,
                               font_size=13, font_color='#c9d4e8', showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("**Stopa uspjeha po specijalizaciji (%)**")
        succ = stats["success_by_specialty"]
        fig3 = go.Figure(go.Bar(
            x=list(succ.keys()), y=list(succ.values()),
            marker=dict(color=list(succ.values()), colorscale=[[0,'#134e4a'],[1,'#5ecfba']]),
            text=[f"{v}%" for v in succ.values()], textposition='outside',
            textfont=dict(color='#c9d4e8', size=10)
        ))
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8899b0'), height=300,
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10, color='#c9d4e8'), showgrid=False),
            yaxis=dict(range=[70,100], showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Prosječni rizik po urgentnosti**")
        risk_d = stats["avg_risk_by_urgency"]
        ordered = {k: risk_d[k] for k in ["blago","umereno","hitno"] if k in risk_d}
        fig4 = go.Figure(go.Bar(
            x=list(ordered.keys()), y=list(ordered.values()),
            marker=dict(color=["#4ade80","#facc15","#f87171"]),
            text=[f"{v}%" for v in ordered.values()], textposition='outside',
            textfont=dict(color='#c9d4e8', size=13)
        ))
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8899b0'), height=300,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(tickfont=dict(size=12, color='#c9d4e8'), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("**🏥 Pregled svih doktora u sistemu**")
    doc_display = engine.doctors_df[["name","specialty","success_rate","avg_wait_days","experience_years","rating"]].copy()
    doc_display.columns = ["Ime doktora","Specijalizacija","Uspjeh %","Čekanje (d)","Iskustvo (g)","Ocjena"]
    st.dataframe(
        doc_display, use_container_width=True, hide_index=True,
        column_config={
            "Uspjeh %": st.column_config.ProgressColumn("Uspjeh %", min_value=0, max_value=100),
            "Ocjena": st.column_config.NumberColumn("Ocjena ⭐", format="%.1f"),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ What-if Simulator":

    st.markdown('<div class="section-header">⚡ What-if Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Simulirajte kako odlaganje posjete ljekaru utiče na rizik pogoršanja stanja.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        wi_symptoms = st.multiselect("Simptomi", options=all_symptoms,
                                      default=["bol u grudima","otežano disanje"],
                                      placeholder="Odaberite simptome...")
    with col2:
        wi_age = st.slider("Starost pacijenta", 18, 85, 50)
        wi_urgency = st.select_slider("Urgentnost", options=["blago","umereno","hitno"], value="umereno")

    if wi_symptoms:
        query = "; ".join(wi_symptoms)
        scenarios = engine.whatif_scenarios(query, wi_urgency, wi_age)
        current_risk = scenarios[0]["risk"]
        risk_color = "#4ade80" if current_risk < 30 else "#facc15" if current_risk < 60 else "#f87171"

        col_g, col_s = st.columns([1, 2], gap="large")
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_risk,
                number={"suffix":"%","font":{"color":risk_color,"size":52}},
                delta={"reference":scenarios[1]["risk"],
                       "increasing":{"color":"#f87171"},"decreasing":{"color":"#4ade80"},"suffix":"%"},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#8899b0","tickfont":{"color":"#8899b0"}},
                    "bar":{"color":risk_color,"thickness":0.3},
                    "bgcolor":"#1e2d3d","bordercolor":"rgba(0,0,0,0)",
                    "steps":[{"range":[0,30],"color":"#0f2d1e"},{"range":[30,60],"color":"#2d2a0e"},{"range":[60,100],"color":"#2d0e0e"}],
                    "threshold":{"value":current_risk,"line":{"color":risk_color,"width":4}}
                }
            ))
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=20,r=20,t=40,b=20), font=dict(color="#c9d4e8")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("<div style='text-align:center;color:#8899b0;font-size:0.82rem;margin-top:-10px;'>Trenutni rizik pogoršanja</div>", unsafe_allow_html=True)

        with col_s:
            s_labels = [s["label"] for s in scenarios]
            s_risks = [s["risk"] for s in scenarios]
            s_colors = ["#4ade80" if r < 30 else "#facc15" if r < 60 else "#f87171" for r in s_risks]
            fig_sc = go.Figure(go.Bar(
                x=s_labels, y=s_risks, marker=dict(color=s_colors),
                text=[f"{r}%" for r in s_risks], textposition='outside',
                textfont=dict(color='#c9d4e8', size=14)
            ))
            fig_sc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8899b0'), height=280,
                margin=dict(l=10,r=10,t=30,b=30),
                yaxis=dict(range=[0,105],showgrid=True,gridcolor='rgba(255,255,255,0.06)',
                           title="Rizik (%)",titlefont=dict(color='#8899b0')),
                xaxis=dict(tickfont=dict(size=12,color='#c9d4e8'),showgrid=False),
                title=dict(text="Rizik po scenariju odlaganja",font=dict(color='#c9d4e8',size=13),x=0.5)
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        st.markdown("**🎚️ Prilagodite broj dana odlaganja**")
        custom_days = st.slider("Dana odlaganja", 0, 30, 7)
        custom_risk = engine.whatif_risk(query, wi_urgency, custom_days, wi_age)
        custom_color = "#4ade80" if custom_risk < 30 else "#facc15" if custom_risk < 60 else "#f87171"

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{custom_color}">{custom_risk}%</div>
                <div class="metric-label">Rizik za +{custom_days} dana</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            increase = custom_risk - scenarios[0]["risk"]
            sign = "+" if increase >= 0 else ""
            inc_color = "#f87171" if increase > 0 else "#4ade80"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{inc_color}">{sign}{increase:.1f}%</div>
                <div class="metric-label">Porast rizika</div>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            advice = "HITNO" if custom_risk > 65 else "USKORO" if custom_risk > 35 else "PLANIRANO"
            adv_color = "#f87171" if custom_risk > 65 else "#facc15" if custom_risk > 35 else "#4ade80"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{adv_color};font-size:1.4rem;">{advice}</div>
                <div class="metric-label">Preporuka sistema</div>
            </div>""", unsafe_allow_html=True)

        if custom_risk > 65:
            st.error(f"⚠️ **Hitna preporuka:** Odlaganje za {custom_days} dana povećava rizik na **{custom_risk}%**. Preporučujemo hitan pregled.")
        elif custom_risk > 35:
            st.warning(f"⚡ **Preporuka:** Odlaganje za {custom_days} dana povećava rizik na **{custom_risk}%**. Zakažite pregled u narednih 2–3 dana.")
        else:
            st.success(f"✅ **Situacija je stabilna.** Rizik od **{custom_risk}%**. Ipak, preporučujemo zakazivanje pregleda.")
    else:
        st.info("ℹ️ Odaberite simptome da biste pokrenuli What-if analizu.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: BAZA PACIJENATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Baza pacijenata":

    st.markdown('<div class="section-header">📋 Baza pacijenata</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Simulirani dataset od 500 pacijenata — pretraživanje i filtriranje po svim parametrima.</div>', unsafe_allow_html=True)

    df = engine.patients_df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        spec_filter = st.multiselect("Specijalizacija", options=sorted(df["specialty"].unique()), default=[])
    with col2:
        urg_filter = st.multiselect("Urgentnost", options=["hitno","umereno","blago"], default=[])
    with col3:
        age_range = st.slider("Raspon starosti", 18, 85, (18, 85))

    mask = pd.Series([True] * len(df))
    if spec_filter:
        mask &= df["specialty"].isin(spec_filter)
    if urg_filter:
        mask &= df["urgency"].isin(urg_filter)
    mask &= (df["age"] >= age_range[0]) & (df["age"] <= age_range[1])

    filtered = df[mask]
    st.caption(f"Prikazano {len(filtered)} od {len(df)} pacijenata")

    display = filtered[["patient_id","age","gender","symptoms_text",
                          "specialty","doctor_name","urgency","wait_days","risk_score","treatment_success"]].copy()
    display.columns = ["ID","Starost","Pol","Simptomi","Specijalizacija","Doktor","Urgentnost","Čekanje","Rizik %","Uspjeh"]
    display["Uspjeh"] = display["Uspjeh"].map({1:"✅ Da", 0:"❌ Ne"})

    st.dataframe(
        display, use_container_width=True, hide_index=True, height=450,
        column_config={"Rizik %": st.column_config.ProgressColumn("Rizik %", min_value=0, max_value=100)}
    )
