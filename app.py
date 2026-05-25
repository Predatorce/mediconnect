"""
app.py — MediConnect AI
Sistem za preporuku doktora i termina baziran na mašinskom učenju.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page config (mora biti prvi Streamlit poziv) ────────────────────────────
st.set_page_config(
    page_title="MediConnect AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark theme overrides */
.stApp { background-color: #0f1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16192b 100%);
    border: 1px solid rgba(99,202,183,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #63cab7;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #7a8499;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Doctor recommendation card */
.doc-card {
    background: #1a1d2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.doc-card-top {
    background: #1a1d2e;
    border: 1px solid rgba(99,202,183,0.4);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 0 25px rgba(99,202,183,0.08);
}
.doc-name { font-size: 1.1rem; font-weight: 600; color: #e8eaf0; }
.doc-spec { font-size: 0.85rem; color: #7a8499; margin-top: 2px; }
.score-big { font-size: 2rem; font-weight: 700; font-family: monospace; }
.badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 20px;
    background: rgba(99,202,183,0.15);
    color: #63cab7;
    border: 1px solid rgba(99,202,183,0.3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Risk colors */
.risk-low  { color: #4ade80; }
.risk-med  { color: #facc15; }
.risk-high { color: #f87171; }

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e8eaf0;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.85rem;
    color: #7a8499;
    margin-bottom: 20px;
}

/* Symptom tags */
.stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(99,202,183,0.2) !important;
    color: #63cab7 !important;
    border: 1px solid rgba(99,202,183,0.4) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
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


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2.5rem;">⚕️</div>
        <div style="font-size:1.3rem; font-weight:700; color:#e8eaf0;">MediConnect AI</div>
        <div style="font-size:0.7rem; color:#7a8499; letter-spacing:0.1em; margin-top:4px;">
            SISTEM ZA PREPORUKU DOKTORA
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigacija",
        ["🔍 Pretraga i preporuka", "📊 Dashboard", "⚡ What-if Simulator", "📋 Baza pacijenata"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#4a5568;">
    <b style="color:#63cab7">Tehnologije:</b><br>
    • TF-IDF Embeddings (Sklearn)<br>
    • KNN Klasifikator<br>
    • Cosine Similarity Search<br>
    • What-if Risk Model<br>
    • Plotly vizualizacije<br><br>
    <b style="color:#63cab7">Dataset:</b> 500 simuliranih pacijenata<br>
    8 medicinskih specijalizacija
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PRETRAGA I PREPORUKA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Pretraga i preporuka":

    st.markdown('<div class="section-header">🔍 Preporuka doktora i termina</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Unesite simptome — sistem će pronaći najboljeg specijaliste koristeći KNN + Cosine Similarity.</div>', unsafe_allow_html=True)

    # ── Input forma ──────────────────────────────────────────────────────────
    import json
    with open("data/specialty_symptoms.json", encoding="utf-8") as f:
        specialty_symptoms = json.load(f)

    all_symptoms = sorted(set(s for lst in specialty_symptoms.values() for s in lst))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Odaberite simptome (checkboxovi)**")
        selected_symptoms = st.multiselect(
            "Simptomi",
            options=all_symptoms,
            placeholder="Počnite kucati simptom...",
            label_visibility="collapsed"
        )

        st.markdown("**Ili opišite slobodnim tekstom**")
        free_text = st.text_area(
            "Opis",
            placeholder="npr. Imam jak bol u grudima i otežano disanje već 3 dana...",
            height=90,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**Profil pacijenta**")
        age = st.slider("Starost", 18, 85, 45)
        urgency = st.select_slider(
            "Urgentnost",
            options=["blago", "umereno", "hitno"],
            value="umereno"
        )
        urgency_colors = {"blago": "🟢", "umereno": "🟡", "hitno": "🔴"}
        st.markdown(f"Izabrana urgentnost: {urgency_colors[urgency]} **{urgency.upper()}**")

    # Kombinuj simptome
    all_selected = list(selected_symptoms)
    if free_text.strip():
        # Ekstrakcija simptoma iz slobodnog teksta
        for symptom in all_symptoms:
            if symptom.lower() in free_text.lower() and symptom not in all_selected:
                all_selected.append(symptom)
        if not all_selected and free_text.strip():
            all_selected = [free_text.strip()]

    # Prikaži odabrane
    if all_selected:
        tags_html = " ".join([
            f'<span style="display:inline-block;margin:3px;padding:4px 12px;border-radius:20px;'
            f'background:rgba(99,202,183,0.15);border:1px solid rgba(99,202,183,0.4);'
            f'color:#63cab7;font-size:0.8rem;">{s}</span>'
            for s in all_selected
        ])
        st.markdown(f"**Odabrani simptomi ({len(all_selected)}):** {tags_html}", unsafe_allow_html=True)

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
                
                card_class = "doc-card-top" if i == 0 else "doc-card"
                badge = '<span class="badge">BEST MATCH ★</span>' if i == 0 else ""
                avatar = ["♥", "🧠", "🫁", "🫀", "⚗️", "🦴", "✨", "🧩"][row["doctor_id"] - 1]

                c1, c2, c3, c4 = st.columns([0.5, 3, 2, 2])
                with c1:
                    st.markdown(f"""
                    <div style="font-size:2.2rem;text-align:center;margin-top:10px;">{avatar}</div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="padding:10px 0">
                        <div class="doc-name">{row['name']} {badge}</div>
                        <div class="doc-spec">{row['specialty']}</div>
                        <div style="margin-top:8px;font-size:0.8rem;color:#7a8499;">
                            ⭐ {row['rating']} ocjena &nbsp;|&nbsp; 
                            📅 {row['experience_years']} god. iskustva &nbsp;|&nbsp;
                            ✅ {row['success_rate']}% uspjeh
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div style="text-align:center;padding:10px 0;">
                        <div style="font-size:0.7rem;color:#7a8499;text-transform:uppercase;letter-spacing:0.08em;">Match Score</div>
                        <div style="font-size:2.2rem;font-weight:700;color:{score_color};font-family:monospace">{score}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div style="text-align:center;padding:10px 0;">
                        <div style="font-size:0.7rem;color:#7a8499;text-transform:uppercase;letter-spacing:0.08em;">Čekanje</div>
                        <div style="font-size:2.2rem;font-weight:700;color:#60a5fa;font-family:monospace">{wait}d</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Progress bar
                st.progress(int(score))
                if i < 2:
                    st.divider()

            # ── Slični pacijenti ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 👥 Najsličniji pacijenti iz baze")
            st.caption("KNN pretraga — pronađeni pacijenti sa sličnim simptomima i ishodima")

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

            # ── Score objašnjenje ─────────────────────────────────────────────
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
    st.markdown('<div class="section-sub">Statistike sistema, distribucije i performanse modela.</div>', unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_patients']}</div>
            <div class="metric-label">Pacijenata u bazi</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_success_rate']}%</div>
            <div class="metric-label">Prosj. stopa uspjeha</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_wait_days']}d</div>
            <div class="metric-label">Prosj. čekanje</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['specialties_count']}</div>
            <div class="metric-label">Specijalizacija</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts row 1 ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribucija pacijenata po specijalizaciji**")
        spec_data = stats["specialty_dist"]
        fig = go.Figure(go.Bar(
            x=list(spec_data.values()),
            y=list(spec_data.keys()),
            orientation='h',
            marker=dict(
                color=list(spec_data.values()),
                colorscale=[[0, '#1a1d2e'], [0.5, '#0d6b5e'], [1, '#63cab7']],
                line=dict(width=0)
            ),
            text=list(spec_data.values()),
            textposition='outside',
            textfont=dict(color='#e8eaf0', size=11)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a8499'),
            height=320,
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(color='#e8eaf0', size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Distribucija urgentnosti pacijenata**")
        urg_data = stats["urgency_dist"]
        colors_urg = {"hitno": "#f87171", "umereno": "#facc15", "blago": "#4ade80"}
        fig2 = go.Figure(go.Pie(
            labels=list(urg_data.keys()),
            values=list(urg_data.values()),
            hole=0.55,
            marker=dict(colors=[colors_urg.get(k, "#63cab7") for k in urg_data.keys()]),
            textfont=dict(color='#e8eaf0', size=12),
            textinfo='percent+label'
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a8499'),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            annotations=[dict(text='Urgentnost', x=0.5, y=0.5, font_size=13,
                               font_color='#e8eaf0', showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ───────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Stopa uspjeha po specijalizaciji (%)**")
        succ = stats["success_by_specialty"]
        fig3 = go.Figure(go.Bar(
            x=list(succ.keys()),
            y=list(succ.values()),
            marker=dict(color=list(succ.values()),
                        colorscale=[[0,'#134e4a'],[1,'#63cab7']]),
            text=[f"{v}%" for v in succ.values()],
            textposition='outside',
            textfont=dict(color='#e8eaf0', size=10)
        ))
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a8499'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10, color='#e8eaf0'), showgrid=False),
            yaxis=dict(range=[70, 100], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Prosječni rizik pogoršanja po urgentnosti**")
        risk_d = stats["avg_risk_by_urgency"]
        ordered = {k: risk_d[k] for k in ["blago", "umereno", "hitno"] if k in risk_d}
        colors_r = ["#4ade80", "#facc15", "#f87171"]
        fig4 = go.Figure(go.Bar(
            x=list(ordered.keys()),
            y=list(ordered.values()),
            marker=dict(color=colors_r),
            text=[f"{v}%" for v in ordered.values()],
            textposition='outside',
            textfont=dict(color='#e8eaf0', size=13)
        ))
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a8499'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(tickfont=dict(size=12, color='#e8eaf0'), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Doctors table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🏥 Pregled svih doktora u sistemu**")
    doc_display = engine.doctors_df[[
        "name", "specialty", "success_rate", "avg_wait_days", "experience_years", "rating"
    ]].copy()
    doc_display.columns = ["Ime doktora", "Specijalizacija", "Uspjeh %", "Čekanje (d)", "Iskustvo (g)", "Ocjena"]
    st.dataframe(
        doc_display,
        use_container_width=True,
        hide_index=True,
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
    st.markdown('<div class="section-sub">Analizirajte kako odlaganje posjete ljekaru utiče na rizik pogoršanja stanja.</div>', unsafe_allow_html=True)

    import json
    with open("data/specialty_symptoms.json", encoding="utf-8") as f:
        specialty_symptoms = json.load(f)
    all_symptoms = sorted(set(s for lst in specialty_symptoms.values() for s in lst))

    col1, col2 = st.columns([2, 1])
    with col1:
        wi_symptoms = st.multiselect("Simptomi", options=all_symptoms,
                                      default=["bol u grudima", "otežano disanje"],
                                      placeholder="Odaberite simptome...")
    with col2:
        wi_age = st.slider("Starost pacijenta", 18, 85, 50)
        wi_urgency = st.select_slider("Urgentnost", options=["blago", "umereno", "hitno"], value="umereno")

    if wi_symptoms:
        query = "; ".join(wi_symptoms)
        scenarios = engine.whatif_scenarios(query, wi_urgency, wi_age)

        # ── Gauge chart za trenutni rizik ─────────────────────────────────
        current_risk = scenarios[0]["risk"]
        risk_color = "#4ade80" if current_risk < 30 else "#facc15" if current_risk < 60 else "#f87171"

        col_g, col_s = st.columns([1, 2])
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_risk,
                number={"suffix": "%", "font": {"color": risk_color, "size": 52}},
                delta={"reference": scenarios[1]["risk"], "increasing": {"color": "#f87171"},
                       "decreasing": {"color": "#4ade80"}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7a8499", "tickfont": {"color": "#7a8499"}},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "bgcolor": "#1a1d2e",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": "#0f2d1e"},
                        {"range": [30, 60], "color": "#2d2a0e"},
                        {"range": [60, 100], "color": "#2d0e0e"}
                    ],
                    "threshold": {"value": current_risk, "line": {"color": risk_color, "width": 4}}
                }
            ))
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color="#e8eaf0")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"<div style='text-align:center;color:#7a8499;font-size:0.8rem;'>Trenutni rizik pogoršanja</div>", unsafe_allow_html=True)

        with col_s:
            # ── Scenario bar chart ────────────────────────────────────────
            s_labels = [s["label"] for s in scenarios]
            s_risks = [s["risk"] for s in scenarios]
            s_colors = ["#4ade80" if r < 30 else "#facc15" if r < 60 else "#f87171" for r in s_risks]

            fig_sc = go.Figure(go.Bar(
                x=s_labels, y=s_risks,
                marker=dict(color=s_colors),
                text=[f"{r}%" for r in s_risks],
                textposition='outside',
                textfont=dict(color='#e8eaf0', size=14)
            ))
            fig_sc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#7a8499'),
                height=280,
                margin=dict(l=10, r=10, t=30, b=30),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                           title="Rizik pogoršanja (%)", titlefont=dict(color='#7a8499')),
                xaxis=dict(tickfont=dict(size=12, color='#e8eaf0'), showgrid=False),
                title=dict(text="Rizik po scenariju odlaganja", font=dict(color='#e8eaf0', size=13),
                           x=0.5)
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Interaktivni slider ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("**🎚️ Prilagodite broj dana odlaganja**")
        custom_days = st.slider("Dana odlaganja", 0, 30, 7)
        custom_risk = engine.whatif_risk(query, wi_urgency, custom_days, wi_age)
        custom_color = "#4ade80" if custom_risk < 30 else "#facc15" if custom_risk < 60 else "#f87171"

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{custom_color}">{custom_risk}%</div>
                <div class="metric-label">Rizik za +{custom_days} dana</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            increase = custom_risk - scenarios[0]["risk"]
            sign = "+" if increase >= 0 else ""
            inc_color = "#f87171" if increase > 0 else "#4ade80"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{inc_color}">{sign}{increase:.1f}%</div>
                <div class="metric-label">Porast rizika</div>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            advice = "HITNO" if custom_risk > 65 else "USKORO" if custom_risk > 35 else "PLANIRANO"
            adv_color = "#f87171" if custom_risk > 65 else "#facc15" if custom_risk > 35 else "#4ade80"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{adv_color};font-size:1.4rem;">{advice}</div>
                <div class="metric-label">Preporuka sistema</div>
            </div>""", unsafe_allow_html=True)

        # ── Tekst preporuka ───────────────────────────────────────────────
        if custom_risk > 65:
            st.error(f"⚠️ **Hitna preporuka:** Vaši simptomi ukazuju na visok rizik. Odlaganje za {custom_days} dana povećava rizik pogoršanja na **{custom_risk}%**. Preporučujemo hitan pregled.")
        elif custom_risk > 35:
            st.warning(f"⚡ **Preporuka:** Odlaganje za {custom_days} dana povećava rizik na **{custom_risk}%**. Zakažite pregled u narednih 2–3 dana.")
        else:
            st.success(f"✅ **Situacija je stabilna.** Odlaganje za {custom_days} dana nosi rizik od **{custom_risk}%**. Ipak, preporučujemo zakazivanje pregleda.")
    else:
        st.info("Odaberite simptome da biste pokrenuli What-if analizu.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: BAZA PACIJENATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Baza pacijenata":

    st.markdown('<div class="section-header">📋 Baza pacijenata</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Simulirani dataset od 500 pacijenata — pretraživanje i filtriranje.</div>', unsafe_allow_html=True)

    df = engine.patients_df.copy()

    # Filteri
    col1, col2, col3 = st.columns(3)
    with col1:
        spec_filter = st.multiselect("Filtriranje po specijalizaciji",
                                      options=sorted(df["specialty"].unique()),
                                      default=[])
    with col2:
        urg_filter = st.multiselect("Filtriranje po urgentnosti",
                                     options=["hitno", "umereno", "blago"],
                                     default=[])
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

    display = filtered[[
        "patient_id", "age", "gender", "symptoms_text",
        "specialty", "doctor_name", "urgency", "wait_days", "risk_score", "treatment_success"
    ]].copy()
    display.columns = ["ID", "Starost", "Pol", "Simptomi", "Specijalizacija",
                        "Doktor", "Urgentnost", "Čekanje", "Rizik %", "Uspjeh"]
    display["Uspjeh"] = display["Uspjeh"].map({1: "✅ Da", 0: "❌ Ne"})

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=450,
        column_config={
            "Rizik %": st.column_config.ProgressColumn("Rizik %", min_value=0, max_value=100),
        }
    )
