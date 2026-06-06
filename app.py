"""
MediConnect - Sistem za preporuku doktora i termina
Predmet: Data Mining | Student: Andrej Zajovic 22/096 | FIST
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="MediConnect",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* Sakri sidebar i njegovo dugme potpuno */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

.stApp { background-color: #F0F4F8 !important; }
.main .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1200px !important; }

/* Navigacija na vrhu */
.topnav {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 0;
    margin: 0 -2rem 2rem -2rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    position: sticky;
    top: 0;
    z-index: 999;
}

/* Kartice */
.wcard {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.kpi {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.kpi-val { font-size: 1.9rem; font-weight: 800; color: #0D9488; line-height: 1.1; }
.kpi-lbl { font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-top: 3px; }
.ptitle { font-size: 1.6rem; font-weight: 800; color: #0F172A; margin-bottom: 4px; }
.psub   { font-size: 0.88rem; color: #64748B; margin-bottom: 22px; }

/* Inputi */
.stTextArea textarea {
    background: #FFFFFF !important; border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important; color: #0F172A !important; font-size: 0.9rem !important;
}
.stMultiSelect > div > div {
    background: #FFFFFF !important; border: 1.5px solid #CBD5E1 !important; border-radius: 12px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #CCFBF1 !important; color: #0F766E !important;
    border: 1px solid #99F6E4 !important; border-radius: 6px !important; font-weight: 600 !important;
}

/* Dugme */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0D9488 0%, #0284C7 100%) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 0.95rem !important; padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 16px rgba(13,148,136,0.3) !important; transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(13,148,136,0.4) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg,#0D9488,#0284C7) !important; border-radius: 4px !important;
}
hr { border-color: #E2E8F0 !important; }
.stDataFrame { border-radius: 12px !important; border: 1px solid #E2E8F0 !important; overflow: hidden !important; }
.stAlert { border-radius: 12px !important; border-left-width: 4px !important; }
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ── Ucitavanje modela ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    from ml_engine import MediConnectEngine
    e = MediConnectEngine(data_dir="data")
    e.fit()
    return e

with st.spinner("Ucitavanje sistema..."):
    engine = load_engine()

stats = engine.dashboard_stats()

with open("data/specialty_symptoms.json", encoding="utf-8") as f:
    specialty_symptoms = json.load(f)
all_symptoms = sorted(set(s for lst in specialty_symptoms.values() for s in lst))


# ── Navigacija na vrhu ────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
    <div style="display:flex;align-items:center;gap:10px;padding:14px 24px 14px 0;
                border-right:1px solid #F1F5F9;margin-right:16px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#0D9488,#0284C7);
                    border-radius:9px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;">⚕️</div>
        <div>
            <div style="font-size:0.95rem;font-weight:800;color:#0F172A;line-height:1.1;">MediConnect</div>
            <div style="font-size:0.6rem;color:#0D9488;font-weight:700;letter-spacing:0.05em;">DATA MINING · FIST</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigacioni tabovi — uvijek vidljivi, ne mogu se zatvoriti
pages = ["🔍 Pretraga", "📊 Dashboard", "⚡ What-if", "📋 Baza pacijenata"]
selected = st.tabs(pages)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRETRAGA
# ═══════════════════════════════════════════════════════════════════════════════
with selected[0]:
    st.markdown('<div class="ptitle">🔍 Preporuka doktora i termina</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Unesite simptome — KNN algoritam pronalazi najboljeg specijaliste iz baze od 500 pacijenata.</div>', unsafe_allow_html=True)

    st.markdown('<div class="wcard">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("##### 🩺 Odaberite simptome")
        selected_symptoms = st.multiselect(
            "simptomi", options=all_symptoms,
            placeholder="Pocnite kucati simptom...",
            label_visibility="collapsed"
        )
        st.markdown("##### ✍️ Ili opisite slobodnim tekstom")
        free_text = st.text_area(
            "tekst", height=95, label_visibility="collapsed",
            placeholder="npr. Imam jak bol u grudima i otezano disanje vec 3 dana..."
        )
    with col2:
        st.markdown("##### 👤 Profil pacijenta")
        age = st.slider("Starost pacijenta", 18, 85, 45)
        urgency = st.select_slider(
            "Hitnost stanja",
            options=["blago", "umereno", "hitno"],
            value="umereno"
        )
        urg_cfg = {
            "blago":   ("#D1FAE5", "#065F46", "#059669", "Planirani pregled"),
            "umereno": ("#FEF9C3", "#713F12", "#CA8A04", "Preporucuje se pregled"),
            "hitno":   ("#FEE2E2", "#7F1D1D", "#DC2626", "Hitno trazite pomoc"),
        }
        bg, tc, bc, txt = urg_cfg[urgency]
        st.markdown(f"""
        <div style="background:{bg};border-left:4px solid {bc};border-radius:10px;
                    padding:12px 16px;margin-top:6px;">
            <div style="font-weight:700;color:{tc};font-size:0.92rem;">{urgency.upper()}</div>
            <div style="color:{tc};opacity:0.75;font-size:0.8rem;margin-top:2px;">{txt}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    all_selected = list(selected_symptoms)
    if free_text.strip():
        for s in all_symptoms:
            if s.lower() in free_text.lower() and s not in all_selected:
                all_selected.append(s)
        if not all_selected:
            all_selected = [free_text.strip()]

    if all_selected:
        tags = " ".join([
            f'<span style="display:inline-block;margin:3px 4px;padding:5px 14px;border-radius:20px;'
            f'background:#CCFBF1;border:1px solid #99F6E4;color:#0F766E;font-size:0.82rem;font-weight:600;">{s}</span>'
            for s in all_selected
        ])
        st.markdown(f"**Odabrani simptomi ({len(all_selected)}):** {tags}", unsafe_allow_html=True)
        st.markdown("")

    search_btn = st.button("🩺  Pronadi najboljeg doktora", type="primary", use_container_width=True)

    if search_btn:
        if not all_selected:
            st.warning("Molimo unesite bar jedan simptom.")
        else:
            query_str = "; ".join(all_selected)
            with st.spinner("Pretrazujem bazu pacijenata..."):
                recs    = engine.recommend_doctors(query_str, urgency=urgency, age=age, top_k=3)
                similar = engine.find_similar_patients(query_str, top_k=5)

            st.markdown("---")
            st.markdown("### 🏥 Top 3 preporuke doktora")

            avatars  = ["♥", "🧠", "🫁", "🫀", "⚗️", "🦴", "✨", "🧩"]
            sc_color = lambda s: "#059669" if s > 75 else "#CA8A04" if s > 50 else "#DC2626"

            for i, (_, row) in enumerate(recs.iterrows()):
                score  = row["final_score"]
                wait   = engine.predict_wait_time(row["specialty"], urgency)
                av     = avatars[row["doctor_id"] - 1]
                cc     = sc_color(score)
                border = "2px solid #0D9488" if i == 0 else "1px solid #E2E8F0"
                shadow = "0 4px 20px rgba(13,148,136,0.1)" if i == 0 else "0 1px 6px rgba(0,0,0,0.04)"
                badge  = '<span style="background:linear-gradient(135deg,#0D9488,#0284C7);color:white;font-size:0.62rem;padding:3px 10px;border-radius:20px;font-weight:700;margin-left:8px;">BEST MATCH</span>' if i == 0 else ""
                rank   = ["🥇", "🥈", "🥉"][i]
                av_bg  = "linear-gradient(135deg,#CCFBF1,#CFFAFE)" if i == 0 else "#F8FAFC"
                av_br  = "#99F6E4" if i == 0 else "#E2E8F0"

                st.markdown(f"""
                <div style="background:#FFFFFF;border:{border};border-radius:16px;
                            padding:20px 24px;margin-bottom:14px;box-shadow:{shadow};">
                    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                        <div style="font-size:1.6rem;min-width:28px;">{rank}</div>
                        <div style="width:50px;height:50px;background:{av_bg};border-radius:13px;
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:1.5rem;flex-shrink:0;border:1px solid {av_br};">{av}</div>
                        <div style="flex:1;min-width:150px;">
                            <div style="font-size:1.02rem;font-weight:700;color:#0F172A;">{row['name']} {badge}</div>
                            <div style="color:#64748B;font-size:0.85rem;margin-top:2px;">{row['specialty']}</div>
                            <div style="margin-top:7px;font-size:0.77rem;color:#94A3B8;">
                                ⭐ {row['rating']} &nbsp;·&nbsp;
                                {row['experience_years']} god. iskustva &nbsp;·&nbsp;
                                {row['success_rate']}% uspjeh
                            </div>
                        </div>
                        <div style="text-align:center;min-width:85px;padding:8px 14px;
                                    background:#F8FAFC;border-radius:12px;border:1px solid #E2E8F0;">
                            <div style="font-size:0.62rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700;">Match</div>
                            <div style="font-size:2rem;font-weight:800;color:{cc};font-family:monospace;line-height:1.1;">{score}%</div>
                        </div>
                        <div style="text-align:center;min-width:85px;padding:8px 14px;
                                    background:#EFF6FF;border-radius:12px;border:1px solid #BFDBFE;">
                            <div style="font-size:0.62rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700;">Cekanje</div>
                            <div style="font-size:2rem;font-weight:800;color:#2563EB;font-family:monospace;line-height:1.1;">{wait}d</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))

            st.markdown("---")
            st.markdown("### 👥 Najslicniji pacijenti iz baze")
            st.caption("KNN pretraga — 5 pacijenata sa najslicnijim simptomima i ishodima lijecenja")

            df_sim = similar[[
                "patient_id", "age", "gender", "symptoms_text",
                "specialty", "doctor_name", "wait_days", "risk_score", "similarity_score"
            ]].copy()
            df_sim.columns = ["ID", "Starost", "Pol", "Simptomi", "Specijalizacija", "Doktor", "Cekanje", "Rizik %", "Slicnost %"]
            st.dataframe(
                df_sim, use_container_width=True, hide_index=True,
                column_config={
                    "Slicnost %": st.column_config.ProgressColumn("Slicnost %", min_value=0, max_value=100),
                    "Rizik %":    st.column_config.ProgressColumn("Rizik %",    min_value=0, max_value=100),
                }
            )

            with st.expander("Kako je izracunat Match Score?"):
                st.markdown("""
                | Komponenta | Tezina | Opis |
                |---|---|---|
                | Cosine Similarity | 45% | Slicnost simptoma sa bazom pacijenata |
                | Stopa uspjeha | 25% | Uspjesnost lijecenja doktora |
                | Penalizacija cekanja | 15% | Krace cekanje = visi skor |
                | Iskustvo doktora | 10% | Normalizovane godine iskustva |
                | Korekcija hitnosti | 5% | Prilagodba za hitne slucajeve |
                """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with selected[1]:
    st.markdown('<div class="ptitle">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Statistike sistema, distribucije pacijenata i performanse modela.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    for col, val, lbl, ico in zip(
        [k1, k2, k3, k4],
        [stats['total_patients'], f"{stats['avg_success_rate']}%", f"{stats['avg_wait_days']}d", stats['specialties_count']],
        ["Pacijenata u bazi", "Prosj. stopa uspjeha", "Prosj. cekanje", "Specijalizacija"],
        ["👥", "✅", "⏱", "🏥"]
    ):
        with col:
            st.markdown(f"""<div class="kpi">
                <div style="font-size:1.4rem;margin-bottom:4px;">{ico}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="wcard">', unsafe_allow_html=True)
        st.markdown("**Distribucija po specijalizaciji**")
        spec_data = stats["specialty_dist"]
        fig = go.Figure(go.Bar(
            x=list(spec_data.values()), y=list(spec_data.keys()), orientation='h',
            marker=dict(color="#0D9488", opacity=0.85),
            text=list(spec_data.values()), textposition='outside',
            textfont=dict(color='#334155', size=11)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=0, r=40, t=10, b=10),
            font=dict(color='#64748B'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(color='#334155', size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="wcard">', unsafe_allow_html=True)
        st.markdown("**Distribucija hitnosti pacijenata**")
        urg_data = stats["urgency_dist"]
        fig2 = go.Figure(go.Pie(
            labels=list(urg_data.keys()), values=list(urg_data.values()), hole=0.58,
            marker=dict(colors=["#EF4444", "#F59E0B", "#10B981"]),
            textfont=dict(color='#0F172A', size=12), textinfo='percent+label'
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            font=dict(color='#64748B'),
            annotations=[dict(text='Hitnost', x=0.5, y=0.5, font_size=12, font_color='#334155', showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown('<div class="wcard">', unsafe_allow_html=True)
        st.markdown("**Stopa uspjeha po specijalizaciji (%)**")
        succ   = stats["success_by_specialty"]
        vals   = list(succ.values())
        colors = ["#059669" if v >= 92 else "#0284C7" if v >= 88 else "#D97706" for v in vals]
        fig3 = go.Figure(go.Bar(
            x=list(succ.keys()), y=vals,
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v}%" for v in vals], textposition='outside',
            textfont=dict(color='#334155', size=10)
        ))
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=290, margin=dict(l=10, r=10, t=10, b=80),
            font=dict(color='#64748B'),
            xaxis=dict(tickangle=-35, tickfont=dict(color='#334155', size=10), showgrid=False),
            yaxis=dict(range=[70, 100], showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="wcard">', unsafe_allow_html=True)
        st.markdown("**Prosjecni rizik po hitnosti**")
        risk_d  = stats["avg_risk_by_urgency"]
        ordered = {k: risk_d[k] for k in ["blago", "umereno", "hitno"] if k in risk_d}
        fig4 = go.Figure(go.Bar(
            x=list(ordered.keys()), y=list(ordered.values()),
            marker=dict(color=["#10B981", "#F59E0B", "#EF4444"], opacity=0.85),
            text=[f"{v}%" for v in ordered.values()], textposition='outside',
            textfont=dict(color='#334155', size=13)
        ))
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=290, margin=dict(l=10, r=10, t=10, b=40),
            font=dict(color='#64748B'),
            xaxis=dict(tickfont=dict(color='#334155', size=13), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏥 Svi doktori u sistemu**")
    doc_df = engine.doctors_df[["name", "specialty", "success_rate", "avg_wait_days", "experience_years", "rating"]].copy()
    doc_df.columns = ["Ime doktora", "Specijalizacija", "Uspjeh %", "Cekanje (d)", "Iskustvo (g)", "Ocjena"]
    st.dataframe(
        doc_df, use_container_width=True, hide_index=True,
        column_config={
            "Uspjeh %": st.column_config.ProgressColumn("Uspjeh %", min_value=0, max_value=100),
            "Ocjena":   st.column_config.NumberColumn("Ocjena", format="%.1f"),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF
# ═══════════════════════════════════════════════════════════════════════════════
with selected[2]:
    st.markdown('<div class="ptitle">⚡ What-if Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Analizirajte kako odlaganje posjete ljekaru povecava rizik pogorsanja zdravstvenog stanja.</div>', unsafe_allow_html=True)

    st.markdown('<div class="wcard">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        wi_symp = st.multiselect(
            "Simptomi za analizu", options=all_symptoms,
            default=["bol u grudima", "otezano disanje"]
        )
    with c2:
        wi_age = st.slider("Starost", 18, 85, 50)
        wi_urg = st.select_slider("Hitnost", options=["blago", "umereno", "hitno"], value="umereno")
    st.markdown('</div>', unsafe_allow_html=True)

    if wi_symp:
        query     = "; ".join(wi_symp)
        scenarios = engine.whatif_scenarios(query, wi_urg, wi_age)
        cur_risk  = scenarios[0]["risk"]
        rc  = "#059669" if cur_risk < 30 else "#D97706" if cur_risk < 60 else "#DC2626"
        rbg = "#D1FAE5" if cur_risk < 30 else "#FEF9C3" if cur_risk < 60 else "#FEE2E2"

        colg, cols = st.columns([1, 2], gap="large")
        with colg:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cur_risk,
                number={"suffix": "%", "font": {"color": rc, "size": 54, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#CBD5E1", "tickfont": {"color": "#94A3B8", "size": 10}},
                    "bar":  {"color": rc, "thickness": 0.28},
                    "bgcolor": "#F8FAFC", "bordercolor": "#E2E8F0", "borderwidth": 1,
                    "steps": [
                        {"range": [0,  30], "color": "#D1FAE5"},
                        {"range": [30, 60], "color": "#FEF9C3"},
                        {"range": [60,100], "color": "#FEE2E2"},
                    ],
                }
            ))
            fig_g.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=255, margin=dict(l=20, r=20, t=30, b=10),
                font=dict(color="#64748B")
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown(f"""<div style="text-align:center;background:{rbg};border-radius:10px;
                padding:10px;font-size:0.85rem;font-weight:700;color:{rc};">
                Trenutni rizik pogorsanja</div>""", unsafe_allow_html=True)

        with cols:
            s_labels = [s["label"] for s in scenarios]
            s_risks  = [s["risk"]  for s in scenarios]
            s_cols   = ["#059669" if r < 30 else "#F59E0B" if r < 60 else "#EF4444" for r in s_risks]
            fig_sc = go.Figure(go.Bar(
                x=s_labels, y=s_risks,
                marker=dict(color=s_cols, opacity=0.85),
                text=[f"{r}%" for r in s_risks],
                textposition='outside',
                textfont=dict(color='#334155', size=14)
            ))
            fig_sc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=255,
                margin=dict(l=10, r=10, t=50, b=30),
                font=dict(color='#64748B'),
                title=dict(text="Rizik pogorsanja po scenariju", font=dict(color='#334155', size=13), x=0.5),
                yaxis=dict(range=[0, 110], showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                           title_text="Rizik (%)", title_font=dict(color='#94A3B8')),
                xaxis=dict(tickfont=dict(size=13, color='#334155'), showgrid=False),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        st.markdown("**Podesite broj dana odlaganja**")
        cdays = st.slider("Broj dana odlaganja", 0, 30, 7)
        crisk = engine.whatif_risk(query, wi_urg, cdays, wi_age)
        cc    = "#059669" if crisk < 30 else "#D97706" if crisk < 60 else "#DC2626"
        cbg   = "#D1FAE5" if crisk < 30 else "#FEF9C3" if crisk < 60 else "#FEE2E2"
        inc   = crisk - scenarios[0]["risk"]
        adv   = "HITNO"   if crisk > 65 else "USKORO"  if crisk > 35 else "PLANIRANO"
        ac    = "#DC2626" if crisk > 65 else "#D97706" if crisk > 35 else "#059669"
        abg   = "#FEE2E2" if crisk > 65 else "#FEF9C3" if crisk > 35 else "#D1FAE5"

        m1, m2, m3 = st.columns(3, gap="medium")
        for col, val, lbl, vc, vbg in [
            (m1, f"{crisk}%",    f"Rizik za +{cdays} dana", cc, cbg),
            (m2, f"+{inc:.1f}%", "Porast rizika",
             "#DC2626" if inc > 0 else "#059669",
             "#FEE2E2" if inc > 0 else "#D1FAE5"),
            (m3, adv, "Preporuka", ac, abg),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:{vbg};border-radius:14px;padding:20px 16px;text-align:center;
                            border:1px solid {vc}25;">
                    <div style="font-size:1.85rem;font-weight:800;color:{vc};font-family:monospace;line-height:1.1;">{val}</div>
                    <div style="font-size:0.7rem;color:{vc};text-transform:uppercase;letter-spacing:0.07em;
                                font-weight:700;margin-top:5px;opacity:0.8;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        if crisk > 65:
            st.error(f"Hitna preporuka: Odlaganje za {cdays} dana povecava rizik na {crisk}%. Odmah potrazite ljekarsku pomoc.")
        elif crisk > 35:
            st.warning(f"Preporuka: Rizik raste na {crisk}%. Zakazite pregled u narednih 2-3 dana.")
        else:
            st.success(f"Situacija je stabilna. Procijenjeni rizik: {crisk}%. Preporucujemo redovni pregled.")
    else:
        st.info("Odaberite simptome da biste pokrenuli analizu.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAZA
# ═══════════════════════════════════════════════════════════════════════════════
with selected[3]:
    st.markdown('<div class="ptitle">📋 Baza pacijenata</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Simulirani dataset od 500 pacijenata — pretrazivanje i filtriranje po svim parametrima.</div>', unsafe_allow_html=True)

    df = engine.patients_df.copy()

    st.markdown('<div class="wcard">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        spec_f = st.multiselect("Specijalizacija", options=sorted(df["specialty"].unique()), default=[])
    with c2:
        urg_f  = st.multiselect("Hitnost", options=["hitno", "umereno", "blago"], default=[])
    with c3:
        age_r  = st.slider("Raspon starosti", 18, 85, (18, 85))
    st.markdown('</div>', unsafe_allow_html=True)

    mask = pd.Series([True] * len(df))
    if spec_f: mask &= df["specialty"].isin(spec_f)
    if urg_f:  mask &= df["urgency"].isin(urg_f)
    mask &= (df["age"] >= age_r[0]) & (df["age"] <= age_r[1])
    filtered = df[mask]

    st.caption(f"Prikazano {len(filtered)} od {len(df)} pacijenata")

    disp = filtered[[
        "patient_id", "age", "gender", "symptoms_text",
        "specialty", "doctor_name", "urgency", "wait_days", "risk_score", "treatment_success"
    ]].copy()
    disp.columns = ["ID", "Starost", "Pol", "Simptomi", "Specijalizacija", "Doktor", "Hitnost", "Cekanje", "Rizik %", "Uspjeh"]
    disp["Uspjeh"] = disp["Uspjeh"].map({1: "Da", 0: "Ne"})

    st.dataframe(
        disp, use_container_width=True, hide_index=True, height=480,
        column_config={"Rizik %": st.column_config.ProgressColumn("Rizik %", min_value=0, max_value=100)}
    )
