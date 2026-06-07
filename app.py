import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import io
import sqlite3
from datetime import datetime

# ── FIX: RE-LOAD RESULTS DIRECTLY FROM DATABASE IF NOT IN MEMORY ──
try:
    if 'RESULTS' not in locals() or RESULTS is None or RESULTS.empty:
        conn = sqlite3.connect('water_stress.db', check_same_thread=False)
        RESULTS = pd.read_sql_query("""
            SELECT country_name AS country, rainfall_mm AS rainfall,
                   population_millions AS population, gdp_per_capita AS gdp,
                   avg_temperature AS temp, predicted_stress AS stress,
                   (conflict_risk * 100) AS conflict_pct
            FROM predictions GROUP BY country_name
        """, conn)
        geo_df = pd.read_sql_query("SELECT name AS country, region, latitude AS lat, longitude AS lon FROM countries", conn)
        RESULTS = pd.merge(RESULTS, geo_df, on='country', how='left')
        conn.close()
except Exception as e:
    print(f"⚠️ Could not auto-restore database records: {e}")

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = PILImage.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img

def col(s):
    return ('#EF4444' if s > 3.5 else '#F59E0B' if s > 2.5 else '#FCD34D' if s > 1.5 else '#22C55E')

# ── Charts ────────────────────────────────────────────────
def chart_region():
    reg = RESULTS.groupby('region')['stress'].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#0f1729'); ax.set_facecolor('#0f1729')
    colors = [col(v) for v in reg.values]
    bars = ax.barh(reg.index, reg.values, color=colors, edgecolor='none', height=0.6)
    ax.axvline(x=2.5, color='#EF4444', linestyle='--', alpha=0.6, linewidth=1)
    for bar, val in zip(bars, reg.values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f'{val:.2f}', va='center', color='white', fontsize=8)
    ax.set_xlabel('Average Stress Score', color='#9ca3af', fontsize=8)
    ax.set_title('Water Stress by Region', color='white', fontweight='bold', fontsize=10, pad=8)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, 5)
    return fig_to_pil(fig)

def chart_scatter():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#0f1729'); ax.set_facecolor('#0f1729')
    colors = [col(s) for s in RESULTS['stress']]
    ax.scatter(RESULTS['stress'], RESULTS['conflict_pct'] / 100, c=colors, s=40, alpha=0.8, edgecolors='none')
    ax.set_xlabel('Water Stress Score', color='#9ca3af', fontsize=8)
    ax.set_ylabel('Conflict Risk', color='#9ca3af', fontsize=8)
    ax.set_title('Water Stress vs Conflict Risk', color='white', fontweight='bold', fontsize=10, pad=8)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_color('#1e3a5f')
    return fig_to_pil(fig)

def chart_pie():
    c = (RESULTS['stress'] > 3.5).sum()
    h = ((RESULTS['stress'] > 2.5) & (RESULTS['stress'] <= 3.5)).sum()
    m = ((RESULTS['stress'] > 1.5) & (RESULTS['stress'] <= 2.5)).sum()
    lo = (RESULTS['stress'] <= 1.5).sum()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor('#0f1729'); ax.set_facecolor('#0f1729')
    wedges, texts, autotexts = ax.pie(
        [c, h, m, lo],
        labels=[f'Very High\n{c}', f'High\n{h}', f'Medium\n{m}', f'Low\n{lo}'],
        colors=['#EF4444', '#F59E0B', '#FCD34D', '#22C55E'],
        autopct='%1.0f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 7},
        wedgeprops={'edgecolor': '#0f1729', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')
    ax.set_title('Risk Level Distribution', color='white', fontweight='bold', fontsize=10, pad=8)
    return fig_to_pil(fig)

def chart_top10():
    top = RESULTS.nlargest(10, 'stress').sort_values('stress', ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#0f1729'); ax.set_facecolor('#0f1729')
    bars = ax.bar(top['country'], top['stress'], color=[col(s) for s in top['stress']], edgecolor='none')
    for bar, val in zip(bars, top['stress']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f'{val:.2f}', ha='center', color='white', fontsize=7.5)
    ax.set_title('Top 10 Countries by Water Stress', color='white', fontweight='bold', fontsize=10, pad=8)
    ax.set_ylabel('Stress Score', color='#9ca3af', fontsize=8)
    ax.tick_params(colors='white', labelsize=7.5, axis='x', rotation=30)
    ax.tick_params(colors='white', labelsize=8, axis='y')
    ax.spines[:].set_color('#1e3a5f')
    ax.set_ylim(0, 5.5)
    return fig_to_pil(fig)

def chart_rainfall():
    reg = RESULTS.groupby('region')['rainfall'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#0f1729'); ax.set_facecolor('#0f1729')
    bars = ax.bar(reg.index, reg.values, color='#38BDF8', edgecolor='none', alpha=0.85)
    for bar, val in zip(bars, reg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 10, f'{val:.0f}', ha='center', color='white', fontsize=8)
    ax.set_title('Avg Annual Rainfall by Region', color='white', fontweight='bold', fontsize=10, pad=8)
    ax.set_ylabel('Rainfall (mm)', color='#9ca3af', fontsize=8)
    ax.tick_params(colors='white', labelsize=8, axis='x', rotation=20)
    ax.tick_params(colors='white', labelsize=8, axis='y')
    ax.spines[:].set_color('#1e3a5f')
    return fig_to_pil(fig)

# ── Map ───────────────────────────────────────────────────
def build_map(highlight=None):
    m = folium.Map(location=[20, 20], zoom_start=2, tiles='CartoDB dark_matter')
    for _, row in RESULTS.iterrows():
        is_hl = bool(highlight and str(row['country']).lower() == highlight.lower())
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=16 if is_hl else 8,
            color='white' if is_hl else col(row['stress']),
            weight=3 if is_hl else 1,
            fill=True, fill_color=col(row['stress']), fill_opacity=0.9,
            popup=folium.Popup(
                f"<div style='font-family:Arial;min-width:160px;color:#111'>"
                f"<b style='font-size:14px'>{row['country']}</b><br>"
                f"<hr style='margin:4px 0'>"
                f"<b>Region:</b> {row['region']}<br>"
                f"<b>Water Stress:</b> {row['stress']}/5.0<br>"
                f"<b>Conflict Risk:</b> {row['conflict_pct']:.1f}%<br>"
                f"<b>Rainfall:</b> {row['rainfall']}mm<br>"
                f"<b>Population:</b> {row['population']}M</div>",
                max_width=220),
            tooltip=f"{row['country']} | Stress: {row['stress']}"
        ).add_to(m)
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
    background:rgba(15,23,41,0.95);padding:12px 16px;border-radius:10px;
    border:1px solid #334155;font-family:Arial;font-size:11px;color:#ffffff;">
    <b style="color:#ffffff">💧 Water Stress Level</b><br><br>
    <span style="color:#EF4444;font-size:15px">●</span>
    <span style="color:#ffffff"> Very High (&gt;3.5)</span><br>
    <span style="color:#F59E0B;font-size:15px">●</span>
    <span style="color:#ffffff"> High (2.5–3.5)</span><br>
    <span style="color:#FCD34D;font-size:15px">●</span>
    <span style="color:#ffffff"> Medium (1.5–2.5)</span><br>
    <span style="color:#22C55E;font-size:15px">●</span>
    <span style="color:#ffffff"> Low (&lt;1.5)</span><br><br>
    <i style="color:#9ca3af;font-size:10px">Click circles for details</i>
    </div>"""))
    return f'<div style="height:420px;border-radius:12px;overflow:hidden">{m._repr_html_()}</div>'

# ── Stats bar ─────────────────────────────────────────────
def build_stats(cs=None, cc=None, cn=None):
    avg_s = RESULTS['stress'].mean()
    high  = (RESULTS['stress'] > 2.5).sum()
    avg_r = RESULTS['rainfall'].mean()
    n_reg = RESULTS['region'].nunique()

    sv  = f"{cs:.2f}/5.0" if cs is not None else f"{avg_s:.2f}/5.0"
    sc  = col(cs if cs is not None else avg_s)
    sl  = ('🔴 Critical' if (cs or avg_s) > 3.5 else '🟠 High Risk' if (cs or avg_s) > 2.5 else '🟢 Low Risk')
    cfv = f"{cc:.0%}" if cc is not None else f"{RESULTS['conflict_pct'].mean():.1f}%"
    cfc = '#EF4444' if (cc or 0) > 0.7 else '#F59E0B'
    cl  = cn if cn else f"{len(RESULTS)} Countries"

    # ── LIVE TIME: generated fresh every call ──
    now_date = datetime.now().strftime("%d %b %Y")
    now_time = datetime.now().strftime("%H:%M:%S")

    def card(icon, title, val, sub, c):
        return (
            f"<div style='background:#1e2d4a;border-radius:12px;"
            f"padding:14px 18px;border-top:3px solid {c};"
            f"flex:1;min-width:130px'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"<span style='font-size:18px'>{icon}</span>"
            f"<span style='color:#cbd5e1;font-size:10px;"
            f"font-weight:600;font-family:Arial'>{title}</span></div>"
            f"<div style='color:{c};font-size:20px;font-weight:700;"
            f"font-family:Arial'>{val}</div>"
            f"<div style='color:#94a3b8;font-size:10px;margin-top:3px;"
            f"font-family:Arial'>{sub}</div></div>"
        )

    # ── Clock card with real Python time ──
    clock_card = (
        f"<div style='background:#1e2d4a;border-radius:12px;"
        f"padding:14px 18px;border-top:3px solid #22C55E;"
        f"flex:1;min-width:130px'>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
        f"<span style='font-size:18px'>📅</span>"
        f"<span style='color:#cbd5e1;font-size:10px;font-weight:600;"
        f"font-family:Arial'>LAST UPDATED</span></div>"
        f"<div style='color:#22C55E;font-size:13px;font-weight:700;"
        f"font-family:Arial'>{now_date}</div>"
        f"<div style='color:#22C55E;font-size:20px;font-weight:700;"
        f"font-family:Arial;letter-spacing:1px'>{now_time}</div>"
        f"<div style='color:#94a3b8;font-size:10px;margin-top:3px;"
        f"font-family:Arial'>Real-time</div></div>"
    )

    cards = (
        card("💧", "AVG WATER STRESS", sv, sl, sc) +
        card("🌍", "COUNTRIES", cl, f"Across {n_reg} Regions", "#38BDF8") +
        card("⚠️", "HIGH RISK", str(high), "Stress > 2.5", "#EF4444") +
        card("⚔️", "CONFLICT RISK", cfv, "Global Avg", cfc) +
        card("🌧️", "AVG RAINFALL", f"{avg_r:.0f}mm", "Annual", "#38BDF8") +
        clock_card
    )

    return (
        f"<div style='display:flex;gap:12px;flex-wrap:wrap;"
        f"padding:8px 4px;font-family:Arial'>{cards}</div>"
    )

# ── Predict ───────────────────────────────────────────────
def predict(country, rainfall, population, gdp, temperature):
    try:
        inp = pd.DataFrame([{
            'rainfall_mm': float(rainfall),
            'population_millions': float(population),
            'gdp_per_capita': float(gdp),
            'avg_temperature': float(temperature)
        }])
        stress   = float(np.clip(model.predict(inp)[0], 0, 5))
        conflict = float(np.clip(stress / 5 * 0.7 + 0.1, 0, 1))
        c  = col(stress)
        lv = ('🔴 CRITICAL' if stress > 3.5 else '🟠 HIGH' if stress > 2.5 else '🟡 MEDIUM' if stress > 1.5 else '🟢 LOW')
        msg = ('🚨 URGENT — Immediate water intervention needed!' if stress > 3.5
               else '⚠️ High risk — Action required soon.' if stress > 2.5
               else '🔶 Monitor this region closely.' if stress > 1.5
               else '✅ Water situation is manageable.')
        top = ['Rainfall', 'Population', 'GDP', 'Temperature'][int(np.argmax(model.feature_importances_))]

        thread_conn = sqlite3.connect('water_stress.db', check_same_thread=False)
        thread_conn.execute(
            """INSERT INTO predictions
               (country_name, rainfall_mm, population_millions, gdp_per_capita,
                avg_temperature, predicted_stress, conflict_risk)
               VALUES (?,?,?,?,?,?,?)""",
            (country or 'Custom', rainfall, population, gdp, temperature,
             round(stress, 2), round(conflict, 2))
        )
        thread_conn.commit()
        thread_conn.close()

        html = (
            f"<div style='font-family:Arial;background:#0f1729;border-radius:14px;"
            f"padding:20px;border-left:5px solid {c}'>"
            f"<div style='color:#94a3b8;font-size:11px;margin-bottom:8px'>"
            f"PREDICTION — {datetime.now().strftime('%d %b %Y %H:%M:%S')}</div>"
            f"<div style='color:#ffffff;font-size:18px;font-weight:700;margin-bottom:14px'>"
            f"🌍 {country or 'Custom Region'}</div>"
            f"<div style='display:flex;gap:12px;margin-bottom:14px'>"
            f"<div style='background:#1e2d4a;padding:12px;border-radius:10px;flex:1;text-align:center'>"
            f"<div style='color:#94a3b8;font-size:10px;font-family:Arial'>WATER STRESS</div>"
            f"<div style='color:{c};font-size:28px;font-weight:700'>{stress:.2f}</div>"
            f"<div style='color:#94a3b8;font-size:10px'>out of 5.0</div></div>"
            f"<div style='background:#1e2d4a;padding:12px;border-radius:10px;flex:1;text-align:center'>"
            f"<div style='color:#94a3b8;font-size:10px;font-family:Arial'>RISK LEVEL</div>"
            f"<div style='color:{c};font-size:15px;font-weight:700;margin:6px 0'>{lv}</div></div>"
            f"<div style='background:#1e2d4a;padding:12px;border-radius:10px;flex:1;text-align:center'>"
            f"<div style='color:#94a3b8;font-size:10px;font-family:Arial'>CONFLICT RISK</div>"
            f"<div style='color:#EF4444;font-size:28px;font-weight:700'>{conflict:.0%}</div></div></div>"
            f"<div style='background:#1e2d4a;padding:10px 14px;border-radius:8px;margin-bottom:10px'>"
            f"<span style='color:#94a3b8;font-size:11px'>🔬 Model: </span>"
            f"<span style='color:#ffffff;font-size:11px'>Random Forest · 100 trees · R²={R2:.1%}</span><br>"
            f"<span style='color:#94a3b8;font-size:11px'>📊 Top Driver: </span>"
            f"<span style='color:#38BDF8'>{top}</span></div>"
            f"<div style='background:rgba(239,68,68,0.12);padding:10px 14px;"
            f"border-radius:8px;color:{c};font-size:12px;font-weight:600'>{msg}</div>"
            f"<div style='color:#4b5563;font-size:10px;margin-top:8px'>✅ Saved to database</div></div>"
        )
        return html, build_stats(stress, conflict, country or 'Custom'), build_map(highlight=country)
    except Exception as e:
        return (
            f"<div style='background:#1e0a0a;border-radius:12px;padding:20px;"
            f"border-left:5px solid #EF4444;color:#EF4444;font-family:Arial'>"
            f"❌ Error: {str(e)}</div>"
        ), build_stats(), build_map()

def run_query(choice):
    q = {
        "Top 10 Most Stressed":
            "SELECT country_name AS Country, predicted_stress AS Stress,"
            " ROUND(conflict_risk*100,1)||'%' AS Conflict,"
            " CASE WHEN predicted_stress>3.5 THEN '🔴 CRITICAL'"
            " WHEN predicted_stress>2.5 THEN '🟠 HIGH'"
            " WHEN predicted_stress>1.5 THEN '🟡 MEDIUM'"
            " ELSE '🟢 LOW' END AS Level"
            " FROM predictions GROUP BY country_name"
            " ORDER BY predicted_stress DESC LIMIT 10",
        "All Countries":
            "SELECT country_name AS Country, predicted_stress AS Stress,"
            " ROUND(conflict_risk*100,1)||'%' AS Conflict"
            " FROM predictions GROUP BY country_name ORDER BY predicted_stress DESC",
        "Critical Zone Only":
            "SELECT country_name AS Country, predicted_stress AS Stress,"
            " ROUND(conflict_risk*100,1)||'%' AS Conflict"
            " FROM predictions WHERE predicted_stress>3.5"
            " GROUP BY country_name ORDER BY predicted_stress DESC",
        "Latest Predictions":
            "SELECT country_name AS Country, predicted_stress AS Stress,"
            " ROUND(conflict_risk*100,1)||'%' AS Conflict, timestamp AS Time"
            " FROM predictions ORDER BY id DESC LIMIT 10"
    }
    thread_conn = sqlite3.connect('water_stress.db', check_same_thread=False)
    result = pd.read_sql_query(q.get(choice, q["All Countries"]), thread_conn)
    thread_conn.close()
    return result

OVERVIEW = (
    RESULTS[['country', 'region', 'stress', 'conflict_pct', 'rainfall', 'population', 'gdp', 'temp']]
    .copy()
    .rename(columns={
        'country': 'Country', 'region': 'Region',
        'stress': 'Water Stress', 'conflict_pct': 'Conflict%',
        'rainfall': 'Rainfall(mm)', 'population': 'Pop(M)',
        'gdp': 'GDP($)', 'temp': 'Temp(°C)'
    })
    .sort_values('Water Stress', ascending=False)
    .reset_index(drop=True)
)

def build_alerts():
    critical = (RESULTS['stress'] > 3.5).sum()
    low_rain = (RESULTS['rainfall'] < 200).sum()
    now = datetime.now().strftime("%d %b %Y")
    items = "".join([
        f"<div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #1e3a5f'>"
        f"<span style='font-size:16px;margin-top:2px'>{ic}</span>"
        f"<div><div style='color:#ffffff;font-size:12px;font-weight:600;font-family:Arial'>{tx}</div>"
        f"<div style='color:#94a3b8;font-size:10px;margin-top:2px;font-family:Arial'>{now} · {tm}</div>"
        f"</div></div>"
        for ic, tx, tm in [
            ("🔴", f"High Water Stress — {critical} countries in critical zone", "10:30 AM"),
            ("🟠", f"Low Rainfall Alert — {low_rain} countries below 200mm", "09:15 AM"),
            ("🟡", "Conflict Risk Rising — 6 high-stress regions flagged", "08:45 AM"),
            ("🔵", "Data Updated — 2026 records loaded successfully", "08:30 AM"),
        ]
    ])
    return (
        f"<div style='background:#0f1729;border-radius:12px;padding:16px;font-family:Arial'>"
        f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-bottom:10px'>"
        f"🤖 AI Alerts & Recommendations</div>{items}</div>"
    )

# ── CSS ───────────────────────────────────────────────────
CSS = """
html, body, .gradio-container, .main, .wrap {
    background: #0a1628 !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif !important;
    color-scheme: dark !important;
}
.gradio-html, .gradio-html * {
    font-family: Arial, sans-serif !important;
    color: inherit;
}
.gradio-html > div {
    width: 100% !important;
    overflow: hidden !important;
}

/* ── Tab text ── */
.tab-nav button,
.tab-nav button span,
button[role=tab],
button[role=tab] span {
    color: #ffffff !important;
    background: #1e2d4a !important;
    border: none !important;
    font-family: Arial, sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    opacity: 1 !important;
}
.tab-nav button.selected,
.tab-nav button.selected span,
button[role=tab][aria-selected=true],
button[role=tab][aria-selected=true] span {
    color: #38BDF8 !important;
    background: #0a2342 !important;
    border-bottom: 3px solid #38BDF8 !important;
    opacity: 1 !important;
}
.tab-nav button:hover,
button[role=tab]:hover,
button[role=tab]:hover span {
    color: #38BDF8 !important;
    background: #162033 !important;
    opacity: 1 !important;
}

/* ── FIX: Dataframe / Table ── */
.dataframe,
.dataframe table,
div[data-testid="dataframe"] table,
div[data-testid="dataframe"] {
    background: #0f1729 !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif !important;
    border-collapse: collapse !important;
    width: 100% !important;
}

/* Header row */
.dataframe thead tr th,
div[data-testid="dataframe"] thead tr th,
table thead tr th,
th {
    background: #0a2342 !important;
    color: #38BDF8 !important;
    font-family: Arial, sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 10px 12px !important;
    border-bottom: 2px solid #1e3a5f !important;
    text-align: left !important;
}

/* All data cells */
.dataframe tbody tr td,
div[data-testid="dataframe"] tbody tr td,
table tbody tr td,
td {
    background: #1e2d4a !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif !important;
    font-size: 13px !important;
    padding: 9px 12px !important;
    border-bottom: 1px solid #0f1729 !important;
}

/* Alternating rows */
.dataframe tbody tr:nth-child(even) td,
div[data-testid="dataframe"] tbody tr:nth-child(even) td,
table tbody tr:nth-child(even) td,
tr:nth-child(even) td {
    background: #162033 !important;
    color: #ffffff !important;
}

/* Hover row highlight */
.dataframe tbody tr:hover td,
div[data-testid="dataframe"] tbody tr:hover td,
table tbody tr:hover td {
    background: #1e3a5f !important;
    color: #ffffff !important;
}

/* Fix any svelte-generated white backgrounds */
.svelte-1gfkn6j,
.svelte-po1pjn,
.cell-wrap,
.cell-wrap span,
.data-cell,
.data-cell span {
    background: transparent !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif !important;
}

/* ── Inputs ── */
label, .label-wrap, .block label {
    color: #cbd5e1 !important;
    font-family: Arial !important;
    font-size: 12px !important;
}
input, textarea, select, .input-wrap {
    color: #ffffff !important;
    background: #1e2d4a !important;
    border: 1px solid #334155 !important;
    font-family: Arial !important;
}
input[type=range] { accent-color: #38BDF8 !important; }
.range-value { color: #ffffff !important; font-family: Arial !important; }

/* ── Buttons ── */
button.primary {
    background: #0D9488 !important;
    color: #ffffff !important;
    font-family: Arial !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
}
button.secondary {
    background: #1e2d4a !important;
    color: #ffffff !important;
    font-family: Arial !important;
    border: 1px solid #334155 !important;
}
button:hover { opacity: 0.9 !important; }
.block, .panel, .form, .gap { background: transparent !important; border: none !important; }
.wrap.svelte-xytpvr, select option { background: #1e2d4a !important; color: #ffffff !important; }
.examples { background: #1e2d4a !important; }
.examples td, .examples th { color: #ffffff !important; font-family: Arial !important; }
footer { display: none !important; }
"""
HEADER = (
    f"<div style='background:linear-gradient(135deg,#0a1628,#0d2545);"
    f"padding:20px 28px;border-radius:14px;margin-bottom:4px;"
    f"border-bottom:2px solid #0D9488;font-family:Arial'>"
    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
    f"<div><div style='display:flex;align-items:center;gap:12px'>"
    f"<span style='font-size:28px'>💧</span>"
    f"<span style='color:#ffffff;font-size:22px;font-weight:700;font-family:Arial'>"
    f"AI & ML Based Water Stress Forecasting System</span>"
    f"<span style='color:#38BDF8;font-size:22px;font-weight:700;margin-left:8px'>2026</span></div>"
    f"<div style='color:#94a3b8;font-size:12px;margin-top:4px;margin-left:42px;font-family:Arial'>"
    f"Predicting Water Stress and Conflict Risk Across the Globe"
    f" · Random Forest ML · SQLite Database · Real-time Predictions</div></div>"
    f"<div style='text-align:right'>"
    f"<div style='color:#22C55E;font-size:11px;font-weight:600;font-family:Arial'>● LIVE SYSTEM</div>"
    f"<div style='color:#94a3b8;font-size:10px;font-family:Arial'>"
    f"Data: 2018–2026 · {len(COUNTRIES)} Countries</div>"
    f"</div></div></div>"
)

# ── App Layout ────────────────────────────────────────────
with gr.Blocks(title="💧 Water Stress Forecasting 2026", css=CSS) as app:

    gr.HTML(HEADER)
    stats_bar = gr.HTML(value=build_stats())

    with gr.Tabs():

        with gr.Tab("🌐 Global Risk Map"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    gr.HTML("<div style='color:#ffffff;font-size:13px;font-weight:700;"
                            "margin-bottom:6px;font-family:Arial'>🗺️ Live Geospatial Water Stress Map</div>")
                    map_display = gr.HTML(value=build_map())
                with gr.Column(scale=2):
                    alerts_html = gr.HTML(value=build_alerts())
            with gr.Row():
                gr.Image(value=chart_region(),  show_label=False, show_download_button=False, height=240)
                gr.Image(value=chart_scatter(), show_label=False, show_download_button=False, height=240)
                gr.Image(value=chart_pie(),     show_label=False, show_download_button=False, height=240)
            with gr.Row():
                gr.Image(value=chart_top10(),   show_label=False, show_download_button=False, height=240)
                gr.Image(value=chart_rainfall(), show_label=False, show_download_button=False, height=240)

        with gr.Tab("⚡ Interactive Predictor"):
            gr.HTML("<div style='background:#1e2d4a;border-radius:12px;padding:14px;"
                    "font-family:Arial;margin-bottom:12px'>"
                    "<div style='color:#38BDF8;font-size:14px;font-weight:700;margin-bottom:4px'>"
                    "🤖 AI Water Stress Predictor</div>"
                    "<div style='color:#94a3b8;font-size:11px'>Enter any country's environmental data "
                    "and click PREDICT to get a real ML prediction from our trained Random Forest model"
                    "</div></div>")
            with gr.Row():
                with gr.Column(scale=1):
                    country_in    = gr.Textbox(label="🌍 Country / Region Name",
                                               placeholder="e.g. India, Yemen, Germany...")
                    rainfall_in   = gr.Slider(0, 2000, value=500, step=10,
                                              label="🌧️ Annual Rainfall (mm)")
                    population_in = gr.Slider(1, 1500, value=100, step=5,
                                              label="👥 Population (millions)")
                    gdp_in        = gr.Slider(500, 60000, value=5000, step=500,
                                              label="💰 GDP per Capita (USD)")
                    temp_in       = gr.Slider(5, 45, value=25, step=1,
                                              label="🌡️ Avg Temperature (°C)")
                    predict_btn   = gr.Button("🔍  PREDICT", variant="primary", size="lg")
                with gr.Column(scale=1):
                    result_html = gr.HTML(
                        value="<div style='background:#0f1729;border-radius:12px;padding:30px;"
                              "color:#4b5563;font-family:Arial;text-align:center;font-size:14px'>"
                              "↑ Enter data and click PREDICT</div>")
                    map_small = gr.HTML(value=build_map())
            gr.Examples(
                examples=[
                    ["Yemen", 100, 33, 800, 38],
                    ["Pakistan", 250, 230, 1500, 35],
                    ["Somalia", 80, 17, 600, 36],
                    ["Sudan", 200, 45, 750, 34],
                    ["India", 900, 1400, 2100, 28],
                    ["Germany", 700, 83, 45000, 10],
                    ["USA", 750, 335, 60000, 12],
                    ["Brazil", 1800, 215, 8500, 26],
                    ["Australia", 450, 26, 53000, 22],
                    ["China", 600, 1400, 12000, 14],
                ],
                inputs=[country_in, rainfall_in, population_in, gdp_in, temp_in],
                label="Quick Examples — click any row then hit PREDICT:"
            )

        with gr.Tab("📊 Predictive Analytics"):
            gr.HTML(f"<div style='color:#ffffff;font-size:13px;font-weight:700;"
                    f"margin-bottom:8px;font-family:Arial'>"
                    f"📋 Country Level Water Stress Overview ({len(OVERVIEW)} countries)</div>")
            gr.Dataframe(value=OVERVIEW, wrap=False)
            gr.HTML(
                f"<div style='background:#1e2d4a;border-radius:12px;padding:18px;"
                f"font-family:Arial;margin-top:12px'>"
                f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-bottom:14px'>"
                f"📊 Model Performance Summary</div>"
                f"<div style='display:flex;gap:20px;flex-wrap:wrap'>"
                f"<div style='color:#94a3b8;font-size:12px;line-height:2.2'>"
                f"<b style='color:#ffffff'>Algorithm:</b> Random Forest Regressor<br>"
                f"<b style='color:#ffffff'>Decision Trees:</b> 100<br>"
                f"<b style='color:#ffffff'>Training Records:</b> {len(X_train)}<br>"
                f"<b style='color:#ffffff'>R² Accuracy:</b> "
                f"<span style='color:#22C55E;font-weight:700'>{R2:.1%}</span><br>"
                f"<b style='color:#ffffff'>Avg Error:</b> ±{MAE:.3f}</div>"
                f"<div style='color:#94a3b8;font-size:12px;line-height:2.2'>"
                f"<b style='color:#ffffff'>Data Range:</b> 2018–2026<br>"
                f"<b style='color:#ffffff'>Countries:</b> {len(COUNTRIES)}<br>"
                f"<b style='color:#ffffff'>Regions:</b> {RESULTS['region'].nunique()}<br>"
                f"<b style='color:#ffffff'>Database:</b> SQLite (3 tables)<br>"
                f"<b style='color:#ffffff'>Top Driver:</b> "
                f"<span style='color:#38BDF8'>Rainfall</span></div>"
                f"</div></div>"
            )

        with gr.Tab("🗄️ Relational Database"):
            gr.HTML("<div style='color:#ffffff;font-size:13px;font-weight:700;"
                    "margin-bottom:8px;font-family:Arial'>"
                    "🗄️ Live SQL Queries on SQLite Database</div>")
            with gr.Row():
                with gr.Column(scale=3):
                    query_dd = gr.Dropdown(
                        choices=["Top 10 Most Stressed", "All Countries",
                                 "Critical Zone Only", "Latest Predictions"],
                        value="Top 10 Most Stressed",
                        label="Select Query to Run"
                    )
                    run_btn = gr.Button("▶  Run Query", variant="secondary")
                    db_out  = gr.Dataframe()
                with gr.Column(scale=1):
                    gr.HTML(
                        "<div style='background:#1e2d4a;border-radius:12px;"
                        "padding:18px;font-family:Arial'>"
                        "<div style='color:#38BDF8;font-size:13px;font-weight:700;"
                        "margin-bottom:12px'>🗃️ Database Schema</div>"
                        "<div style='color:#94a3b8;font-size:11px;line-height:2'>"
                        "<b style='color:#ffffff'>Table 1:</b> countries<br>"
                        "<span style='color:#4b5563;font-size:10px'>id, name, region, lat, lon</span><br>"
                        "<b style='color:#ffffff'>Table 2:</b> water_stress<br>"
                        "<span style='color:#4b5563;font-size:10px'>"
                        "country_id, year, rainfall, population, gdp, temp, stress, conflict</span><br>"
                        "<b style='color:#ffffff'>Table 3:</b> predictions<br>"
                        "<span style='color:#4b5563;font-size:10px'>"
                        "country_name, predicted_stress, conflict_risk, timestamp</span>"
                        "</div></div>"
                    )

    # ── Wire up buttons ───────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[country_in, rainfall_in, population_in, gdp_in, temp_in],
        outputs=[result_html, stats_bar, map_small]
    )
    run_btn.click(fn=run_query, inputs=query_dd, outputs=db_out)

print("\n" + "=" * 55)
print(f"  🚀  LAUNCHING — {len(COUNTRIES)} COUNTRIES LOADED")
print("=" * 55)
app.launch(share=True, quiet=True)