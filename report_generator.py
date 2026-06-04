"""
MEDICAL REPORT GENERATOR — drop-in replacement for download_report() in app.py
─────────────────────────────────────────────────────────────────────────────
Install deps once:
    pip install reportlab matplotlib
─────────────────────────────────────────────────────────────────────────────
Usage: copy this file alongside app.py and replace the download_report route
with the one at the bottom.
"""

import io, os, math, textwrap
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── PALETTE ────────────────────────────────────────────────────────────────
C_DARK       = colors.HexColor("#0B1120")
C_NAVY       = colors.HexColor("#0F2044")
C_ACCENT     = colors.HexColor("#1A6FA8")
C_ACCENT_LT  = colors.HexColor("#2A8FD8")
C_LOW        = colors.HexColor("#1A7A4A")
C_LOW_BG     = colors.HexColor("#E8F5EE")
C_MID        = colors.HexColor("#8A5E00")
C_MID_BG     = colors.HexColor("#FFF8E1")
C_HIGH       = colors.HexColor("#8B1A1A")
C_HIGH_BG    = colors.HexColor("#FFF0F0")
C_GREY_DARK  = colors.HexColor("#2C3E50")
C_GREY       = colors.HexColor("#546E7A")
C_GREY_LT    = colors.HexColor("#90A4AE")
C_BORDER     = colors.HexColor("#CFD8DC")
C_ROW_ALT    = colors.HexColor("#F7FAFC")
C_WHITE      = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ─── STYLES ─────────────────────────────────────────────────────────────────
def build_styles():
    return {
        "title": ParagraphStyle("title",
            fontName="Helvetica-Bold", fontSize=22, textColor=C_WHITE,
            leading=28, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle",
            fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#B0C4DE"),
            leading=14),
        "section_head": ParagraphStyle("section_head",
            fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT,
            spaceBefore=14, spaceAfter=6, leading=16,
            borderPad=(0,0,2,0)),
        "label": ParagraphStyle("label",
            fontName="Helvetica-Bold", fontSize=8.5, textColor=C_GREY,
            leading=12, spaceAfter=1),
        "value": ParagraphStyle("value",
            fontName="Helvetica", fontSize=10, textColor=C_DARK,
            leading=14),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=9.5, textColor=C_GREY_DARK,
            leading=15, spaceAfter=5, alignment=TA_JUSTIFY),
        "body_bold": ParagraphStyle("body_bold",
            fontName="Helvetica-Bold", fontSize=9.5, textColor=C_DARK,
            leading=15),
        "bullet": ParagraphStyle("bullet",
            fontName="Helvetica", fontSize=9.5, textColor=C_GREY_DARK,
            leading=15, leftIndent=14, bulletIndent=4, spaceAfter=2),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8, textColor=C_GREY_LT,
            leading=11, alignment=TA_CENTER, spaceAfter=4),
        "risk_label": ParagraphStyle("risk_label",
            fontName="Helvetica-Bold", fontSize=18, leading=22),
        "risk_sub": ParagraphStyle("risk_sub",
            fontName="Helvetica", fontSize=9, textColor=C_GREY,
            leading=13, spaceAfter=4),
        "footer": ParagraphStyle("footer",
            fontName="Helvetica", fontSize=7.5, textColor=C_GREY_LT,
            alignment=TA_CENTER, leading=11),
        "disclaimer": ParagraphStyle("disclaimer",
            fontName="Helvetica-Oblique", fontSize=8, textColor=C_GREY_LT,
            leading=12, alignment=TA_CENTER),
    }


# ─── RISK META ───────────────────────────────────────────────────────────────
def risk_meta(pct):
    if pct < 40:
        return {
            "level": "LOW RISK",
            "color": C_LOW, "bg": C_LOW_BG,
            "icon": "✓",
            "summary": (
                "Your current symptom profile falls within the low-risk range for Long COVID. "
                "This suggests that the post-COVID impact on your daily functioning is relatively "
                "mild at this time."
            ),
            "condition_detail": (
                "Low-risk individuals typically experience sporadic or mild symptoms that do not "
                "significantly impair daily activities. While the prognosis is generally favourable, "
                "it is important to remain vigilant: Long COVID symptoms can fluctuate. Monitoring "
                "your condition over the coming weeks is advisable."
            ),
            "measures": [
                "Maintain a regular sleep schedule (7–9 hours per night) to support immune recovery.",
                "Engage in gentle, graduated physical activity — short daily walks are an excellent start.",
                "Eat a balanced, anti-inflammatory diet rich in vegetables, whole grains, and omega-3 fats.",
                "Limit alcohol and avoid smoking, as both impede respiratory and neurological recovery.",
                "Stay well-hydrated (minimum 2 litres of water daily).",
                "Schedule a follow-up assessment in 4 weeks to track any symptom progression.",
                "Report any sudden worsening of symptoms to a healthcare provider promptly.",
            ],
            "followup": "Recommended reassessment: 4 weeks"
        }
    elif pct < 70:
        return {
            "level": "MEDIUM RISK",
            "color": C_MID, "bg": C_MID_BG,
            "icon": "⚠",
            "summary": (
                "Your current symptom profile falls within the moderate-risk range for Long COVID. "
                "Several symptom clusters are contributing to a meaningful burden on your health "
                "and daily functioning."
            ),
            "condition_detail": (
                "Moderate-risk patients commonly experience a combination of fatigue, cognitive "
                "difficulties (brain fog), and respiratory or neurological symptoms that may "
                "fluctuate in intensity. Without appropriate management, these symptoms can "
                "persist for months. A structured recovery plan and medical guidance are "
                "important at this stage."
            ),
            "measures": [
                "Consult a healthcare provider or Long COVID clinic within the next 1–2 weeks.",
                "Practice pacing: balance activity with rest using the 'energy envelope' technique — "
                "avoid overexertion even on good days.",
                "Consider cognitive rehabilitation exercises if brain fog is prominent (puzzles, mindfulness).",
                "Breathing exercises (pursed-lip breathing, diaphragmatic breathing) for respiratory symptoms.",
                "Keep a daily symptom diary to identify triggers and patterns.",
                "Prioritise sleep hygiene: consistent bedtime, dark and cool room, no screens 1 hour before bed.",
                "Discuss with your doctor whether physiotherapy or occupational therapy referral is appropriate.",
                "Avoid activities that cause symptom flare-ups (post-exertional malaise is common).",
            ],
            "followup": "Recommended reassessment: 2 weeks"
        }
    else:
        return {
            "level": "HIGH RISK",
            "color": C_HIGH, "bg": C_HIGH_BG,
            "icon": "!",
            "summary": (
                "Your current symptom profile falls within the high-risk range for Long COVID. "
                "Multiple symptom clusters are significantly impacting your health and quality "
                "of life. Prompt medical evaluation is strongly recommended."
            ),
            "condition_detail": (
                "High-risk Long COVID patients often present with debilitating fatigue, marked "
                "cognitive impairment, chest pain or tightness, and significant respiratory "
                "difficulties. These symptoms can indicate underlying organ involvement (cardiac, "
                "pulmonary, or neurological). Specialist assessment — including blood tests, "
                "pulmonary function tests, and cardiac screening — may be warranted."
            ),
            "measures": [
                "URGENT: Seek a specialist or Long COVID clinic appointment within the next week.",
                "Do not engage in strenuous exercise — physical overexertion can significantly worsen symptoms.",
                "If you experience chest pain, severe shortness of breath, or palpitations, seek emergency care.",
                "Request comprehensive blood work: full blood count, inflammatory markers (CRP, ESR), D-dimer.",
                "Ask about referral to cardiology and/or pulmonology if chest or breathing symptoms are severe.",
                "Begin a strict pacing protocol — rest before fatigue sets in, not after.",
                "Psychological support (CBT, peer support groups) is strongly recommended alongside physical care.",
                "Inform family members or carers of your condition so they can assist and monitor your wellbeing.",
                "Do not stop any prescribed medications without medical advice.",
            ],
            "followup": "Recommended reassessment: 1 week — with clinical supervision"
        }


# ─── CATEGORY GUIDANCE ───────────────────────────────────────────────────────
CATEGORY_GUIDANCE = {
    "Respiratory": {
        "desc": "Includes cough and breathing difficulty.",
        "advice": "Practice diaphragmatic breathing daily. Avoid cold, dry air. Use a humidifier if needed. "
                  "If breathlessness at rest is present, seek urgent medical attention.",
    },
    "Neurological": {
        "desc": "Includes headache, memory/brain fog, and smell loss.",
        "advice": "Prioritise cognitive rest alongside physical rest. Reduce screen time. "
                  "Smell training (sniffing strong scents like lemon or cloves twice daily) may aid olfactory recovery.",
    },
    "Fatigue": {
        "desc": "Includes general fatigue and physical weakness.",
        "advice": "Post-exertional malaise (PEM) is a hallmark of Long COVID. Use the pacing strategy: "
                  "plan activities in small blocks, take breaks proactively, and never push through fatigue.",
    },
    "Mental": {
        "desc": "Includes sleep difficulties and mental wellbeing.",
        "advice": "Sleep disruption worsens all other Long COVID symptoms. Maintain consistent sleep times, "
                  "limit caffeine after noon, and consider a short mindfulness or breathing routine before bed.",
    },
}


# ─── MATPLOTLIB CHARTS ───────────────────────────────────────────────────────
def make_radar_chart(categories):
    labels = list(categories.keys())
    values = list(categories.values())
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True), facecolor="#F9FBFD")
    ax.set_facecolor("#F9FBFD")

    ax.plot(angles, values_plot, "o-", linewidth=2, color="#1A6FA8")
    ax.fill(angles, values_plot, alpha=0.18, color="#1A6FA8")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color="#2C3E50", fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], size=7, color="#90A4AE")
    ax.grid(color="#CFD8DC", linewidth=0.7, linestyle="--")
    ax.spines["polar"].set_color("#CFD8DC")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=160, bbox_inches="tight", facecolor="#F9FBFD")
    plt.close()
    buf.seek(0)
    return buf


def make_bar_chart(symptom_names, symptom_values):
    colors_bar = [
        "#34D399" if v <= 3 else "#FBBF24" if v <= 6 else "#F87171"
        for v in symptom_values
    ]
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#F9FBFD")
    ax.set_facecolor("#F9FBFD")

    bars = ax.bar(symptom_names, symptom_values, color=colors_bar,
                  edgecolor="white", linewidth=0.8, width=0.6)

    for bar, val in zip(bars, symptom_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=8,
                color="#2C3E50", fontweight="bold")

    ax.set_ylim(0, 11)
    ax.set_ylabel("Severity (0–10)", fontsize=8, color="#546E7A")
    ax.tick_params(axis="x", labelsize=7.5, colors="#546E7A")
    ax.tick_params(axis="y", labelsize=7.5, colors="#90A4AE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFD8DC")
    ax.spines["bottom"].set_color("#CFD8DC")
    ax.grid(axis="y", color="#E8EEF2", linewidth=0.7, linestyle="--")

    legend_patches = [
        mpatches.Patch(color="#34D399", label="Low (0–3)"),
        mpatches.Patch(color="#FBBF24", label="Moderate (4–6)"),
        mpatches.Patch(color="#F87171", label="High (7–10)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="upper right",
              framealpha=0.7, edgecolor="#CFD8DC")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=160, bbox_inches="tight", facecolor="#F9FBFD")
    plt.close()
    buf.seek(0)
    return buf


# ─── PAGE TEMPLATE (header/footer on every page) ─────────────────────────────
class MedicalPageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        self.report_meta = kwargs.pop("report_meta", {})
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header()
            self.draw_footer(total)
            super().showPage()
        super().save()

    def draw_header(self):
        w, h = A4
        # Dark banner
        self.setFillColor(C_DARK)
        self.rect(0, h - 52 * mm, w, 52 * mm, fill=1, stroke=0)
        # Accent stripe
        self.setFillColor(C_ACCENT)
        self.rect(0, h - 52 * mm, 5 * mm, 52 * mm, fill=1, stroke=0)

        # Logo text
        self.setFillColor(C_WHITE)
        self.setFont("Helvetica-Bold", 16)
        self.drawString(MARGIN, h - 18 * mm, "CovidSense AI")
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#90BFDF"))
        self.drawString(MARGIN, h - 25 * mm, "Long COVID Risk Intelligence Report")

        # Right side: patient + date
        m = self.report_meta
        self.setFillColor(C_WHITE)
        self.setFont("Helvetica-Bold", 9)
        name_str = m.get("name", "—")
        self.drawRightString(w - MARGIN, h - 16 * mm, name_str)
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#90BFDF"))
        self.drawRightString(w - MARGIN, h - 23 * mm,
            f"Report Date: {m.get('date', '')}  |  Ref: {m.get('ref', '')}")

        # Risk pill
        risk = m.get("risk_level", "")
        risk_pct = m.get("risk_pct", 0)
        if risk_pct < 40:
            rc = colors.HexColor("#1A7A4A")
        elif risk_pct < 70:
            rc = colors.HexColor("#8A5E00")
        else:
            rc = colors.HexColor("#8B1A1A")

        pill_x = w - MARGIN - 80
        pill_y = h - 40 * mm
        self.setFillColor(rc)
        self.roundRect(pill_x, pill_y, 80, 14, 7, fill=1, stroke=0)
        self.setFillColor(C_WHITE)
        self.setFont("Helvetica-Bold", 8)
        self.drawCentredString(pill_x + 40, pill_y + 3.5, risk)

    def draw_footer(self, total):
        w = A4[0]
        self.setFillColor(C_BORDER)
        self.rect(MARGIN, 12 * mm, w - 2 * MARGIN, 0.3, fill=1, stroke=0)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(C_GREY_LT)
        self.drawString(MARGIN, 8 * mm,
            "⚠ This report is generated by an AI system and is intended for informational purposes only. "
            "It does not constitute medical advice. Always consult a qualified healthcare professional.")
        self.drawRightString(w - MARGIN, 8 * mm,
            f"Page {self._pageNumber} of {total}")


# ─── MAIN GENERATOR ──────────────────────────────────────────────────────────
def generate_medical_report(report_data: dict, output_path: str = "report.pdf"):
    """
    report_data keys:
        patient  : { name, age, gender }
        symptoms : list of 10 floats (0–10)
        result   : (level_str, pct_float, _)
        ml       : { logistic, random_forest, kmeans, hierarchical }
        categories: { Respiratory, Neurological, Fatigue, Mental }
    """
    p  = report_data["patient"]
    sym = report_data["symptoms"]
    res = report_data["result"]
    ml  = report_data["ml"]
    cats= report_data["categories"]

    pct   = float(res[1])
    risk  = risk_meta(pct)
    ref   = datetime.now().strftime("CS-%Y%m%d-%H%M")
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")

    styles = build_styles()

    # Build doc
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=58 * mm, bottomMargin=22 * mm,
        title=f"Long COVID Risk Report – {p['name']}",
        author="CovidSense AI",
    )

    story = []
    W = PAGE_W - 2 * MARGIN   # usable width

    def hr(space_before=6, space_after=6):
        return [Spacer(1, space_before), HRFlowable(width="100%", thickness=0.5, color=C_BORDER), Spacer(1, space_after)]

    def section(title):
        return [Paragraph(f"<font color='#1A6FA8'>■</font> {title}", styles["section_head"])]


    # ── 1. EXECUTIVE SUMMARY BOX ─────────────────────────────────────────────
    risk_color_hex = risk["color"].hexval() if hasattr(risk["color"], "hexval") else "#1A7A4A"

    summary_data = [
        [
            Paragraph(f"<b>{risk['level']}</b>", ParagraphStyle("rl",
                fontName="Helvetica-Bold", fontSize=22,
                textColor=risk["color"], leading=26)),
            Paragraph(
                f"<b>Risk Score: {pct}%</b><br/>"
                f"<font color='#546E7A'>{risk['summary']}</font>",
                ParagraphStyle("rs", fontName="Helvetica", fontSize=9.5,
                    textColor=C_DARK, leading=15)),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[60 * mm, W - 60 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), risk["bg"]),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [6]),
        ("BOX",          (0, 0), (-1, -1), 1, risk["color"]),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
    ]))
    story += [summary_table, Spacer(1, 10)]


    # ── 2. PATIENT INFORMATION ───────────────────────────────────────────────
    story += section("Patient Information")
    info = [
        ["Full Name",      p.get("name", "—"),    "Age",     p.get("age", "—")],
        ["Biological Sex", p.get("gender", "—"),  "Ref. No.", ref],
        ["Report Date",    date_str,               "AI Models Used", "4 (Ensemble)"],
    ]
    info_rows = []
    for row in info:
        info_rows.append([
            Paragraph(row[0], styles["label"]), Paragraph(str(row[1]), styles["value"]),
            Paragraph(row[2], styles["label"]), Paragraph(str(row[3]), styles["value"]),
        ])
    info_table = Table(info_rows, colWidths=[30 * mm, 60 * mm, 30 * mm, 60 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_ROW_ALT),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_WHITE, C_ROW_ALT]),
        ("BOX",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, C_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story += [info_table, Spacer(1, 8)]


    # ── 3. CLINICAL CONDITION OVERVIEW ───────────────────────────────────────
    story += section("Clinical Condition Overview")
    story.append(Paragraph(risk["condition_detail"], styles["body"]))
    story.append(Spacer(1, 4))


    # ── 4. SYMPTOM SEVERITY TABLE ────────────────────────────────────────────
    story += section("Symptom Severity Profile")
    sym_names = ["Fatigue","Cough","Breathing","Headache","Sleep",
                 "Memory","Chest Pain","Smell Loss","Fever","Weakness"]
    sym_header = [Paragraph(c, ParagraphStyle("sh", fontName="Helvetica-Bold",
        fontSize=8.5, textColor=C_WHITE)) for c in ["Symptom","Score","Level","Clinical Note"]]

    def severity_label(v):
        if v <= 3:   return ("Mild",     "#1A7A4A", "#E8F5EE")
        elif v <= 6: return ("Moderate", "#8A5E00", "#FFF8E1")
        else:        return ("Severe",   "#8B1A1A", "#FFF0F0")

    clinical_notes = {
        "Fatigue":    "Persistent tiredness unrelieved by rest; may indicate post-exertional malaise.",
        "Cough":      "Dry or productive; monitor for worsening or blood in sputum.",
        "Breathing":  "Assess oxygen saturation; seek care if SpO₂ < 94%.",
        "Headache":   "Common in Long COVID; rule out secondary causes if severe/sudden.",
        "Sleep":      "Disrupted sleep worsens all Long COVID symptoms; prioritise sleep hygiene.",
        "Memory":     "Brain fog; affects concentration, word-finding, and short-term recall.",
        "Chest Pain": "Requires cardiac evaluation if persistent; do not ignore.",
        "Smell Loss": "Anosmia/hyposmia; smell training may aid recovery.",
        "Fever":      "Low-grade fever is common; monitor for spikes above 38.5°C.",
        "Weakness":   "Muscle weakness may reflect deconditioning or ongoing inflammation.",
    }

    sym_rows = [sym_header]
    for name, val in zip(sym_names, sym):
        lbl, fc, bc = severity_label(val)
        sym_rows.append([
            Paragraph(name, ParagraphStyle("sn", fontName="Helvetica", fontSize=9, textColor=C_DARK)),
            Paragraph(f"{val}/10", ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=9,
                textColor=colors.HexColor(fc))),
            Paragraph(lbl, ParagraphStyle("sl", fontName="Helvetica-Bold", fontSize=8.5,
                textColor=colors.HexColor(fc),
                backColor=colors.HexColor(bc), borderPad=3)),
            Paragraph(clinical_notes.get(name, ""), ParagraphStyle("note",
                fontName="Helvetica", fontSize=8, textColor=C_GREY, leading=11)),
        ])

    sym_table = Table(sym_rows, colWidths=[32 * mm, 18 * mm, 22 * mm, W - 72 * mm])
    sym_style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_ROW_ALT]),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    sym_table.setStyle(TableStyle(sym_style))
    story += [sym_table, Spacer(1, 8)]


    # ── 5. SYMPTOM BAR CHART ─────────────────────────────────────────────────
    story += section("Visualisation — Symptom Severity")
    bar_buf = make_bar_chart(sym_names, sym)
    bar_img = RLImage(bar_buf, width=W, height=W * 0.38)
    story.append(bar_img)
    story.append(Paragraph("Figure 1 — Colour-coded symptom severity across all ten clinical indicators.",
        styles["caption"]))
    story.append(Spacer(1, 6))


    # ── 6. CATEGORY ANALYSIS + RADAR ─────────────────────────────────────────
    story += section("Category Domain Analysis")

    radar_buf = make_radar_chart(cats)
    radar_img = RLImage(radar_buf, width=65 * mm, height=65 * mm)

    cat_content = []
    for cat, val in cats.items():
        lbl, fc, bc = severity_label(val)
        g = CATEGORY_GUIDANCE.get(cat, {})
        cat_content.append(Paragraph(
            f"<b><font color='{fc}'>{cat}</font></b> — {val}/10 ({lbl})<br/>"
            f"<font color='#546E7A'><i>{g.get('desc','')}</i></font>",
            ParagraphStyle("ch", fontName="Helvetica", fontSize=9, leading=14, spaceAfter=3)))
        cat_content.append(Paragraph(g.get("advice", ""), styles["body"]))
        cat_content.append(Spacer(1, 5))

    cat_table = Table([[radar_img, cat_content]], colWidths=[70 * mm, W - 70 * mm])
    cat_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (1, 0), (1, 0),   12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story += [cat_table, Spacer(1, 6)]


    # ── 7. ML MODEL RESULTS ───────────────────────────────────────────────────
    story += section("AI Model Predictions")
    story.append(Paragraph(
        "Four independent machine learning models were applied to your symptom profile. "
        "Each model predicts a cluster class: <b>0 = Low burden</b>, <b>1 = Moderate burden</b>, "
        "<b>2 = High burden</b>. Agreement across models increases diagnostic confidence.",
        styles["body"]))

    ml_header = [Paragraph(c, ParagraphStyle("mlh", fontName="Helvetica-Bold",
        fontSize=8.5, textColor=C_WHITE)) for c in
        ["Model", "Prediction", "Type", "Clinical Significance"]]
    ml_descriptions = {
        "logistic":     ("Logistic Regression",    "Linear discriminant", "Establishes a baseline linear boundary between risk classes."),
        "random_forest":("Random Forest",          "Ensemble tree model",  "Aggregates 100+ decision trees; robust to outliers and noise."),
        "kmeans":       ("K-Means Clustering",     "Unsupervised clustering","Groups patients by symptom similarity; validates class membership."),
        "hierarchical": ("Hierarchical Clustering","Agglomerative",        "Builds nested clusters; confirms population-level grouping."),
    }

    ml_rows = [ml_header]
    for key, (model_name, model_type, sig) in ml_descriptions.items():
        val = ml.get(key, "—")
        v_lbl = {0: "Low (0)", 1: "Moderate (1)", 2: "High (2)"}.get(val, str(val))
        v_col = {0: "#1A7A4A", 1: "#8A5E00", 2: "#8B1A1A"}.get(val, "#546E7A")
        ml_rows.append([
            Paragraph(model_name, ParagraphStyle("mn", fontName="Helvetica-Bold",
                fontSize=9, textColor=C_DARK)),
            Paragraph(v_lbl, ParagraphStyle("mv", fontName="Helvetica-Bold",
                fontSize=9, textColor=colors.HexColor(v_col))),
            Paragraph(model_type, ParagraphStyle("mt", fontName="Helvetica-Oblique",
                fontSize=8.5, textColor=C_GREY)),
            Paragraph(sig, ParagraphStyle("ms", fontName="Helvetica", fontSize=8.5,
                textColor=C_GREY_DARK, leading=12)),
        ])

    ml_table = Table(ml_rows, colWidths=[42 * mm, 26 * mm, 36 * mm, W - 104 * mm])
    ml_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_ROW_ALT]),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [ml_table, Spacer(1, 8)]


    # ── 8. CARE RECOMMENDATIONS ───────────────────────────────────────────────
    story += section("Personalised Care Recommendations")
    story.append(Paragraph(
        f"Based on your <b>{risk['level']}</b> assessment and symptom profile, "
        "the following evidence-informed steps are recommended:",
        styles["body"]))
    story.append(Spacer(1, 5))
    for i, step in enumerate(risk["measures"], 1):
        story.append(Paragraph(f"<b>{i}.</b>  {step}", styles["bullet"]))
    story.append(Spacer(1, 6))

    # Follow-up box
    fu_data = [[Paragraph(f"🗓 {risk['followup']}", ParagraphStyle("fu",
        fontName="Helvetica-Bold", fontSize=10, textColor=C_ACCENT))]]
    fu_table = Table(fu_data, colWidths=[W])
    fu_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#EBF5FB")),
        ("BOX",          (0, 0), (-1, -1), 1, C_ACCENT),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ]))
    story += [fu_table, Spacer(1, 10)]


    # ── 9. DISCLAIMER ────────────────────────────────────────────────────────
    story += hr()
    story.append(Paragraph(
        "DISCLAIMER — This report has been generated by an artificial intelligence system "
        "using self-reported symptom data. It is provided for informational and educational "
        "purposes only and does not constitute a medical diagnosis or professional medical advice. "
        "All clinical decisions must be made in consultation with a qualified and registered healthcare practitioner.",
        styles["disclaimer"]))

    # ── BUILD ─────────────────────────────────────────────────────────────────
    meta = {
        "name": p.get("name", "—"),
        "date": date_str,
        "ref":  ref,
        "risk_level": risk["level"],
        "risk_pct": pct,
    }

    doc.build(
        story,
        canvasmaker=lambda *a, **kw: MedicalPageCanvas(*a, report_meta=meta, **kw)
    )
    return output_path


# ─── FLASK ROUTE (paste into app.py) ─────────────────────────────────────────
"""
from report_generator import generate_medical_report

@app.route("/download_report")
def download_report():
    path = generate_medical_report(last_report, "report.pdf")
    return send_file(path, as_attachment=True)
"""
