import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    chat = None
    OLLAMA_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HCP Commercial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .app-subtitle {
        color: #9aa4b2;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .section-note {
        color: #9aa4b2;
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa4b2;
    }

    .ai-card {
        background: linear-gradient(
            135deg,
            rgba(62, 129, 255, 0.12),
            rgba(120, 80, 255, 0.08)
        );
        border: 1px solid rgba(120, 150, 255, 0.22);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .ai-card h3 {
        margin-top: 0;
    }

    .small-muted {
        color: #8e98a8;
        font-size: 0.78rem;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: linear-gradient(
            145deg,
            #111318 0%,
            #0d0f13 100%
        );

        border: 1px solid #292d35;
        border-radius: 14px;

        padding: 22px;
        min-height: 125px;

        display: flex;
        align-items: center;
        justify-content: space-between;

        box-shadow:
            0 4px 14px rgba(0,0,0,0.18);

        transition:
            border-color 0.2s ease,
            transform 0.2s ease;
    }

    .kpi-card:hover {
        border-color: #555b66;
        transform: translateY(-2px);
    }

    .kpi-content {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .kpi-label {
        color: #a6abb5;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #f5f5f5;
        font-size: 2rem;
        font-weight: 650;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }

    .kpi-subtitle {
        color: #737985;
        font-size: 0.72rem;
        margin-top: 2px;
    }

    .kpi-icon {
        width: 58px;
        height: 58px;

        border-radius: 50%;
        border: 1px solid #3a3e47;

        background: #171a20;

        display: flex;
        align-items: center;
        justify-content: center;

        flex-shrink: 0;
        margin-left: 14px;

        color: #e8e8e8;

        font-size: 30px;
        font-weight: 300;
        line-height: 1;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = (
    Path(__file__).resolve().parent
    / "Data"
    / "synthetic_hcp_segmentation_output.csv"
)

hcp_data = pd.read_csv(DATA_PATH)


# ============================================================
# RECREATE SCORE CONTRIBUTIONS
# ============================================================

scoring_variables = [
    "Patient_Volume",
    "Target_Share",
    "Category_Growth",
    "New_Patient_Share"
]

score_scaler = StandardScaler()

score_scaled = pd.DataFrame(
    score_scaler.fit_transform(
        hcp_data[scoring_variables]
    ),
    columns=scoring_variables,
    index=hcp_data.index
)

score_scaled["Target_Share_Headroom"] = (
    -score_scaled["Target_Share"]
)

score_scaled["Growth_Opportunity"] = (
    score_scaled["Category_Growth"].clip(lower=0)
)


hcp_data["Volume_Contribution"] = (
    0.40 * score_scaled["Patient_Volume"]
)

hcp_data["Headroom_Contribution"] = (
    0.30 * score_scaled["Target_Share_Headroom"]
)

hcp_data["Growth_Contribution"] = (
    0.15 * score_scaled["Growth_Opportunity"]
)

hcp_data["New_Patient_Contribution"] = (
    0.15 * score_scaled["New_Patient_Share"]
)


# ============================================================
# AI EVIDENCE ENGINE
# Python determines the evidence.
# Gemma interprets the evidence when available.
# A deterministic fallback is used when no local LLM is available.
# ============================================================

def build_hcp_evidence(hcp):

    volume_q75 = (
        hcp_data["Patient_Volume"]
        .quantile(0.75)
    )

    drivers = []
    limitations = []


    # --------------------------------------------------------
    # Patient volume
    # --------------------------------------------------------

    if hcp["Patient_Volume"] >= volume_q75:

        drivers.append(
            f"High patient volume "
            f"({hcp['Patient_Volume']:,.0f})"
        )

    else:

        limitations.append(
            f"Patient volume is below the "
            f"high-volume benchmark "
            f"({hcp['Patient_Volume']:,.0f})"
        )


    # --------------------------------------------------------
    # Target share
    # --------------------------------------------------------

    if hcp["Target_Share"] < 0.25:

        drivers.append(
            f"Target-share headroom "
            f"({hcp['Target_Share']:.1%} target share)"
        )

    elif hcp["Target_Share"] >= 0.45:

        limitations.append(
            f"Relatively high existing target share "
            f"({hcp['Target_Share']:.1%})"
        )


    # --------------------------------------------------------
    # Category growth
    # --------------------------------------------------------

    if hcp["Category_Growth"] > 0:

        drivers.append(
            f"Positive category growth "
            f"({hcp['Category_Growth']:.1%})"
        )

    else:

        limitations.append(
            f"Negative category growth "
            f"({hcp['Category_Growth']:.1%})"
        )


    # --------------------------------------------------------
    # New patient share
    # --------------------------------------------------------

    if hcp["New_Patient_Share"] >= 0.30:

        drivers.append(
            f"High new-patient share "
            f"({hcp['New_Patient_Share']:.1%})"
        )

    elif hcp["New_Patient_Share"] < 0.15:

        limitations.append(
            f"Lower new-patient share "
            f"({hcp['New_Patient_Share']:.1%})"
        )


    # --------------------------------------------------------
    # Fallback text
    # --------------------------------------------------------

    if not drivers:

        drivers = [
            "No strong positive opportunity signal identified"
        ]

    if not limitations:

        limitations = [
            "No major limiting signal identified"
        ]


    return drivers, limitations

# ============================================================
# AI FALLBACK — INDIVIDUAL HCP
# Used when Ollama is unavailable.
# ============================================================

def generate_fallback_interpretation(hcp):

    drivers, limitations = build_hcp_evidence(hcp)

    driver_text = " ".join(drivers)
    limitation_text = " ".join(limitations)

    return f"""
### Commercial Profile

{hcp["HCP_ID"]} is a {hcp["Business_Segment"]} HCP with a
{hcp["Priority_Tier"]} classification and an Opportunity Score
of {hcp["Opportunity_Score"]:.2f}. The profile is based on
synthetic commercial signals across patient volume proxy,
target-share headroom, category growth, and new-patient share.

### Key Opportunity Drivers

{driver_text}.

### Limiting Factors

{limitation_text}.

### Priority Rationale

The HCP's Priority Tier and Opportunity Score reflect the
combined contribution of the commercial opportunity signals
used by the analytical scoring framework.

### Recommended Action

The recommended engagement approach is:
**{hcp["Recommended_Engagement_Strategy"]}**

This recommendation is based on the synthetic HCP segmentation
and prioritization framework.

### Important Caveat

This HCP-level information is synthetic and is used only for
methodology demonstration.
"""

# ============================================================
# AI — INDIVIDUAL HCP
# ============================================================

def generate_ai_interpretation(hcp):

    drivers, limitations = (
        build_hcp_evidence(hcp)
    )

    prompt = f"""
You are a life-sciences commercial intelligence copilot.

You are interpreting ONE SYNTHETIC HCP for a methodology
demonstration.

The Python analytical engine has already calculated the
segmentation, opportunity score, priority tier, and evidence
signals.

Your job is to EXPLAIN those outputs, not recalculate them.

STRICT RULES:

- Use only the supplied information.
- Do not invent clinical facts.
- Do not invent patient characteristics.
- Do not invent competitor behavior.
- Do not invent payer information.
- Do not introduce external market facts.
- Do not claim causation.
- Do not describe this as a real physician.
- Do not reinterpret the scoring formula.
- Do not call patient volume real-world patient count.
- Do not contradict the Python-generated opportunity drivers
  or limiting factors.
- Keep the answer concise and commercially focused.
- Do not discuss generic HCP segmentation theory.

Return exactly these sections:

### Commercial Profile

2-3 sentences describing the overall HCP profile.

### Key Opportunity Drivers

Explain the positive signals supplied by Python.

### Limiting Factors

Explain the limiting signals supplied by Python.

### Priority Rationale

Explain why the supplied Opportunity Score and Priority Tier
are commercially relevant. Do not recalculate them.

### Recommended Action

Explain the supplied engagement strategy in practical
commercial terms.

### Important Caveat

One short sentence explaining that the HCP-level data is
synthetic.

------------------------------------------------------------

HCP INFORMATION

HCP ID: {hcp["HCP_ID"]}
Province: {hcp["Province"]}
Specialty: {hcp["Specialty"]}

Patient Volume: {hcp["Patient_Volume"]:,.0f}
Category TRx: {hcp["Category_TRx"]:,.0f}
Target TRx: {hcp["Target_TRx"]:,.0f}
Target Share: {hcp["Target_Share"]:.1%}
Category Growth: {hcp["Category_Growth"]:.1%}
New Patient Share: {hcp["New_Patient_Share"]:.1%}

Business Segment: {hcp["Business_Segment"]}
K-Means Cluster: {hcp["KMeans_Cluster"]}

Opportunity Score: {hcp["Opportunity_Score"]:.2f}
Priority Tier: {hcp["Priority_Tier"]}

Field Engagement: {hcp["Field_Engagement"]:.0f}
Digital Engagement: {hcp["Digital_Engagement"]:.0f}

Recommended Engagement Strategy:
{hcp["Recommended_Engagement_Strategy"]}

------------------------------------------------------------

PYTHON-GENERATED POSITIVE SIGNALS:

{chr(10).join("- " + x for x in drivers)}

PYTHON-GENERATED LIMITING SIGNALS:

{chr(10).join("- " + x for x in limitations)}
"""

    try:
        if chat is None:
            raise RuntimeError("Ollama package is unavailable.")

        response = chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"], True

    except Exception:
        return generate_fallback_interpretation(hcp), False


# ============================================================
# AI — PORTFOLIO COPILOT
# ============================================================

def generate_copilot_answer(
    question,
    filtered,
    selected_hcp=None
):

    if filtered.empty:

        return (
            "There are no HCPs matching the current filters."
        )

    def generate_copilot_fallback(question, filtered_data):
        question_lower = question.lower()

        if any(
                phrase in question_lower
                for phrase in [
                    "prioritize for conversion",
                    "prioritize",
                    "conversion"
                ]
        ):
            conversion_hcps = (
                filtered_data[
                    filtered_data["Business_Segment"]
                    .isin([
                        "High-Potential Conversion",
                        "Emerging Opportunity"
                    ])
                ]
                    .sort_values(
                    "Opportunity_Score",
                    ascending=False
                )
                    .head(5)
            )

            if conversion_hcps.empty:
                return (
                    "No HCPs in the filtered population match the "
                    "conversion-oriented business segments."
                )

            lines = []

            for _, row in conversion_hcps.iterrows():
                lines.append(
                    f"- **{row['HCP_ID']}** — "
                    f"{row['Specialty']}, {row['Province']} | "
                    f"{row['Business_Segment']} | "
                    f"Commercial Opportunity {row['Opportunity_Score']:.2f}"
                )

            return (
                    "### Conversion-Oriented HCPs\n\n"
                    "Within the currently filtered HCP population, the "
                    "highest-opportunity HCPs in the conversion-oriented "
                    "segments are:\n\n"
                    + "\n".join(lines)
                    + "\n\n"
                      "These results are based on the synthetic segmentation "
                      "and Opportunity Score."
            )

        elif any(
                phrase in question_lower
                for phrase in [
                    "specialt",
                    "strongest opportunity",
                    "highest opportunity"
                ]
        ):
            specialty_summary = (
                filtered_data
                .groupby("Specialty")
                .agg(
                    HCP_Count=("HCP_ID", "count"),
                    Avg_Opportunity=("Opportunity_Score", "mean")
                )
                .sort_values(
                    "Avg_Opportunity",
                    ascending=False
                )
                .head(5)
                .reset_index()
            )

            lines = []

            for _, row in specialty_summary.iterrows():
                lines.append(
                    f"- **{row['Specialty']}** — "
                    f"Average Commercial Opportunity "
                    f"{row['Avg_Opportunity']:.2f} "
                    f"({row['HCP_Count']} HCPs)"
                )

            return (
                    "### Specialty Opportunity\n\n"
                    "The specialties with the highest average commercial "
                    "opportunity in the current filtered population are:\n\n"
                    + "\n".join(lines)
                    + "\n\n"
                      "Average opportunity indicates opportunity intensity; "
                      "it should be considered alongside HCP count."
            )

        elif any(
                phrase in question_lower
                for phrase in [
                    "where",
                    "province",
                    "geograph",
                    "concentrated"
                ]
        ):
            province_summary = (
                filtered_data
                .groupby("Province")
                .agg(
                    HCP_Count=("HCP_ID", "count"),
                    High_Priority_HCPs=(
                        "Priority_Tier",
                        lambda x: (x == "High Priority").sum()
                    ),
                    Avg_Opportunity=(
                        "Opportunity_Score",
                        "mean"
                    )
                )
                .sort_values(
                    "High_Priority_HCPs",
                    ascending=False
                )
                .head(5)
                .reset_index()
            )

            lines = []

            for _, row in province_summary.iterrows():
                lines.append(
                    f"- **{row['Province']}** — "
                    f"{row['High_Priority_HCPs']} high-priority HCPs, "
                    f"{row['HCP_Count']} HCPs overall, "
                    f"average Commercial Opportunity "
                    f"{row['Avg_Opportunity']:.2f}"
                )

            return (
                    "### Geographic Opportunity\n\n"
                    "The provinces with the highest concentration of "
                    "high-priority HCPs in the current filtered population are:\n\n"
                    + "\n".join(lines)
            )

        return (
            "I can currently answer grounded questions about "
            "conversion-oriented HCPs, specialty opportunity, and "
            "geographic concentration using the filtered synthetic dataset. "
            "Try one of those question types."
        )

    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    segment_summary = (
        filtered["Business_Segment"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # Priority summary
    # --------------------------------------------------------

    priority_summary = (
        filtered["Priority_Tier"]
        .value_counts()
        .reindex(
            [
                "High Priority",
                "Medium Priority",
                "Low Priority"
            ],
            fill_value=0
        )
        .to_dict()
    )


    # --------------------------------------------------------
    # Specialty summary
    # --------------------------------------------------------

    specialty_summary = (
        filtered
        .groupby("Specialty")
        .agg(
            HCP_Count=("HCP_ID", "count"),
            Avg_Opportunity_Score=(
                "Opportunity_Score",
                "mean"
            ),
            Avg_Target_Share=(
                "Target_Share",
                "mean"
            ),
            Avg_Category_Growth=(
                "Category_Growth",
                "mean"
            )
        )
        .sort_values(
            "Avg_Opportunity_Score",
            ascending=False
        )
        .head(8)
        .round(3)
        .to_dict("index")
    )


    # --------------------------------------------------------
    # Top HCPs
    # --------------------------------------------------------

    top_hcps = (
        filtered
        .sort_values(
            "Opportunity_Score",
            ascending=False
        )
        .head(10)
        [
            [
                "HCP_ID",
                "Province",
                "Specialty",
                "Business_Segment",
                "Priority_Tier",
                "Opportunity_Score",
                "Patient_Volume",
                "Target_Share",
                "Category_Growth",
                "New_Patient_Share",
                "Recommended_Engagement_Strategy"
            ]
        ]
        .to_dict("records")
    )


    # --------------------------------------------------------
    # Selected HCP context
    # --------------------------------------------------------

    selected_context = ""

    if selected_hcp is not None:

        selected_context = f"""
SELECTED HCP:

{selected_hcp["HCP_ID"]}
Province = {selected_hcp["Province"]}
Specialty = {selected_hcp["Specialty"]}
Business Segment = {selected_hcp["Business_Segment"]}
Priority = {selected_hcp["Priority_Tier"]}
Opportunity Score = {selected_hcp["Opportunity_Score"]:.2f}
"""


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a life-sciences commercial intelligence copilot.

Answer the user's commercial question using ONLY the analytical
evidence supplied below.

The Python engine has already performed all calculations and
filtering.

Do not invent data.
Do not recalculate metrics.
Do not introduce external market facts.
Do not make causal claims.

USER QUESTION:

{question}

FILTERED HCP COUNT:

{len(filtered):,}

BUSINESS SEGMENT COUNTS:

{segment_summary}

PRIORITY COUNTS:

{priority_summary}

TOP SPECIALTIES BY AVERAGE OPPORTUNITY SCORE:

{specialty_summary}

TOP HCPs BY OPPORTUNITY SCORE:

{top_hcps}

{selected_context}

Answer the question directly.

Use supplied numbers where useful.

If the evidence is insufficient, explicitly say so.

Remember:

This is synthetic data for methodology demonstration only.
"""

    try:
        if chat is None:
            raise RuntimeError("Ollama package is unavailable.")

        response = chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"], True

    except Exception:
        return generate_copilot_fallback(
            question,
            filtered_data
        ), False


# ============================================================
# HEADER
# ============================================================

st.title(
    "HCP Commercial Intelligence"
)

st.markdown(
    """
    <div class="app-subtitle">
        Synthetic commercial segmentation, prioritization
        and AI-assisted interpretation
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Synthetic data for methodology demonstration only. "
    "It does not represent real physicians, patients, prescriptions, "
    "products, or commercial performance."
)

st.markdown(
    """
    <div class="ai-card">

    <h3>How to explore this demo</h3>

    <ol>
        <li>Use the sidebar to define the HCP universe you want to analyze.</li>
        <li>Start with <b>Overview</b> to understand segment, priority, and engagement patterns.</li>
        <li>Use <b>Opportunity</b> to identify where commercial opportunity is concentrated across specialties and provinces.</li>
        <li>Use <b>HCP Explorer</b> to understand why an individual HCP is prioritized and what engagement approach is recommended.</li>
        <li>Use <b>AI Copilot</b> to translate analytical signals into concise commercial interpretation.</li>
    </ol>

    <div class="small-muted">
        The analytical engine determines the segmentation and prioritization;
        AI is used only to interpret the resulting evidence.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

def reset_filters():
    st.session_state["province_filter"] = "All"
    st.session_state["specialty_filter"] = "All"
    st.session_state["segment_filter"] = "All"
    st.session_state["priority_filter"] = "All"


all_provinces = sorted(
    hcp_data["Province"].unique()
)

all_specialties = sorted(
    hcp_data["Specialty"].unique()
)

all_segments = sorted(
    hcp_data["Business_Segment"].unique()
)

all_priorities = [
    "High Priority",
    "Medium Priority",
    "Low Priority"
]


selected_provinces = st.sidebar.selectbox(
    "Province",
    ["All"] + all_provinces,
    key="province_filter"
)

selected_specialties = st.sidebar.selectbox(
    "Specialty",
    ["All"] + all_specialties,
    key="specialty_filter"
)

selected_segments = st.sidebar.selectbox(
    "Business Segment",
    ["All"] + all_segments,
    key="segment_filter"
)

selected_priority = st.sidebar.selectbox(
    "Priority Tier",
    ["All"] + all_priorities,
    key="priority_filter"
)

if st.sidebar.button(
    "Reset Filters",
    on_click=reset_filters
):
    st.rerun()



# ============================================================
# APPLY FILTERS
# ============================================================

filtered_data = hcp_data.copy()

if selected_provinces != "All":
    filtered_data = filtered_data[
        filtered_data["Province"] == selected_provinces
    ]

if selected_specialties != "All":
    filtered_data = filtered_data[
        filtered_data["Specialty"] == selected_specialties
    ]

if selected_segments != "All":
    filtered_data = filtered_data[
        filtered_data["Business_Segment"] == selected_segments
    ]

if selected_priority != "All":
    filtered_data = filtered_data[
        filtered_data["Priority_Tier"] == selected_priority
    ]


# ============================================================
# TABS
# ============================================================

overview_tab, opportunity_tab, explorer_tab, ai_tab = st.tabs(
    [
        "Overview",
        "Opportunity",
        "HCP Explorer",
        "AI Copilot"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.subheader(
        "Commercial Overview"
    )


    # ========================================================
    # KPI CARD FUNCTION
    # ========================================================

    def kpi_card(
        title,
        value,
        subtitle,
        icon
    ):

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-content">

                    <div class="kpi-label">
                        {title}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-subtitle">
                        {subtitle}
                    </div>

                </div>

                <div class="kpi-icon">
                    {icon}
                </div>

            </div>
            """
        )


    # ========================================================
    # KPI ICONS
    #
    # Using text symbols rather than inline SVG.
    # This avoids Streamlit HTML sanitization issues.
    # ========================================================

    globe_icon = "◎"

    activity_icon = "⌁"

    target_icon = "⊙"

    opportunity_icon = "✦"


    # ========================================================
    # KPI VALUES
    # ========================================================

    if filtered_data.empty:

        avg_patient_volume = 0

        avg_target_share = 0

        avg_opportunity_score = 0

    else:

        avg_patient_volume = (
            filtered_data[
                "Patient_Volume"
            ].mean()
        )

        avg_target_share = (
            filtered_data[
                "Target_Share"
            ].mean()
        )

        avg_opportunity_score = (
            filtered_data[
                "Opportunity_Score"
            ].mean()
        )


    # ========================================================
    # KPI CARDS
    # ========================================================
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)


    with col1:

        kpi_card(
            "HCP Universe",
            f"{len(filtered_data):,}",
            "Synthetic HCP population",
            globe_icon
        )


    with col2:

        kpi_card(
            "Avg Patient Volume",
            f"{avg_patient_volume:,.0f}",
            "Annual synthetic volume",
            activity_icon
        )


    with col3:

        kpi_card(
            "Avg Target Share",
            f"{avg_target_share:.1%}",
            "Current category share",
            target_icon
        )


    with col4:

        kpi_card(
            "Avg Opportunity",
            f"{avg_opportunity_score:.2f}",
            "Relative prioritization signal",
            opportunity_icon
        )

    st.write("")

    # ========================================================
    # EXECUTIVE TAKEAWAYS
    # ========================================================

    st.subheader("Executive Takeaways")

    if filtered_data.empty:

        st.info(
            "Adjust the filters to generate commercial takeaways."
        )

    else:

        high_priority_pct = (
                (
                    filtered_data["Priority_Tier"]
                    .eq("High Priority")
                    .mean()
                )
                * 100
        )

        high_potential_count = (
            filtered_data["Business_Segment"]
            .eq("High-Potential Conversion")
            .sum()
        )

        high_potential_pct = (
                high_potential_count
                / len(filtered_data)
                * 100
        )

        high_priority = filtered_data[
            filtered_data["Priority_Tier"] == "High Priority"
            ]

        if high_priority.empty:
            top_province = "No High Priority HCPs"
        else:
            top_province = (
                high_priority["Province"]
                .value_counts()
                .idxmax()
            )

        top_specialty = (
            filtered_data
            .groupby("Specialty")[
                "Opportunity_Score"
            ]
            .mean()
            .idxmax()
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                **Priority concentration**

                **{high_priority_pct:.1f}%** of the filtered HCP
                universe is classified as **High Priority**.

                This indicates the proportion of the current universe
                receiving the highest commercial prioritization signal.
                """
            )

        with col2:

            st.markdown(
                f"""
                **Conversion opportunity**

                **{high_potential_pct:.1f}%** of the filtered HCP
                universe falls into the **High-Potential Conversion**
                segment.

                This segment represents HCPs identified by the framework
                as having conversion-oriented opportunity characteristics.
                """
            )

        with col3:

            st.markdown(
                f"""
                **Geographic concentration**

                **{top_province}** has the largest number of High Priority
                HCPs in the current filtered universe.

                **{top_specialty}** has the highest average commercial
                opportunity score among specialties.
                """
            )
        st.caption(
            "Commercial Opportunity combines Patient Volume (40%), Share Headroom (30%), "
            "Category Growth (15%), and New Patient Share (15%)."
        )


        # CANADA OPPORTUNITY MAP

        st.subheader("Geographic Commercial Opportunity")

        st.caption(
            "Commercial opportunity represents the average opportunity score of HCPs within each province. "
            "Provinces are shown in relative opportunity bands based on the currently selected filters."
        )

        province_summary = (
            filtered_data
            .groupby("Province")
            .agg(
                HCPs=("HCP_ID", "count"),
                High_Priority_HCPs=(
                    "Priority_Tier",
                    lambda x: (x == "High Priority").sum()
                ),
                Avg_Commercial_Opportunity=(
                    "Opportunity_Score",
                    "mean"
                )
            )
            .reset_index()
        )

        province_summary["High_Priority_%"] = (
                province_summary["High_Priority_HCPs"]
                / province_summary["HCPs"]
                * 100
        )

        if len(province_summary) >= 4:

            province_summary["Opportunity_Band"] = pd.qcut(
                province_summary["Avg_Commercial_Opportunity"].rank(
                    method="first"
                ),
                q=4,
                labels=[
                    "Lower Opportunity",
                    "Moderate Opportunity",
                    "High Opportunity",
                    "Very High Opportunity"
                ]
            )

        else:

            province_summary["Opportunity_Band"] = "Selected Province"

        color_map = {
            "Lower Opportunity": "#3D3B5C",
            "Moderate Opportunity": "#514B78",
            "High Opportunity": "#5367A8",
            "Very High Opportunity": "#4267E8",
            "Selected Province": "#8B5CF6"

        }

        map_data = province_summary.copy()

        fig_map = px.choropleth(
            map_data,
            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson",
            locations="Province",
            featureidkey="properties.name",
            color="Opportunity_Band",
            color_discrete_map=color_map,
            category_orders={
                "Opportunity_Band": [
                    "Lower Opportunity",
                    "Moderate Opportunity",
                    "High Opportunity",
                    "Very High Opportunity",
                    "Selected Province"
                ]
            },
            hover_name="Province",
            hover_data={
                "HCPs": True,
                "High_Priority_HCPs": True,
                "High_Priority_%": ":.1f",
                "Avg_Commercial_Opportunity": ":.1f",
                "Opportunity_Band": True
            },
            labels={
                "HCPs": "HCPs",
                "High_Priority_HCPs": "High Priority HCPs",
                "High_Priority_%": "High Priority %",
                "Avg_Commercial_Opportunity": "Avg Commercial Opportunity",
                "Opportunity_Band": "Opportunity Band"
            }
        )

        # TERRITORIES: NO DATA

        territory_data = pd.DataFrame({
            "Province": [
                "Yukon",
                "Northwest Territories",
                "Nunavut"
            ]
        })

        fig_map.add_trace(
            go.Choropleth(
                geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson",
                locations=territory_data["Province"],
                featureidkey="properties.name",
                z=[1, 1, 1],
                colorscale=[
                    [0, "#292D35"],
                    [1, "#292D35"]
                ],
                showscale=False,
                marker_line_color="rgba(255,255,255,0.25)",
                marker_line_width=1,
                hovertemplate="<b>%{location}</b><br>No data available<extra></extra>",
                showlegend=False
            )
        )

        fig_map.update_geos(
            showcountries=True,
            countrycolor="rgba(255,255,255,0.55)",
            countrywidth=1.2,
            showcoastlines=False,
            showland=False,
            showlakes=False,
            fitbounds="locations",
            bgcolor="rgba(0,0,0,0)",
            showframe=False
        )

        fig_map.update_traces(
            marker_line_color="rgba(255,255,255,0.45)",
            marker_line_width=1.2
        )
        province_labels = {
            "British Columbia": (54.5, -125.5),
            "Alberta": (54.5, -114.0),
            "Saskatchewan": (54.0, -106.0),
            "Manitoba": (54.5, -98.5),
            "Ontario": (49.5, -84.0),
            "Quebec": (51.5, -71.5),
            "New Brunswick": (46.5, -66.5),
            "Nova Scotia": (45.0, -63.0),
            "Prince Edward Island": (46.4, -63.2),
            "Newfoundland and Labrador": (53.5, -57.0)
        }

        for province, (lat, lon) in province_labels.items():

            province_row = province_summary[
                province_summary["Province"] == province
                ]

            if province_row.empty:
                continue

            opportunity = province_row["Avg_Commercial_Opportunity"].iloc[0]

            fig_map.add_scattergeo(
                lat=[lat],
                lon=[lon],
                text=[
                    f"<b>{province}</b><br>{opportunity:.1f}"
                ],
                mode="text",
                textfont=dict(
                    color="white",
                    size=10
                ),
                hoverinfo="skip",
                showlegend=False
            )
        fig_map.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(
                title=dict(
                    text="Commercial Opportunity",
                    font=dict(color="white")
                ),
                font=dict(color="white"),
                orientation="v",
                x=1.02,
                y=0.5
            )
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

        # GEOGRAPHIC COMMERCIAL OPPORTUNITY

        st.subheader("Geographic Commercial Opportunity")


        if len(province_summary) >= 4:

            province_summary["Opportunity_Band"] = pd.qcut(
                province_summary["Avg_Commercial_Opportunity"].rank(
                    method="first"
                ),
                q=4,
                labels=[
                    "Lower Opportunity",
                    "Moderate Opportunity",
                    "High Opportunity",
                    "Very High Opportunity"
                ]
            )

        else:

            province_summary["Opportunity_Band"] = (
                "Selected Province"
            )

        province_summary = province_summary.sort_values(
            "Avg_Commercial_Opportunity",
            ascending=False
        )

        st.dataframe(
            province_summary[
                [
                    "Province",
                    "HCPs",
                    "High_Priority_HCPs",
                    "Avg_Commercial_Opportunity",
                    "High_Priority_%",
                    "Opportunity_Band"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # SEGMENT + PRIORITY
    # ========================================================

    st.subheader("How to Interpret the Segmentation")

    st.info(
        "**Business Segment** → describes who the HCP is commercially  \n"
        "↓  \n"
        "**Priority Tier** → identifies how strongly to prioritize the HCP  \n"
        "↓  \n"
        "**Engagement Strategy** → recommends how to engage the HCP"
    )

    st.caption("Profile → Prioritize → Act")

    col1, col2 = st.columns(2)



    with col1:

        st.subheader(
            "Business Segment Distribution"
        )
        st.caption(
            "Groups HCPs by their current commercial position and growth potential, "
            "from established/maintenance profiles to conversion and emerging opportunities."
        )

        segment_counts = (
            filtered_data[
                "Business_Segment"
            ]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Business_Segment",
            "HCP_Count"
        ]


        fig = px.pie(
            segment_counts,
            names="Business_Segment",
            values="HCP_Count",
            hole=0.55,
            template="plotly_dark"
        )


        fig.update_layout(
            height=400,
            legend_title_text=""
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Priority Distribution"
        )
        st.caption(
            "Ranks HCPs by overall commercial opportunity using patient volume, "
            "target share headroom, category growth, and new-patient share."
        )

        priority_counts = (
            filtered_data[
                "Priority_Tier"
            ]
            .value_counts()
            .reindex(
                all_priorities,
                fill_value=0
            )
            .reset_index()
        )

        priority_counts.columns = [
            "Priority_Tier",
            "HCP_Count"
        ]


        fig = px.bar(
            priority_counts,
            x="Priority_Tier",
            y="HCP_Count",
            text="HCP_Count",
            template="plotly_dark"
        )


        fig.update_traces(
            textposition="outside"
        )


        fig.update_layout(
            height=400,
            xaxis_title=None,
            yaxis_title="HCP Count"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # PRIORITY + ENGAGEMENT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Priority by Business Segment"
        )
        st.caption(
            "Shows how commercial priority is distributed within each HCP business segment, highlighting where high-priority opportunities are concentrated."
        )

        segment_priority = (
            filtered_data
            .groupby(
                [
                    "Business_Segment",
                    "Priority_Tier"
                ]
            )
            .size()
            .reset_index(
                name="HCP_Count"
            )
        )

        fig = px.bar(
            segment_priority,
            x="Business_Segment",
            y="HCP_Count",
            color="Priority_Tier",
            barmode="stack",
            category_orders={
                "Priority_Tier": all_priorities
            },
            template="plotly_dark"
        )

        fig.update_layout(
            height=450,
            xaxis_tickangle=-20,
            xaxis_title=None,
            yaxis_title="HCP Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Engagement Strategy Mix"
        )
        st.caption(
            "Recommended engagement strategy translates HCP segmentation and priority into an actionable engagement approach, such as maintaining, developing, converting, or expanding the relationship."
        )

        engagement_counts = (
            filtered_data[
                "Recommended_Engagement_Strategy"
            ]
            .value_counts()
            .reset_index()
        )

        engagement_counts.columns = [
            "Recommended_Engagement_Strategy",
            "HCP_Count"
        ]

        fig = px.bar(
            engagement_counts,
            x="HCP_Count",
            y="Recommended_Engagement_Strategy",
            orientation="h",
            template="plotly_dark"
        )

        fig.update_layout(
            height=450,
            xaxis_title="HCP Count",
            yaxis_title=None
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# OPPORTUNITY TAB
# ============================================================

with opportunity_tab:

    st.subheader(
        "Opportunity Landscape"
    )

    st.info(
        "**Commercial Opportunity Methodology**  \n"
        "Patient Volume Proxy **40%** • "
        "Target Share Headroom **30%** • "
        "Category Growth **15%** • "
        "New Patient Share **15%**"
    )

    st.caption(
        "Patient Volume Proxy represents synthetic modeled patient activity "
        "and is not derived from actual claims or prescription data."
    )


    st.markdown(
        """
        <div class="section-note">
            Identify HCPs with scale, target-share headroom
            and positive market signals.
        </div>
        """,
        unsafe_allow_html=True
    )


    if filtered_data.empty:

        st.warning(
            "No HCPs match the current filters."
        )

    else:

        # ====================================================
        # LANDSCAPE
        # ====================================================

        opportunity_data = (
            filtered_data.copy()
        )


        opportunity_data[
            "Opportunity_Size"
        ] = (
            opportunity_data[
                "Opportunity_Score"
            ]
            - hcp_data[
                "Opportunity_Score"
            ].min()
            + 0.1
        )


        fig = px.scatter(
            opportunity_data,
            x="Target_Share",
            y="Patient_Volume",
            color="Business_Segment",
            size="Opportunity_Size",
            hover_data={
                "HCP_ID": True,
                "Province": True,
                "Specialty": True,
                "Business_Segment": True,
                "Priority_Tier": True,
                "Opportunity_Score": ":.2f",
                "Patient_Volume": ":,.0f",
                "Target_Share": ":.1%",
                "Category_Growth": ":.1%",
                "New_Patient_Share": ":.1%",
                "Opportunity_Size": False
            },
            opacity=0.65,
            template="plotly_dark",
            labels={
                "Target_Share":
                    "Target Share",
                "Patient_Volume":
                    "Patient Volume",
                "Business_Segment":
                    "Business Segment"
            }
        )


        fig.update_traces(
            marker=dict(
                line=dict(
                    width=0.5,
                    color="rgba(255,255,255,0.6)"
                )
            )
        )


        fig.update_layout(
            height=600,
            xaxis=dict(
                tickformat=".0%"
            ),
            legend_title_text="Business Segment"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # SPECIALTY
        # ====================================================

        st.subheader(
            "Opportunity by Specialty"
        )

        st.caption(
            "Compares average commercial opportunity across specialties. "
            "Use alongside HCP count to distinguish opportunity intensity from market size."
        )


        specialty_opportunity = (
            filtered_data
            .groupby("Specialty")
            .agg(
                HCP_Count=(
                    "HCP_ID",
                    "count"
                ),
                Avg_Opportunity_Score=(
                    "Opportunity_Score",
                    "mean"
                ),
                Avg_Patient_Volume=(
                    "Patient_Volume",
                    "mean"
                ),
                Avg_Target_Share=(
                    "Target_Share",
                    "mean"
                ),
                Avg_Category_Growth=(
                    "Category_Growth",
                    "mean"
                ),
                Avg_New_Patient_Share=(
                    "New_Patient_Share",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                "Avg_Opportunity_Score",
                ascending=True
            )
        )


        fig = px.bar(
            specialty_opportunity,
            x="Avg_Opportunity_Score",
            y="Specialty",
            orientation="h",
            text="Avg_Opportunity_Score",
            template="plotly_dark",
            hover_data=[
                "HCP_Count",
                "Avg_Patient_Volume",
                "Avg_Target_Share",
                "Avg_Category_Growth",
                "Avg_New_Patient_Share"
            ],
            labels={
                "Avg_Opportunity_Score":
                    "Average Opportunity Score"
            }
        )


        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )


        fig.update_layout(
            height=550,
            yaxis_title=None
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # PROVINCE ANALYSIS
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "High-Priority HCPs by Province"
            )


            high_priority = (
                filtered_data[
                    filtered_data[
                        "Priority_Tier"
                    ]
                    == "High Priority"
                ]
            )


            province_priority = (
                high_priority[
                    "Province"
                ]
                .value_counts()
                .reset_index()
            )


            province_priority.columns = [
                "Province",
                "HCP_Count"
            ]


            province_priority = (
                province_priority
                .sort_values(
                    "HCP_Count",
                    ascending=True
                )
            )


            fig = px.bar(
                province_priority,
                x="HCP_Count",
                y="Province",
                orientation="h",
                template="plotly_dark",
                text="HCP_Count"
            )


            fig.update_traces(
                textposition="outside"
            )


            fig.update_layout(
                height=500,
                yaxis_title=None
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            st.subheader(
                "Average Opportunity by Province"
            )


            province_opportunity = (
                filtered_data
                .groupby("Province")
                .agg(
                    HCP_Count=(
                        "HCP_ID",
                        "count"
                    ),
                    Avg_Opportunity_Score=(
                        "Opportunity_Score",
                        "mean"
                    )
                )
                .reset_index()
                .sort_values(
                    "Avg_Opportunity_Score",
                    ascending=True
                )
            )


            fig = px.bar(
                province_opportunity,
                x="Avg_Opportunity_Score",
                y="Province",
                orientation="h",
                template="plotly_dark",
                text="Avg_Opportunity_Score"
            )


            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )


            fig.update_layout(
                height=500,
                yaxis_title=None
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # ============================================================
    # HCP EXPLORER
    # ============================================================

    with explorer_tab:

        st.subheader(
            "HCP Explorer"
        )


        st.markdown(
            """
            <div class="section-note">
                Move from portfolio-level signals to individual
                HCP prioritization.
            </div>
            """,
            unsafe_allow_html=True
        )


        if filtered_data.empty:

            st.warning(
                "No HCPs match the current filters."
            )

        else:

            # ====================================================
            # TOP HCP TABLE
            # ====================================================

            st.subheader(
                "Top Opportunity HCPs"
            )


            top_hcps = (
                filtered_data
                .sort_values(
                    "Opportunity_Score",
                    ascending=False
                )
                .head(25)
                [
                    [
                        "HCP_ID",
                        "Province",
                        "Specialty",
                        "Business_Segment",
                        "Priority_Tier",
                        "Patient_Volume",
                        "Target_Share",
                        "Category_Growth",
                        "New_Patient_Share",
                        "Opportunity_Score",
                        "Recommended_Engagement_Strategy"
                    ]
                ]
                .copy()
            )


            top_hcps = top_hcps.rename(
                columns={
                    "HCP_ID":
                        "HCP",

                    "Business_Segment":
                        "Business Segment",

                    "Priority_Tier":
                        "Priority",

                    "Patient_Volume":
                        "Patient Volume",

                    "Target_Share":
                        "Target Share",

                    "Category_Growth":
                        "Category Growth",

                    "New_Patient_Share":
                        "New Patient Share",

                    "Opportunity_Score":
                        "Opportunity Score",

                    "Recommended_Engagement_Strategy":
                        "Recommended Engagement"
                }
            )


            st.dataframe(
                top_hcps.style.format(
                    {
                        "Patient Volume":
                            "{:,.0f}",

                        "Target Share":
                            "{:.1%}",

                        "Category Growth":
                            "{:.1%}",

                        "New Patient Share":
                            "{:.1%}",

                        "Opportunity Score":
                            "{:.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            st.divider()


            # ====================================================
            # SELECTED HCP
            # ====================================================

            st.subheader(
                "Selected HCP"
            )


            hcp_options = (
                filtered_data
                .sort_values(
                    "Opportunity_Score",
                    ascending=False
                )["HCP_ID"]
                .tolist()
            )


            selected_hcp_id = st.selectbox(
                "Select HCP",
                hcp_options
            )


            selected_hcp = (
                hcp_data[
                    hcp_data["HCP_ID"]
                    == selected_hcp_id
                ]
                .iloc[0]
            )

            # PROFILE

            st.markdown(
                f"### {selected_hcp['HCP_ID']}"
            )

            st.caption(
                f"{selected_hcp['Specialty']} · {selected_hcp['Province']}"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.caption("BUSINESS SEGMENT")
                st.markdown(
                    f"**{selected_hcp['Business_Segment']}**"
                )

            with col2:
                st.caption("PRIORITY")
                st.markdown(
                    f"### ⭐ {selected_hcp['Priority_Tier']}"
                )

            st.write("")

            metric_cols = st.columns(4)

            with metric_cols[0]:
                with st.container(border=True):
                    st.caption("OPPORTUNITY SCORE")
                    st.markdown(
                        f"### {selected_hcp['Opportunity_Score']:.2f}"
                    )

            with metric_cols[1]:
                with st.container(border=True):
                    st.caption("PATIENT VOLUME PROXY")
                    st.markdown(
                        f"### {selected_hcp['Patient_Volume']:,.0f}"
                    )

            with metric_cols[2]:
                with st.container(border=True):
                    st.caption("TARGET SHARE")
                    st.markdown(
                        f"### {selected_hcp['Target_Share']:.1%}"
                    )

            with metric_cols[3]:
                with st.container(border=True):
                    st.caption("CATEGORY GROWTH")
                    st.markdown(
                        f"### {selected_hcp['Category_Growth']:.1%}"
                    )
            # ====================================================
            # SCORE EXPLANATION
            # ====================================================

            st.subheader(
                "Why is this HCP prioritized?"
            )
            st.caption(
                "The score reflects the relative contribution of four commercial opportunity signals."
            )


            contribution_data = pd.DataFrame(
                {
                    "Driver": [
                        "Patient Volume",
                        "Share Headroom",
                        "Growth",
                        "New Patient Share"
                    ],

                    "Contribution": [
                        selected_hcp[
                            "Volume_Contribution"
                        ],

                        selected_hcp[
                            "Headroom_Contribution"
                        ],

                        selected_hcp[
                            "Growth_Contribution"
                        ],

                        selected_hcp[
                            "New_Patient_Contribution"
                        ]
                    ]
                }
            )


            fig = px.bar(
                contribution_data,
                x="Contribution",
                y="Driver",
                orientation="h",
                template="plotly_dark",
                text="Contribution"
            )


            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )


            fig.update_layout(
                height=350,
                yaxis_title=None
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.caption(
                "Contributions are explanatory components of the "
                "synthetic Opportunity Score; they are not probabilities "
                "or predicted outcomes."
            )

            st.subheader("Recommended Engagement")

            engagement = selected_hcp["Recommended_Engagement_Strategy"]

            st.info(
                f"**Recommended approach:** {engagement}"
            )

            st.caption(
                "The recommended engagement approach is derived from the "
                "synthetic HCP segmentation and prioritization framework."
            )


# ============================================================
# AI COPILOT
# ============================================================

with ai_tab:

    st.subheader(
        "🤖 AI Commercial Copilot"
    )


    st.markdown(
        """
        <div class="ai-card">

        <h3>Analytics finds the opportunity. AI explains it.</h3>

        <div>
        The segmentation and scoring engine remains deterministic
        and explainable. Gemma 3 interprets those analytical outputs
        locally through Ollama.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if filtered_data.empty:

        st.warning(
            "No HCPs match the current filters."
        )

    else:

        # ====================================================
        # HCP SELECTOR
        # ====================================================

        hcp_options = (
            filtered_data
            .sort_values(
                "Opportunity_Score",
                ascending=False
            )["HCP_ID"]
            .tolist()
        )


        selected_ai_hcp_id = st.selectbox(
            "Select an HCP for AI interpretation",
            hcp_options,
            key="ai_hcp_selector"
        )
        st.caption(
            "HCPs are ordered by Commercial Opportunity within the current filtered population."
        )


        selected_ai_hcp = (
            hcp_data[
                hcp_data["HCP_ID"]
                == selected_ai_hcp_id
            ]
            .iloc[0]
        )



        # ====================================================
        # AI PROFILE
        # ====================================================

        st.markdown(
            f"### {selected_ai_hcp['HCP_ID']}"
        )

        st.caption(
            f"{selected_ai_hcp['Specialty']} · "
            f"{selected_ai_hcp['Province']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.caption("BUSINESS SEGMENT")
            st.markdown(
                f"**{selected_ai_hcp['Business_Segment']}**"
            )

        with col2:
            st.caption("PRIORITY")
            st.markdown(
                f"**{selected_ai_hcp['Priority_Tier']}**"
            )

        metric_cols = st.columns(2)

        with metric_cols[0]:
            st.metric(
                "Commercial Opportunity",
                f"{selected_ai_hcp['Opportunity_Score']:.2f}"
            )

        with metric_cols[1]:
            st.metric(
                "Target Share",
                f"{selected_ai_hcp['Target_Share']:.1%}"
            )

        # ====================================================
        # EVIDENCE BEFORE AI
        # ====================================================

        drivers, limitations = (
            build_hcp_evidence(
                selected_ai_hcp
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "**Positive signals identified by Python**"
            )


            for driver in drivers:

                st.markdown(
                    f"🟢 {driver}"
                )


        with col2:

            st.markdown(
                "**Limiting signals identified by Python**"
            )


            for limitation in limitations:

                st.markdown(
                    f"🔴 {limitation}"
                )


        st.write("")


        # ====================================================
        # GENERATE AI
        # ====================================================

        if st.button(
                "Generate Commercial Interpretation",
                type="primary",
                use_container_width=True
        ):
            with st.spinner("Generating commercial interpretation..."):

                ai_response, used_gemma = generate_ai_interpretation(
                    selected_ai_hcp
                )

                if used_gemma:
                    st.markdown("### AI Commercial Interpretation")
                    st.markdown(ai_response)
                    st.caption(
                        "Generated by Gemma 3 via Ollama. "
                        "The model interprets Python-generated analytical evidence."
                    )
                else:
                    st.markdown("### Evidence-Based Commercial Interpretation")
                    st.markdown(ai_response)
                    st.caption(
                        "Generated from Python-derived analytical evidence. "
                        "Gemma 3 was not available, so the deterministic fallback was used."
                    )

        # ====================================================
        # ASK COPILOT
        # ====================================================

        st.divider()


        st.subheader(
            "Portfolio Commercial Copilot"
        )


        st.caption(
            "Ask a portfolio-level question using the currently filtered HCP universe."

        )

        st.caption(
            "Example questions: "
            "Which HCPs should I prioritize for conversion? · "
            "Which specialties have the highest average opportunity? · "
            "Where are high-priority HCPs concentrated?"
        )

        question = st.text_input(
            "Commercial question",
            placeholder=(
                "Which HCPs should I prioritize for conversion?"
            )
        )

        if st.button("Ask Copilot", use_container_width=True):
            if not question.strip():
                st.warning("Enter a commercial question first.")
            else:
                with st.spinner("Generating commercial portfolio analysis..."):

                    answer, used_gemma = generate_copilot_answer(
                        question,
                        filtered_data,
                        selected_ai_hcp
                    )

                    if used_gemma:
                        st.markdown("### AI Copilot Response")
                        st.markdown(answer)
                        st.caption(
                            "Synthetic HCP Commercial Intelligence Demo · "
                            "Python analytical engine + optional local Gemma 3 via Ollama · "
                            "Evidence-based fallback when AI is unavailable · "
                            "Synthetic data for methodology demonstration only."
                        )
                    else:
                        st.markdown("### Evidence-Based Copilot Response")
                        st.markdown(answer)
                        st.caption(
                            "Generated from Python-derived analytical evidence. "
                            "Gemma 3 was not available, so the deterministic "
                            "fallback was used."
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Synthetic HCP Commercial Intelligence Demo · "
    "Python analytical engine + local Gemma 3 via Ollama · "
    "Synthetic data for methodology demonstration only."
)