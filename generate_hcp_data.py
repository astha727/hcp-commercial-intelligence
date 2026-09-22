import numpy as np
import pandas as pd



market_context = pd.read_csv("Data/market_context.csv")
market_context = market_context[
    market_context["Physician_Count"] > 0
].copy()

total_physician = market_context["Physician_Count"].sum()

total_synthetic_HCP = 2000

market_context["Physician_Share"] = (
    market_context["Physician_Count"] / total_physician
)

market_context["Synthetic_HCP_count"] = (
    market_context["Physician_Share"] * total_synthetic_HCP
).round().astype(int)

market_context["Exact_HCP_count"] = (
    market_context["Physician_Share"] * total_synthetic_HCP
)

market_context["Synthetic_HCP_count"] = (
    market_context["Exact_HCP_count"].astype(int)
)

remaining_hcps = (
    total_synthetic_HCP - market_context["Synthetic_HCP_count"].sum()
)

market_context["Reminder"] = (
market_context["Exact_HCP_count"]- market_context["Synthetic_HCP_count"]
)

#Give remaining HCPs to cells with largest reminders
largest_reminders = (
    market_context["Reminder"].nlargest(remaining_hcps).index
)

market_context.loc[largest_reminders, "Synthetic_HCP_count"] += 1




#market activity index
benchmark_services = market_context["Services_per_Physician"].median()

market_context["Market_Activity_Index"] = (
    market_context["Services_per_Physician"] / benchmark_services
)

#speciality activity index
specialty_median = (
    market_context.groupby("Specialty")["Services_per_Physician"].transform("median")
)

market_context["Specialty_Activity_Index"] = (
    market_context["Services_per_Physician"] / specialty_median
)



#relative specialty intensity

specialty_medians = (
    market_context
    .groupby("Specialty")["Services_per_Physician"]
    .median()
)

reference_specialty = "Pediatrics"

reference_value = specialty_medians[reference_specialty]

specialty_intensity = (
    specialty_medians / reference_value
)

#disease population
disease_population = pd.DataFrame({
    "Province": [
        "Newfoundland and Labrador",
        "Prince Edward Island",
        "Nova Scotia",
        "New Brunswick",
        "Quebec",
        "Ontario",
        "Manitoba",
        "Saskatchewan",
        "Alberta",
        "British Columbia",
        "Yukon",
        "Northwest Territories",
        "Nunavut"
    ],

    "High_Blood_Pressure": [
        125400,
        30600,
        210700,
        181900,
        1326900,
        2625000,
        235100,
        193300,
        689700,
        841000,
        5400,
        5800,
        4500
    ],

    "Mood_Disorder": [
        67600,
        19500,
        150600,
        105600,
        676000,
        1675300,
        157200,
        124500,
        541000,
        644200,
        4600,
        4700,
        3800
    ],

    "Arthritis": [
        139900,
        32900,
        246300,
        179400,
        1273500,
        2676100,
        229800,
        192600,
        776400,
        890600,
        7700,
        5700,
        4000
    ],

    "Diabetes": [
        51900,
        13300,
        84300,
        70000,
        609200,
        1095800,
        93800,
        86200,
        277600,
        308600,
        2200,
        2700,
        np.nan
    ]
})

market_context = market_context.merge(
    disease_population,
    on="Province",
    how="left"
)


#market opp index
market_context["Disease_Signal"] = 0.0

market_context.loc[
    market_context["Specialty"] == "Cardiology",
    "Disease_Signal"
] = market_context["High_Blood_Pressure"]

market_context.loc[
    market_context["Specialty"] == "Psychiatry",
    "Disease_Signal"
] = market_context["Mood_Disorder"]

market_context.loc[
    market_context["Specialty"] == "Orthopedic Surgery",
    "Disease_Signal"
] = market_context["Arthritis"]

market_context.loc[
    market_context["Specialty"] == "Internal Medicine",
    "Disease_Signal"
] = market_context["Diabetes"]

#normalized disease signal
disease_benchmark = (
    market_context.loc[
        market_context["Disease_Signal"] > 0,
        "Disease_Signal"
    ].median()
)

market_context["Disease_Opportunity_Index"] = (
    market_context["Disease_Signal"]
    / disease_benchmark
)

#market intensity
market_context["Market_Intensity"] = (
    np.sqrt(market_context["Disease_Opportunity_Index"])
    * market_context["Specialty_Activity_Index"]
)

#market base index
population_benchmark = market_context["Population"].median()

market_context["Population_Index"] = (
    market_context["Population"] / population_benchmark
)

market_context["Market_Base_Index"] = np.where(
    market_context["Disease_Opportunity_Index"] > 0,
    market_context["Disease_Opportunity_Index"],
    market_context["Population_Index"]
)

market_context["Activity_Index_Final"] = (
    market_context["Specialty_Activity_Index"].fillna(1.0)
)

market_context["Market_Intensity"] = (
    np.sqrt(market_context["Market_Base_Index"])
    * market_context["Activity_Index_Final"]
)

#creating patient volume
hcp_rows = []

hcp_id = 1

for _, row in market_context.iterrows():

    for _ in range(row["Synthetic_HCP_count"]):

        hcp_rows.append({
            "HCP_ID": f"HCP_{hcp_id:04d}",
            "Province": row["Province"],
            "Specialty": row["Specialty"],
            "Market_Intensity": row["Market_Intensity"]
        })

        hcp_id += 1

hcp_data = pd.DataFrame(hcp_rows)

np.random.seed(42)

base_patient_volume = 400

hcp_data["Patient_Volume"] = (
    base_patient_volume
    * np.sqrt(hcp_data["Market_Intensity"])
    * np.random.lognormal(
        mean=0,
        sigma=0.35,
        size=len(hcp_data)
    )
).round().astype(int)

#category prescribing
category_penetration = np.random.uniform(
    0.40,
    0.80,
    size=len(hcp_data)
)

hcp_data["Category_TRx"] = (
    hcp_data["Patient_Volume"]
    * category_penetration
).round().astype(int)

#target product prescribing
target_share = np.random.uniform(
    0.05,
    0.60,
    size=len(hcp_data)
)

hcp_data["Target_TRx"] = (
    hcp_data["Category_TRx"]
    * target_share
).round().astype(int)

hcp_data["Target_Share"] = (
    hcp_data["Target_TRx"]
    / hcp_data["Category_TRx"]
).fillna(0)

#competitor share
hcp_data["Competitor_Share"] = (
    1 - hcp_data["Target_Share"]
)

#category growth
hcp_data["Category_Growth"] = np.random.normal(
    loc=0.05,
    scale=0.12,
    size=len(hcp_data)
)

#new patient share
hcp_data["New_Patient_Share"] = np.random.uniform(
    0.05,
    0.40,
    size=len(hcp_data)
)

#hcp engagement
hcp_data["Field_Engagement"] = np.random.randint(
    0,
    101,
    size=len(hcp_data)
)

hcp_data["Digital_Engagement"] = np.random.randint(
    0,
    101,
    size=len(hcp_data)
)

hcp_data.to_csv(
    "Data/synthetic_hcp_data.csv",
    index=False
)