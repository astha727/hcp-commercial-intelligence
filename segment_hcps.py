import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


hcp_data = pd.read_csv("Data/synthetic_hcp_data.csv")

segmentation_variables = [
    "Patient_Volume",
    "Target_Share",
    "Category_Growth",
    "New_Patient_Share"
]

#business segmentation
hcp_data["Business_Segment"] = "Core / Maintenance"

#high conversion
high_potential_conversion = (
    (hcp_data["Patient_Volume"] > hcp_data["Patient_Volume"].quantile(0.75))
    & (hcp_data["Target_Share"] < hcp_data["Target_Share"].quantile(0.25))
    & (hcp_data["Category_Growth"] > 0)
)

hcp_data.loc[
    high_potential_conversion,
    "Business_Segment"
] = "High-Potential Conversion"

#second rule - stable
high_value_established = (
    (hcp_data["Patient_Volume"] > hcp_data["Patient_Volume"].quantile(0.75))
    & (hcp_data["Target_Share"] > hcp_data["Target_Share"].quantile(0.75))
)

hcp_data.loc[
    high_value_established,
    "Business_Segment"
] = "High-Value Established"

#3rd emerging opp
emerging_opportunity = (
    (hcp_data["Patient_Volume"] <= hcp_data["Patient_Volume"].quantile(0.75))
    & (hcp_data["Target_Share"] < hcp_data["Target_Share"].quantile(0.25))
    & (hcp_data["New_Patient_Share"] > hcp_data["New_Patient_Share"].quantile(0.75))
    & (hcp_data["Category_Growth"] > 0)
)

hcp_data.loc[
    emerging_opportunity,
    "Business_Segment"
] = "Emerging Opportunity"


#normalization before k means
segmentation_variables = [
    "Patient_Volume",
    "Target_Share",
    "Category_Growth",
    "New_Patient_Share"
]

X = hcp_data[segmentation_variables].copy()

#print("\nClustering Feature Matrix:")
#print(X.shape)
#print(X.head())


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#print("\nStandardized Feature Matrix:")
#print(X_scaled.shape)
#print(X_scaled[:5])

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    #print(f"K={k}: Silhouette Score={score:.3f}")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

hcp_data["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

cluster_profile = (
    hcp_data
    .groupby("KMeans_Cluster")[segmentation_variables]
    .mean()
    .round(3)
)

#comparison
comparison = pd.crosstab(
    hcp_data["Business_Segment"],
    hcp_data["KMeans_Cluster"]
)

#print("\nBusiness Segment vs K-Means Cluster:")
#print(comparison)

comparison_pct = pd.crosstab(
    hcp_data["Business_Segment"],
    hcp_data["KMeans_Cluster"],
    normalize="index"
).round(3)


#standardize variables before scoring
scoring_variables = [
    "Patient_Volume",
    "Target_Share",
    "Category_Growth",
    "New_Patient_Share"
]

scaler_score = StandardScaler()

score_scaled = scaler_score.fit_transform(
    hcp_data[scoring_variables]
)

score_scaled = pd.DataFrame(
    score_scaled,
    columns=scoring_variables,
    index=hcp_data.index
)

#print("\nScaled Scoring Variables:")
#print(score_scaled.head())

score_scaled["Target_Share_Headroom"] = (
    -score_scaled["Target_Share"]
)

score_scaled["Growth_Opportunity"] = (
    score_scaled["Category_Growth"].clip(lower=0)
)

#print("\nOpportunity Score Components:")
#print(
    #score_scaled[
        #[
            #"Patient_Volume",
            #"Target_Share_Headroom",
            #"Growth_Opportunity",
            #"New_Patient_Share"
        #]
    #].head()
#)

score_scaled["Opportunity_Score"] = (
    0.40 * score_scaled["Patient_Volume"]
    + 0.30 * score_scaled["Target_Share_Headroom"]
    + 0.15 * score_scaled["Growth_Opportunity"]
    + 0.15 * score_scaled["New_Patient_Share"]
)

hcp_data["Opportunity_Score"] = (
    score_scaled["Opportunity_Score"]
)



top_10 = (
    hcp_data[
        [
            "HCP_ID",
            "Province",
            "Specialty",
            "Business_Segment",
            "Patient_Volume",
            "Target_Share",
            "Category_Growth",
            "New_Patient_Share",
            "Opportunity_Score"
        ]
    ]
    .sort_values("Opportunity_Score", ascending=False)
    .head(10)
)

#print("\nTop 10 HCP Opportunities:")
#print(top_10.to_string(index=False))



#priority tiers
high_priority_threshold = hcp_data["Opportunity_Score"].quantile(0.80)
medium_priority_threshold = hcp_data["Opportunity_Score"].quantile(0.50)

hcp_data["Priority_Tier"] = "Low Priority"

hcp_data.loc[
    hcp_data["Opportunity_Score"] >= medium_priority_threshold,
    "Priority_Tier"
] = "Medium Priority"

hcp_data.loc[
    hcp_data["Opportunity_Score"] >= high_priority_threshold,
    "Priority_Tier"
] = "High Priority"

#print("\nPriority Tier Distribution:")
#print(
    #hcp_data["Priority_Tier"]
    #.value_counts()
#)

#print("\nPriority Tier Percentages:")
#print(
    #hcp_data["Priority_Tier"]
    #.value_counts(normalize=True)
    #.mul(100)
    #.round(1)
#)

high_priority = hcp_data[
    hcp_data["Priority_Tier"] == "High Priority"
].copy()

#print("\nHigh-Priority HCPs by Province:")
#print(
    #high_priority["Province"]
    #.value_counts()
#)


#print("\nHigh-Priority HCPs by Specialty:")
#print(
    #high_priority["Specialty"]
    #.value_counts()
#)

specialty_profile = (
    high_priority
    .groupby("Specialty")
    .agg(
        HCP_Count=("HCP_ID", "count"),
        Avg_Patient_Volume=("Patient_Volume", "mean"),
        Avg_Target_Share=("Target_Share", "mean"),
        Avg_Category_Growth=("Category_Growth", "mean"),
        Avg_New_Patient_Share=("New_Patient_Share", "mean"),
        Avg_Opportunity_Score=("Opportunity_Score", "mean")
    )
    .sort_values("Avg_Opportunity_Score", ascending=False)
)

#print("\nHigh-Priority Specialty Profile:")
#print(
    #specialty_profile.round(3).to_string()
#)

#phase 4 - defining engagement startegy
engagement_profile = (
    high_priority[
        [
            "Business_Segment",
            "Field_Engagement",
            "Digital_Engagement"
        ]
    ]
    .groupby("Business_Segment")
    .agg(
        HCP_Count=("Field_Engagement", "count"),
        Avg_Field_Engagement=("Field_Engagement", "mean"),
        Avg_Digital_Engagement=("Digital_Engagement", "mean")
    )
    .sort_values("HCP_Count", ascending=False)
)

#print("\nHigh-Priority Engagement Profile:")
#print(
    #engagement_profile.round(1).to_string()
#)

hcp_data["Field_Engagement_Level"] = pd.cut(
    hcp_data["Field_Engagement"],
    bins=[-1, 32, 66, 100],
    labels=["Low", "Medium", "High"]
)

hcp_data["Digital_Engagement_Level"] = pd.cut(
    hcp_data["Digital_Engagement"],
    bins=[-1, 32, 66, 100],
    labels=["Low", "Medium", "High"]
)

#activating engagement

hcp_data["Recommended_Engagement_Strategy"] = "Maintain efficiently"

# High-Potential Conversion
hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "High-Potential Conversion")
        & (hcp_data["Field_Engagement_Level"] == "Low")
    ),
    "Recommended_Engagement_Strategy"
] = "Increase field engagement"

hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "High-Potential Conversion")
        & (hcp_data["Field_Engagement_Level"] == "Medium")
    ),
    "Recommended_Engagement_Strategy"
] = "Build field relationship"

hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "High-Potential Conversion")
        & (hcp_data["Field_Engagement_Level"] == "High")
    ),
    "Recommended_Engagement_Strategy"
] = "Deepen field relationship"


# Emerging Opportunity
hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "Emerging Opportunity")
        & (hcp_data["Digital_Engagement_Level"] == "High")
    ),
    "Recommended_Engagement_Strategy"
] = "Digital-first nurture"

hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "Emerging Opportunity")
        & (hcp_data["Digital_Engagement_Level"] != "High")
    ),
    "Recommended_Engagement_Strategy"
] = "Develop engagement"


# High-Value Established
hcp_data.loc[
    hcp_data["Business_Segment"] == "High-Value Established",
    "Recommended_Engagement_Strategy"
] = "Retain and deepen"


# Core / Maintenance with high opportunity
hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "Core / Maintenance")
        & (hcp_data["Priority_Tier"] == "High Priority")
        & (hcp_data["Digital_Engagement_Level"] == "High")
    ),
    "Recommended_Engagement_Strategy"
] = "Digital-led opportunity development"

hcp_data.loc[
    (
        (hcp_data["Business_Segment"] == "Core / Maintenance")
        & (hcp_data["Priority_Tier"] == "High Priority")
        & (hcp_data["Field_Engagement_Level"] == "High")
    ),
    "Recommended_Engagement_Strategy"
] = "Field-led opportunity development"

#print("\nRecommended Engagement Strategy:")
#print(
    #hcp_data["Recommended_Engagement_Strategy"]
    #.value_counts()
#)


# FINAL OUTPUT DATASET


final_columns = [
    # HCP identifiers / context
    "HCP_ID",
    "Province",
    "Specialty",

    # Market context
    "Market_Intensity",

    # Commercial behavior
    "Patient_Volume",
    "Category_TRx",
    "Target_TRx",
    "Target_Share",
    "Competitor_Share",
    "Category_Growth",
    "New_Patient_Share",

    # Engagement
    "Field_Engagement",
    "Digital_Engagement",
    "Field_Engagement_Level",
    "Digital_Engagement_Level",

    # Segmentation
    "Business_Segment",
    "KMeans_Cluster",

    # Prioritization
    "Opportunity_Score",
    "Priority_Tier",

    # Recommended action
    "Recommended_Engagement_Strategy"
]

final_hcp_data = hcp_data[final_columns].copy()

# Save final dataset
final_hcp_data.to_csv(
    "Data/synthetic_hcp_segmentation_output.csv",
    index=False
)

print("\nFinal Dataset Shape:")
print(final_hcp_data.shape)

print("\nFinal Dataset Columns:")
print(final_hcp_data.columns.tolist())

print("\nFinal Dataset Preview:")
print(final_hcp_data.head())

print("\nFinal Dataset Saved:")
print("Data/synthetic_hcp_segmentation_output.csv")