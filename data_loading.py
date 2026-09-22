import numpy as np
import pandas as pd

df = pd.DataFrame({"HCP_ID": [f"HCP_{i:04d}" for i in range(1, 11)]})


specialties = ["Cardiology", "Neurology","Oncology","Endocrinology","Internal Medicine"]

df['Specialty'] = np.random.choice(specialties, size=df.shape[0], replace=True)

geographies = ["Ontario","Alberta","Prince Edward Island","British Columbia","Quebec"]
df['Geography'] = np.random.choice(geographies, size=df.shape[0], replace=True)
print(df.head(10))