import pandas as pd

# Load CSV
file_path = "hospitals_with_types.csv"
df = pd.read_csv(file_path)

# Normalize column names (avoid hidden space bugs)
df.columns = df.columns.str.strip()

# Required columns
NEW_TYPE_COL = "New_Types"
V3_COL = "Hospital_type_v3"

# Safety check
if NEW_TYPE_COL not in df.columns or V3_COL not in df.columns:
    raise ValueError(f"Required columns missing. Found columns: {df.columns.tolist()}")

# STEP 1: Create new column (copy original exactly)
df["New_Types_v2"] = df[NEW_TYPE_COL].astype(str).str.strip()

# STEP 2: Clean fallback column
df[V3_COL] = df[V3_COL].astype(str).str.strip().str.upper()

# STEP 3: Mapping
fallback_map = {
    "TERTIARY": "Advanced Multispecialty",
    "ADVANCED": "Advanced Multispecialty",
    "CORE": "Standard Secondary General",
    "SMALL": "Small Day-Care Clinics"
}

# STEP 4: Apply logic ONLY on New_Types_v2
def apply_fallback(row):
    if row["New_Types_v2"] != "Not Matched":
        return row["New_Types_v2"]
    
    return fallback_map.get(row[V3_COL], "Standard Secondary General")

df["New_Types_v2"] = df.apply(apply_fallback, axis=1)

# STEP 5: (Optional Debug Column)
df["Used_Fallback"] = df[NEW_TYPE_COL].astype(str).str.strip() == "Not Matched"

# STEP 6: Save file
output_path = "hospitals_with_types_v2.csv"
df.to_csv(output_path, index=False)

print("Done. New column 'New_Types_v2' created and file saved at:", output_path)