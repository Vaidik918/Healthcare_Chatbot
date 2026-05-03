"""
Appends reviews_phone and reviews_website from Hospital_data.csv
into hospitals_with_types.csv using positional row alignment.
Both files have the same rows in the same order.
"""
import pandas as pd

hwt = pd.read_csv("hospitals_with_types.csv")
hd  = pd.read_csv("Hospital_data.csv")

print(f"hospitals_with_types.csv rows : {len(hwt)}")
print(f"Hospital_data.csv rows        : {len(hd)}")

if len(hwt) != len(hd):
    print("WARNING: row counts differ — using positional alignment on the shorter length")

n = min(len(hwt), len(hd))

# Sanity: spot-check names at a few positions
print("\n=== Name alignment spot-check ===")
for i in [0, 1, 2, n-3, n-2, n-1]:
    a = str(hwt.iloc[i]["name"]).strip()
    b = str(hd.iloc[i]["name"]).strip()
    status = "OK" if a == b else "MISMATCH"
    print(f"  row {i+1}: [{status}] hwt={a[:45]!r}  hd={b[:45]!r}")

mismatches = sum(
    str(hwt.iloc[i]["name"]).strip() != str(hd.iloc[i]["name"]).strip()
    for i in range(n)
)
print(f"\nTotal name mismatches across {n} rows: {mismatches}")

# Pull the two columns and attach positionally
hwt["reviews_phone"]   = hd["reviews_phone"].values[:n]
hwt["reviews_website"] = hd["reviews_website"].values[:n]

# Save (overwrites hospitals_with_types.csv in place)
hwt.to_csv("hospitals_with_types.csv", index=False)

print(f"\nDone! hospitals_with_types.csv now has {len(hwt.columns)} columns.")
print(f"  reviews_phone non-null  : {hwt['reviews_phone'].notna().sum()} / {n}")
print(f"  reviews_website non-null: {hwt['reviews_website'].notna().sum()} / {n}")
print("\nSample of appended data:")
print(hwt[["name", "reviews_phone", "reviews_website"]].head(5).to_string(index=False))
