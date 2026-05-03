"""
Appends reviews_lat and reviews_lng from Hospital_data.csv
into hospitals_with_types.csv (same row order, row 2 onwards).
"""
import pandas as pd

hwt = pd.read_csv("hospitals_with_types.csv")
hd  = pd.read_csv("Hospital_data.csv")

print(f"hospitals_with_types.csv rows : {len(hwt)}")
print(f"Hospital_data.csv rows        : {len(hd)}")

n = min(len(hwt), len(hd))

# Sanity check
mismatches = sum(
    str(hwt.iloc[i]["name"]).strip() != str(hd.iloc[i]["name"]).strip()
    for i in range(n)
)
print(f"Name mismatches: {mismatches}")

hwt["reviews_lat"] = hd["reviews_lat"].values[:n]
hwt["reviews_lng"] = hd["reviews_lng"].values[:n]

hwt.to_csv("hospitals_with_types.csv", index=False)

print(f"\nDone! hospitals_with_types.csv now has {len(hwt.columns)} columns.")
print(f"  reviews_lat non-null  : {hwt['reviews_lat'].notna().sum()} / {n}")
print(f"  reviews_lng non-null  : {hwt['reviews_lng'].notna().sum()} / {n}")
print("\nSample:")
print(hwt[["name","lat","lng","reviews_lat","reviews_lng"]].head(4).to_string(index=False))
