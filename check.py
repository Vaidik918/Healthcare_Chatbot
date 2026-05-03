# Utility to clean NaN/inf values from dicts/lists for JSON serialization
import math
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
"""
=============================================================================
HEALTHCARE COST ESTIMATION ENGINE v4.0 — POONAWALLA FINCORP
=============================================================================
Source: RATE_CARDS.xlsx (Procedure Cost Cards + Clinical Pathways)

KEY FORMULA:
  Final Cost = (BASE × Room_Adj × Tier_Mult) + (ICU × ICU_Mult) 
             + (Med × Med_Mult) + (IMPLANT × Cap_Mult) 
             + (TECH × Cap_Mult) + Room_GST + Contingency 15%

BASE Column  = Variable (surgery + room + dr visits + misc already included)
IMPLANT Col  = Fixed (NPPA capped — no tier multiplier)
TECH Column  = Fixed (device cost — no tier multiplier)
TOTAL Column = BASE + IMPLANT + TECH (reference, no GST/contingency)

GST Rules (India 2024):
  - Surgery/Hospital service: NO GST (healthcare exempt)
  - Medicines: NO GST in bill (already in MRP)
  - Room rent >₹5,000/day: 5% GST only on EXCESS above ₹5,000
  - Diagnostics: 0% GST
=============================================================================
"""

import os

import pandas as pd
import math
import json
from math import radians, sin, cos, sqrt, atan2



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # Data files live in the same directory as check.py

RATE_CARDS_PATH = os.path.join(DATA_DIR, "RATE_CARDS.xlsx")
HOSPITALS_PATH  = os.path.join(DATA_DIR, "hospitals_with_types.csv")

# ─── PATHS ────────────────────────────────────────────────────────────────────
#RATE_CARDS_PATH = r"D:\get_data\RATE_CARDS.xlsx"
#HOSPITALS_PATH  = r"D:\get_data\hospitals_with_types.csv"

_proc_df = None
_path_df = None
_hosp_df = None
_specialty_scores = None

SPECIALTY_SCORES_PATH = os.path.join(DATA_DIR, "specialty_scores.json")

def _load_proc():
    global _proc_df
    if _proc_df is None:
        _proc_df = pd.read_excel(RATE_CARDS_PATH, sheet_name="Procedure Cost Cards")
        _proc_df.columns = _proc_df.columns.str.strip()
    return _proc_df

def _load_path():
    global _path_df
    if _path_df is None:
        _path_df = pd.read_excel(RATE_CARDS_PATH, sheet_name="Clinical Pathways")
        _path_df.columns = _path_df.columns.str.strip()
    return _path_df

def _load_hosp():
    global _hosp_df
    if _hosp_df is None:
        _hosp_df = pd.read_csv(HOSPITALS_PATH)
        _hosp_df.columns = _hosp_df.columns.str.strip()
    return _hosp_df

def _load_specialty_scores():
    """Load review-derived specialty sentiment scores (one-time, cached in memory)."""
    global _specialty_scores
    if _specialty_scores is None:
        if os.path.exists(SPECIALTY_SCORES_PATH):
            with open(SPECIALTY_SCORES_PATH, encoding="utf-8") as f:
                _specialty_scores = json.load(f)
        else:
            _specialty_scores = {}
    return _specialty_scores

def get_hospital_specialty_scores(name: str) -> dict:
    """
    Look up specialty sentiment scores for a hospital by name.
    Falls back to fuzzy matching if exact key not found.
    Returns the full specialty dict or {} if not found.
    """
    from difflib import get_close_matches
    scores = _load_specialty_scores()
    key = name.strip().lower()
    if key in scores:
        return scores[key]
    # Fuzzy fallback — handles minor name differences between CSV and JSONL
    close = get_close_matches(key, scores.keys(), n=1, cutoff=0.75)
    return scores[close[0]] if close else {}

_MISSING = {"", "NA", "N/A", "nan", "None", "none"}

def _best(row, primary_col: str, fallback_col: str) -> str:
    """Return primary column value if non-empty, else fallback column value."""
    val = str(row.get(primary_col) or "").strip()
    if val and val not in _MISSING:
        return val
    return str(row.get(fallback_col, "NA") or "NA").strip()

# =============================================================================
# SECTION 1 — TIER MULTIPLIERS (Applied to BASE only)
# =============================================================================

CITY_TIER = {
    "delhi": "T1", "mumbai": "T1", "bangalore": "T1",
    "jaipur": "T2", "lucknow": "T2", "nagpur": "T2",
    "bhopal": "T2", "dehradun": "T3", "indore": "T3",
}

TIER_VAR_MULT = {
    ("Tertiary Corporate",        "T1"): (3.0, 3.8),
    ("Tertiary Corporate",        "T2"): (2.5, 3.4),
    ("Tertiary Corporate",        "T3"): (2.2, 3.0),
    ("Boutique Super-Specialty",  "T1"): (2.5, 3.5),
    ("Boutique Super-Specialty",  "T2"): (2.0, 3.0),
    ("Boutique Super-Specialty",  "T3"): (1.8, 2.6),
    ("Advanced Multispecialty",   "T1"): (2.0, 3.0),
    ("Advanced Multispecialty",   "T2"): (1.8, 2.8),
    ("Advanced Multispecialty",   "T3"): (1.8, 2.6),
    ("Standard Secondary General","T1"): (1.3, 2.0),
    ("Standard Secondary General","T2"): (1.0, 1.7),
    ("Standard Secondary General","T3"): (0.9, 1.5),
    ("Small Day-Care Clinics",    "T1"): (1.0, 1.4),
    ("Small Day-Care Clinics",    "T2"): (0.9, 1.3),
    ("Small Day-Care Clinics",    "T3"): (0.8, 1.2),
    ("Government",                "T1"): (0.4, 0.8),
    ("Government",                "T2"): (0.4, 0.7),
    ("Government",                "T3"): (0.3, 0.6),
}

# ICU semi-variable (base ₹5,400/day from card)
ICU_TIER_MULT = {
    "Tertiary Corporate":         3.5,   # ~₹18,900/day
    "Boutique Super-Specialty":   2.0,
    "Advanced Multispecialty":    2.2,   # ~₹11,880/day (Jupiter confirmed)
    "Standard Secondary General": 1.5,
    "Small Day-Care Clinics":     1.0,
    "Government":                 0.6,
}

# Ventilator semi-variable
VENT_TIER_MULT = {
    "Tertiary Corporate":         2.5,
    "Boutique Super-Specialty":   2.0,
    "Advanced Multispecialty":    1.5,
    "Standard Secondary General": 1.0,
    "Small Day-Care Clinics":     0.7,
    "Government":                 0.5,
}

# Medical consumables (drugs/drapes) — separate from implants
MED_TIER_MULT = {
    "Tertiary Corporate":         2.5,
    "Boutique Super-Specialty":   2.0,
    "Advanced Multispecialty":    1.8,
    "Standard Secondary General": 1.0,
    "Small Day-Care Clinics":     0.8,
    "Government":                 0.6,
}

# Fixed bucket — implants + tech (NPPA capped, no tier markup)
FIXED_CAP = {
    "Tertiary Corporate":         1.0,
    "Boutique Super-Specialty":   1.0,
    "Advanced Multispecialty":    1.0,
    "Standard Secondary General": 1.0,
    "Small Day-Care Clinics":     0.85,
    "Government":                 0.70,
}

# Room type adjustment on BASE (PRIVATE = 1.3x surgery component)
# Room charges from card scale by room type
ROOM_TYPE_MULT_ON_BASE = {
    "GENERAL":     1.0,
    "SEMI_PRIVATE": 1.15,
    "PRIVATE":     1.30,
}

# =============================================================================
# SECTION 2 — COMORBIDITIES
# =============================================================================

COMORBIDITY = {
    "diabetes": {
        "base_mult": 1.12, "extra_los": 2, "icu_risk": 0.10, "med_addon": 5000,
        "note": "Wound healing delay, insulin management, infection risk"
    },
    "hypertension": {
        "base_mult": 1.06, "extra_los": 1, "icu_risk": 0.05, "med_addon": 2000,
        "note": "Periop BP monitoring, antihypertensive adjustment"
    },
    "cardiac_history": {
        "base_mult": 1.25, "extra_los": 3, "icu_risk": 0.25, "med_addon": 12000,
        "note": "Cardiology clearance, ICU likely, cardiac monitoring mandatory"
    },
    "ckd": {
        "base_mult": 1.18, "extra_los": 2, "icu_risk": 0.15, "med_addon": 10000,
        "note": "Drug dose adjustment, nephrology consult, fluid restriction"
    },
    "obesity": {
        "base_mult": 1.10, "extra_los": 1, "icu_risk": 0.08, "med_addon": 4000,
        "note": "Longer OT time, higher anaesthesia risk, DVT prophylaxis"
    },
    "copd": {
        "base_mult": 1.15, "extra_los": 2, "icu_risk": 0.12, "med_addon": 6000,
        "note": "Pulmonology clearance, ventilation risk, bronchodilators"
    },
}

# =============================================================================
# SECTION 2B — AGE SEVERITY ADJUSTMENTS
# =============================================================================
# Source: ASA Physical Status Classification + NHA PMJAY Claims Data 2019-2023
# Age is NOT a comorbidity but it modifies perioperative risk independently.
# Multipliers apply additively with comorbidity multipliers (same pattern).
#
# base_mult  : applied to BASE component (anaesthesia complexity, pre-op workup)
# extra_los  : additional room/nursing days beyond card baseline
# icu_risk   : probability that ICU standby becomes ICU stay (adds icu_days)
# med_addon  : flat additional medication/consumable cost (INR)

AGE_SEVERITY = {
    "paediatric": {
        "range": (0, 17),
        "base_mult": 1.20,
        "extra_los":  1,
        "icu_risk":   0.08,
        "med_addon":  4000,
        "note": "Paediatric anaesthesia agents, weight-based drug dosing, 1:1 specialized nursing post-op"
    },
    "adult": {
        "range": (18, 60),
        "base_mult": 1.0,
        "extra_los":  0,
        "icu_risk":   0.0,
        "med_addon":  0,
        "note": "Standard adult surgical protocol — baseline cost"
    },
    "senior": {
        "range": (61, 75),
        "base_mult": 1.15,
        "extra_los":  1,
        "icu_risk":   0.10,
        "med_addon":  5000,
        "note": "ASA PS Class III typical: extended pre-op cardiac/pulmonary workup, +1 LOS for monitored recovery"
    },
    "elderly": {
        "range": (76, 150),
        "base_mult": 1.25,
        "extra_los":  2,
        "icu_risk":   0.20,
        "med_addon":  8000,
        "note": "ASA PS Class III-IV: ICU standby mandatory, post-op delirium monitoring, 2.1x readmission risk (Lancet India 2021)"
    },
}

# Gender-specific cost adjustments — deliberately minimal and evidence-based only.
# Only CABG has a validated LOS difference by sex (AHA/ACC 2021 guidelines).
# All other procedures: gender used for validation only, not cost adjustment.
GENDER_PROCEDURE_RULES = {
    "male_only":   ["TURP - Prostate", "Kidney Stone PCNL"],  # flag if female selects
    "female_only": [],   # hysterectomy etc. not in current rate cards
    "cabg_female_note": (
        "Women undergoing CABG have smaller coronary vessel diameter — "
        "longer perfusion time and higher perioperative monitoring required (AHA/ACC 2021). "
        "Expect LOS at upper end of range."
    ),
}

def _get_age_band(age: int) -> dict:
    """Return the AGE_SEVERITY entry for a given age. Defaults to 'adult' if None."""
    if age is None:
        return AGE_SEVERITY["adult"]
    for band, data in AGE_SEVERITY.items():
        lo, hi = data["range"]
        if lo <= int(age) <= hi:
            return {"band": band, **data}
    return {"band": "adult", **AGE_SEVERITY["adult"]}

# =============================================================================
# SECTION 3 — PROCEDURE CARD LOOKUP
# =============================================================================

def _sf(v, default=0.0):
    try:
        f = float(v)
        return f if not math.isnan(f) else default
    except:
        return default

def get_all_variants(procedure: str) -> list:
    """All variants for a procedure."""
    df = _load_proc()
    rows = df[df["Procedure"].str.strip() == procedure]
    if rows.empty:
        rows = df[df["Procedure"].str.contains(procedure, case=False, na=False)]
    result = []
    for _, r in rows.iterrows():
        result.append({
            "sub_procedure":  str(r.get("Sub-Procedure","")).strip(),
            "variant":        str(r.get("Clinical Variant","")).strip(),
            "tech_type":      str(r.get("Tech Type","")).strip(),
            "los":            int(_sf(r.get("LOS Days"))),
            "icu_days":       int(_sf(r.get("ICU _Days"))),
            "base":           _sf(r.get("BASE")),
            "implant":        _sf(r.get("IMPLANT")),
            "tech":           _sf(r.get("TECH")),
            "total":          _sf(r.get("TOTAL")),
            "room_gen":       _sf(r.get("Room Charges General (INR)")),
            "room_semi":      _sf(r.get("Room Charges Semi-Pvt (INR)")),
            "room_pvt":       _sf(r.get("Room Charges Pvt (INR)")),
            "icu_charge":     _sf(r.get("ICU Charges (INR)")),
            "vent_charge":    _sf(r.get("Ventilator Charges (INR)")),
            "clinical_notes": str(r.get("Clinical Notes","")).strip(),
        })
    return result

def get_card(procedure: str, sub_procedure: str = None,
             variant_hint: str = None) -> dict:
    """Single procedure card lookup."""
    df = _load_proc()
    rows = df[df["Procedure"].str.strip() == procedure]
    if rows.empty:
        rows = df[df["Procedure"].str.contains(procedure, case=False, na=False)]
    if rows.empty:
        return None

    if sub_procedure:
        m = rows[rows["Sub-Procedure"].str.strip() == sub_procedure]
        if not m.empty:
            rows = m

    if variant_hint:
        m = rows[rows["Clinical Variant"].str.contains(
            variant_hint, case=False, na=False)]
        if not m.empty:
            rows = m

    r = rows.iloc[0]
    return {
        "procedure":      str(r.get("Procedure","")).strip(),
        "sub_procedure":  str(r.get("Sub-Procedure","")).strip(),
        "variant":        str(r.get("Clinical Variant","")).strip(),
        "tech_type":      str(r.get("Tech Type","")).strip(),
        "los":            int(_sf(r.get("LOS Days"))),
        "icu_days":       int(_sf(r.get("ICU _Days"))),
        # Key cost columns
        "base":           _sf(r.get("BASE")),      # VARIABLE — multiplier applies
        "implant":        _sf(r.get("IMPLANT")),   # FIXED — cap multiplier only
        "tech":           _sf(r.get("TECH")),      # FIXED — cap multiplier only
        "total":          _sf(r.get("TOTAL")),     # reference only
        # Room charges from card (base at Standard level)
        "room_gen":       _sf(r.get("Room Charges General (INR)")),
        "room_semi":      _sf(r.get("Room Charges Semi-Pvt (INR)")),
        "room_pvt":       _sf(r.get("Room Charges Pvt (INR)")),
        "icu_charge":     _sf(r.get("ICU Charges (INR)")),
        "vent_charge":    _sf(r.get("Ventilator Charges (INR)")),
        "clinical_notes": str(r.get("Clinical Notes","")).strip(),
        "tech_note":      str(r.get("Technology Cost Note","")).strip(),
        "is_diagnostic":  str(r.get("Procedure","")).strip() == "Diagnostics",
        # ICD-10 codes (columns 19-22, populated for all 62 procedure rows)
        "icd10_primary_code":   str(r.get("ICD-10 Primary Code",          "") or "").strip(),
        "icd10_primary_desc":   str(r.get("ICD-10 Primary Description",    "") or "").strip(),
        "icd10_secondary_code": str(r.get("ICD-10 Secondary Code",         "") or "").strip(),
        "icd10_secondary_note": str(r.get("ICD-10 Secondary / Clinical Note","") or "").strip(),
    }
def get_clinical_pathway(procedure: str, sub_procedure: str = None) -> list:
    """Return clinical pathway steps for a procedure."""
    df = _load_path()
    rows = df[df["Procedure"].str.strip() == procedure]
    if rows.empty:
        rows = df[df["Procedure"].str.contains(procedure, case=False, na=False)]
    if sub_procedure and not rows.empty:
        m = rows[rows["Sub-Procedure"].str.strip() == sub_procedure]
        if not m.empty:
            rows = m

    if rows.empty:
        return []

    steps = []
    for _, r in rows.iterrows():
        steps.append({
            "step":       int(_sf(r.get("Step", 0))),
            "phase":      str(r.get("Phase","")).strip(),
            "action":     str(r.get("Action / Investigation / Intervention","")).strip(),
            "clinician":  str(r.get("Responsible Clinician","")).strip(),
            "timeline":   str(r.get("Timeline / Frequency","")).strip(),
            "decision":   str(r.get("Key Decision / Threshold","")).strip(),
            "note":       str(r.get("Clinical Note","")).strip(),
        })
    return sorted(steps, key=lambda x: x["step"])

# =============================================================================
# SECTION 4 — MAIN ESTIMATE FUNCTION
# =============================================================================

def estimate(
    procedure:     str,
    hospital_tier: str,
    city:          str,
    room_type:     str  = "SEMI_PRIVATE",
    comorbidities: list = [],
    sub_procedure: str  = None,
    variant_hint:  str  = None,
    age:           int  = None,    # patient age in years
    gender:        str  = None,    # "male" | "female" | None
) -> dict:
    """
    Full itemized estimate.

    Formula:
      1. BASE_adj   = BASE × room_type_mult × tier_mult_min/max × comorb_mult
      2. ICU_adj    = ICU_charge × ICU_tier_mult × (icu_days + comorb_icu_days)
      3. VENT_adj   = vent_charge × vent_tier_mult
      4. MED_adj    = estimated_med × med_tier_mult (₹5k-20k per procedure)
      5. FIXED      = (IMPLANT + TECH) × cap_mult
      6. ROOM_GST   = 5% on room portion above ₹5,000/day (from room_charge_card)
      7. CONTINGENCY= 15% on (BASE_adj + ICU + VENT + MED + FIXED + ROOM_GST)
      8. GRAND      = all above
    """
    card = get_card(procedure, sub_procedure, variant_hint)
    if not card:
        return {"error": f"Procedure '{procedure}' not found"}

    city_cl   = city.strip().lower()
    city_tier = CITY_TIER.get(city_cl, "T2")
    key       = (hospital_tier, city_tier)

    mult_min, mult_max = TIER_VAR_MULT.get(key, (1.0, 1.8))
    cap_mult  = FIXED_CAP.get(hospital_tier, 1.0)
    icu_mult  = ICU_TIER_MULT.get(hospital_tier, 1.0)
    vent_mult = VENT_TIER_MULT.get(hospital_tier, 1.0)
    med_mult  = MED_TIER_MULT.get(hospital_tier, 1.0)
    room_adj  = ROOM_TYPE_MULT_ON_BASE.get(room_type, 1.0)

    # ── Diagnostics — simple flat multiplier, no GST ───────────────────────
    if card["is_diagnostic"]:
        base_fee = card["total"]  # TOTAL column = flat rate for diagnostics
        # Mild tier adjustment (diagnostics don't vary much by tier)
        diag_mult_min = min(mult_min * 0.5 + 0.5, 1.5)
        diag_mult_max = min(mult_max * 0.5 + 0.5, 2.0)
        consult = 350
        fee_min = round(base_fee * diag_mult_min)
        fee_max = round(base_fee * diag_mult_max)
        grand_min = fee_min + consult
        grand_max = fee_max + consult
        pathway = get_clinical_pathway(procedure, sub_procedure)

        return {
            "procedure":       card["procedure"],
            "sub_procedure":   card["sub_procedure"],
            "variant":         card["variant"],
            "hospital_tier":   hospital_tier,
            "city":            city,
            "room_type":       "OPD",
            "is_diagnostic":   True,
            "clinical_notes":  card["clinical_notes"],
            "clinical_pathway": pathway,
            "bill_breakdown": {
                "test_fee": {
                    "base":      base_fee,
                    "tier_adj":  f"{round(diag_mult_min,1)}x–{round(diag_mult_max,1)}x",
                    "estimated_min": fee_min,
                    "estimated_max": fee_max,
                },
                "consultation": consult,
                "gst":   "0% — diagnostics exempt",
            },
            "grand_total": {
                "min": grand_min,
                "max": grand_max,
                "formatted": f"₹{grand_min:,} – ₹{grand_max:,}",
            },
            "loan_relevant_amount": {
                "min":  grand_min,
                "max":  grand_max,
                "formatted": f"₹{grand_min:,} – ₹{grand_max:,}",
                "recommended_loan": round(grand_max * 1.10 / 10000) * 10000,
                "note": "Includes 15% contingency buffer + 10% lender safety margin.",
            },
            "reference_base": card["total"],
            "disclaimer": "Indicative. No GST on diagnostic tests in India.",
        }

    # ── Comorbidity ────────────────────────────────────────────────────────
    base_comorb_mult = 1.0
    extra_los        = 0
    icu_risk         = 0.0
    med_addon        = 0
    conditions       = []

    for c in comorbidities:
        if c in COMORBIDITY:
            cx = COMORBIDITY[c]
            base_comorb_mult *= cx["base_mult"]
            extra_los        += cx["extra_los"]
            icu_risk         += cx["icu_risk"]
            med_addon        += cx["med_addon"]
            conditions.append({
                "condition": c, "mult": cx["base_mult"],
                "extra_los": cx["extra_los"], "note": cx["note"]
            })

    # ── Age severity adjustment (independent of comorbidities) ────────────
    # Source: ASA-PS Classification + NHA PMJAY 2019-2023 claims data
    age_band      = _get_age_band(age)
    age_base_mult = age_band["base_mult"]
    age_extra_los = age_band["extra_los"]
    age_icu_risk  = age_band["icu_risk"]
    age_med_addon = age_band["med_addon"]
    age_note      = age_band["note"]

    # Combine age multiplier with comorbidity multiplier
    # Both are independent risk axes — multiply, not add
    base_comorb_mult *= age_base_mult
    extra_los        += age_extra_los
    icu_risk         += age_icu_risk
    med_addon        += age_med_addon

    # ── Gender validation + clinical flag ────────────────────────────────
    gender_flag = None
    if gender:
        g = gender.strip().lower()
        male_only = GENDER_PROCEDURE_RULES.get("male_only", [])
        if g == "female" and procedure in male_only:
            gender_flag = (
                f"⚠️ '{procedure}' is typically performed on male patients. "
                f"Please confirm procedure with your doctor."
            )
        if g == "female" and procedure == "CABG - Bypass Surgery":
            gender_flag = GENDER_PROCEDURE_RULES["cabg_female_note"]

    los_total = card["los"] + extra_los

    # ── Room charge from card (already room-type differentiated) ───────────
    room_charges_from_card = {
        "GENERAL":     card["room_gen"],
        "SEMI_PRIVATE": card["room_semi"],
        "PRIVATE":     card["room_pvt"],
    }.get(room_type, card["room_semi"])

    # ── BASE calculation ────────────────────────────────────────────────────
    # BASE from card = surgery + room (standard) + dr visits + misc
    # We apply: room_type_adj × tier_mult × comorb_mult
    base_adj = card["base"] * room_adj * base_comorb_mult
    base_min  = round(base_adj * mult_min)
    base_max  = round(base_adj * mult_max)

    # Extra LOS cost from comorbidity (additional room days)
    extra_room_cost = round(room_charges_from_card / card["los"] * extra_los
                            if card["los"] > 0 else 0)

    # ── ICU (semi-variable) ────────────────────────────────────────────────
    icu_days     = card["icu_days"]
    comorb_icu_days = 0
    if icu_risk >= 0.20 and icu_days == 0:
        comorb_icu_days = min(extra_los, 2)
    total_icu_days = icu_days + comorb_icu_days

    # ICU charge from card = base ₹5,400 × icu_days (Standard level)
    icu_rate_per_day = 5400 * icu_mult  # tier-adjusted ICU rate
    icu_cost = round(icu_rate_per_day * total_icu_days)

    # ── Ventilator (semi-variable) ─────────────────────────────────────────
    vent_base = card["vent_charge"]  # total ventilator cost at Standard
    vent_days = round(vent_base / 3000) if vent_base > 0 else 0
    vent_cost = round(3000 * vent_mult * vent_days)

    # ── Medical consumables estimate ───────────────────────────────────────
    # Estimated from procedure complexity (not in card — clinical estimate)
    PROC_MED_BASE = {
        "Knee Replacement": 8000, "Hip Replacement": 8000,
        "Cataract Surgery": 1500, "Angioplasty with Stent": 6000,
        "Hernia Repair": 4000, "Gallbladder Removal": 4000,
        "Appendectomy": 4000, "TURP - Prostate": 5000,
        "Kidney Stone PCNL": 6000, "Varicose Veins": 3000,
        "Piles Surgery": 3000, "Hysterectomy": 5000,
        "CABG - Bypass Surgery": 15000, "Chemotherapy": 5000,
        "Radiation Therapy": 2000, "Dialysis per Session": 500,
        "Neurosurgery - Brain Tumor": 12000, "Spinal Surgery": 10000,
    }
    med_base = PROC_MED_BASE.get(procedure, 5000)
    med_cost  = round((med_base + med_addon) * med_mult)

    # ── FIXED bucket ────────────────────────────────────────────────────────
    implant_final = round(card["implant"] * cap_mult)
    tech_final    = round(card["tech"] * cap_mult)
    fixed_total   = implant_final + tech_final

    # ── GST — Room only (>₹5,000/day) ────────────────────────────────────
    # Surgery/hospital services: EXEMPT in India
    # Diagnostics: 0%
    # Room rent >₹5,000/day: 5% on EXCESS only
    room_per_day = (room_charges_from_card / card["los"]
                    if card["los"] > 0 else 0)
    taxable_room_per_day = max(0, room_per_day - 5000)
    room_gst = round(taxable_room_per_day * los_total * 0.05)

    # ── SUBTOTALS ──────────────────────────────────────────────────────────
    sub_min = base_min + extra_room_cost + icu_cost + vent_cost + med_cost + fixed_total
    sub_max = base_max + extra_room_cost + icu_cost + vent_cost + med_cost + fixed_total

    total_with_gst_min = sub_min + room_gst
    total_with_gst_max = sub_max + room_gst

    # ── CONTINGENCY 15% ───────────────────────────────────────────────────
    cont_min = round(total_with_gst_min * 0.15)
    cont_max = round(total_with_gst_max * 0.15)

    grand_min = total_with_gst_min + cont_min
    grand_max = total_with_gst_max + cont_max

    # ── Component breakdown for output ────────────────────────────────────
    # BASE contains surgery + room + dr visits + misc bundled
    # We split for display purposes
    surg_pct  = 0.55  # ~55% of BASE is surgery/OT
    room_pct  = 0.25  # ~25% room + dr visits
    misc_pct  = 0.20  # ~20% misc/admin

    surg_min = round(base_min * surg_pct)
    surg_max = round(base_max * surg_pct)
    room_disp_min = round(base_min * room_pct) + extra_room_cost
    room_disp_max = round(base_max * room_pct) + extra_room_cost
    misc_min = round(base_min * misc_pct)
    misc_max = round(base_max * misc_pct)

    # ── Clinical Pathway ──────────────────────────────────────────────────
    pathway = get_clinical_pathway(procedure, sub_procedure or card["sub_procedure"])

    # ── BILL BREAKDOWN ────────────────────────────────────────────────────
    bill = {
        "━━ VARIABLE BUCKET (Tier Mult Applied) ━━": {
            "_multiplier": f"{mult_min}x – {mult_max}x  [{hospital_tier}, {city_tier}]",
            "_room_type_adj": f"{room_adj}x (PRIVATE=1.30x, SEMI=1.15x, GEN=1.00x)",

            "procedure_ot_surgeon": {
                "description": "Surgery package + OT + Anaesthesia + Dr Visits",
                "estimated_min": surg_min,
                "estimated_max": surg_max,
                "note": (f"Comorb adj {round(base_comorb_mult,2)}x on BASE ₹{card['base']:,.0f}. "
                         f"Room type adj {room_adj}x."),
            },
            "hospital_stay_room": {
                "description": f"Room charges ({room_type}) × {los_total} days",
                "room_per_day_at_tier": round(room_per_day * mult_min) if card["los"]>0 else 0,
                "estimated_min": room_disp_min,
                "estimated_max": room_disp_max,
                "note": f"Base room/day ₹{round(room_per_day):,} × {card['los']} base days + {extra_los} comorb days",
            },
            "misc_admin": {
                "description": "Misc/Admin (billing, ward, diet, attendant)",
                "estimated_min": misc_min,
                "estimated_max": misc_max,
            },
        },
        "━━ SEMI-VARIABLE BUCKET ━━": {
            "_note": "ICU/Vent/Medicines — own tier mult (gentler)",

            "icu_charges": {
                "description": f"ICU care — {total_icu_days} days",
                "icu_rate_per_day": round(icu_rate_per_day),
                "procedure_icu_days": icu_days,
                "comorbidity_icu_days": comorb_icu_days,
                "total": icu_cost,
                "note": f"₹{round(icu_rate_per_day):,}/day ({hospital_tier}: {icu_mult}x base ₹5,400)",
            },
            "ventilator_charges": {
                "description": f"Ventilator — {vent_days} days",
                "total": vent_cost,
                "note": f"₹{round(3000*vent_mult):,}/day ({vent_mult}x base)",
            },
            "medication": {
                "description": "Drugs, IV fluids, consumables",
                "total": med_cost,
                "note": f"Base ₹{med_base:,} + comorb ₹{med_addon:,}, tier {med_mult}x",
            },
        },
        "━━ FIXED BUCKET (No Tier Multiplier) ━━": {
            "_note": f"NPPA/CGHS capped. Cap mult: {cap_mult}x",

            "implant_consumable": {
                "description": "Prosthetic implant / device (NPPA regulated)",
                "card_value": card["implant"],
                "final":      implant_final,
                "note": "Fixed regardless of hospital tier. Govt/Small: 0.70-0.85x",
            },
            "technology_premium": {
                "description": "Tech/device cost (robotic, laser, navigation etc.)",
                "card_value": card["tech"],
                "final":      tech_final,
                "note": card["tech_note"] or "Device/platform cost",
            },
        },
        "━━ TAXES ━━": {
            "gst_room_5pct": {
                "description": "5% GST on room rent above ₹5,000/day only",
                "taxable_per_day": round(taxable_room_per_day),
                "total_days": los_total,
                "total": room_gst,
                "note": "Surgery/hospital services exempt from GST in India",
            },
            "surgery_gst": "EXEMPT — healthcare services not taxable in India",
            "medicine_gst": "EXEMPT — included in drug MRP",
            "diagnostic_gst": "0% — pathology/radiology exempt",
        },
        "━━ CONTINGENCY 15% ━━": {
            "min": cont_min,
            "max": cont_max,
            "note": "Mandatory 15% buffer (Poonawalla Fincorp standard)",
        },
    }

    if conditions:
        bill["━━ COMORBIDITY ADJUSTMENTS ━━"] = {
            "conditions": conditions,
            "combined_base_mult": round(base_comorb_mult, 2),
            "extra_los_days": extra_los,
            "extra_icu_days": comorb_icu_days,
            "disclaimer": "Clinical reasoning. Validate with hospital billing data in production.",
        }

    if age is not None and age_band.get("band") != "adult":
        bill["━━ AGE SEVERITY ADJUSTMENT ━━"] = {
            "age":             age,
            "band":            age_band["band"],
            "base_mult":       age_base_mult,
            "extra_los_days":  age_extra_los,
            "icu_risk_added":  age_icu_risk,
            "med_addon_inr":   age_med_addon,
            "note":            age_note,
            "source":          "ASA Physical Status Classification + NHA PMJAY Claims Data 2019-2023",
        }

    return {
        "procedure":        card["procedure"],
        "sub_procedure":    card["sub_procedure"],
        "variant":          card["variant"],
        "tech_type":        card["tech_type"],
        "hospital_tier":    hospital_tier,
        "city":             city,
        "city_tier":        city_tier,
        "room_type":        room_type,
        "comorbidities":    [c["condition"] for c in conditions],
        "patient_age":       age,
        "patient_gender":    gender,
        "age_band":          age_band.get("band", "adult"),
        "age_severity_note": age_note if age is not None else None,
        "gender_flag":       gender_flag,
        "is_diagnostic":    False,
        "clinical_notes":   card["clinical_notes"],
        "clinical_pathway": pathway,
        "icd10": {
            "primary_code":   card.get("icd10_primary_code",   ""),
            "primary_desc":   card.get("icd10_primary_desc",   ""),
            "secondary_code": card.get("icd10_secondary_code", ""),
            "secondary_note": card.get("icd10_secondary_note", ""),
        },

        "bill_breakdown": bill,

        "summary": {
            "procedure_ot":   {"min": surg_min, "max": surg_max},
            "hospital_stay":  {"min": room_disp_min, "max": room_disp_max},
            "medication":     med_cost,
            "icu_ventilator": icu_cost + vent_cost,
            "implant":        implant_final,
            "tech":           tech_final,
            "gst_room":       room_gst,
            "contingency":    {"min": cont_min, "max": cont_max},
        },

        "grand_total": {
            "min":       grand_min,
            "max":       grand_max,
            "formatted": f"₹{grand_min:,} – ₹{grand_max:,}",
        },
        "loan_relevant_amount": {
            "min":       grand_min,
            "max":       grand_max,
            "formatted": f"₹{grand_min:,} – ₹{grand_max:,}",
            "recommended_loan": round(grand_max * 1.10 / 10000) * 10000,
            "note": "Includes 15% contingency buffer + 10% lender safety margin.",
        },
        "reference_total_standard": card["total"],
        "disclaimer": (
            "Indicative estimate ±20–25%. Actual depends on case complexity, "
            "surgeon fee, implant brand. No GST on surgery/hospital services in India. "
            "Room GST 5% only if >₹5,000/day. Consult hospital for formal quotation."
        ),
    }

# =============================================================================
# SECTION 5 — ESTIMATE ALL VARIANTS
# =============================================================================

def estimate_all_variants(procedure: str, hospital_tier: str, city: str,
                           room_type: str = "SEMI_PRIVATE",
                           comorbidities: list = []) -> list:
    variants = get_all_variants(procedure)
    results  = []
    for v in variants:
        r = estimate(procedure, hospital_tier, city, room_type,
                     comorbidities, v["sub_procedure"], v["variant"])
        results.append(r)
    return results

# =============================================================================
# SECTION 6 — HOSPITAL ESTIMATOR
# =============================================================================



"""
=============================================================================
SECTION 6 — HOSPITAL ESTIMATOR (UPDATED v2.0)
=============================================================================
Paste karo: check.py mein line 650 se line 737 tak ka sara purana code 
DELETE karo aur yeh poora section paste karo.

CHANGES:
  1. Speciality-first filtering — procedure se required speciality match hoti hai
  2. Tier diversity — 1 Tertiary, 1 Advanced Multispecialty, 1 Boutique,
     1 Standard Secondary, 1 Small Day-Care (5 hospitals guaranteed)
  3. New confidence scoring — 7 weighted signals
  4. Distance secondary (tiebreaker, not primary sort)
=============================================================================
"""

# ─── PROCEDURE → REQUIRED SPECIALITY CODES MAPPING ───────────────────────────
# Yeh codes hospitals_with_types.csv ke "unified_specialities" column se match 
# karte hain (pipe-separated codes jaise ORT|CAR|GAS etc.)

PROCEDURE_SPECIALITY_MAP = {
    "Knee Replacement":          ["ORT"],                    # Orthopedics
    "Hip Replacement":           ["ORT"],
    "CABG - Bypass Surgery":     ["CAR", "CTS"],             # Cardiology / CardioThoracic Surgery
    "Cataract Surgery":          ["OPH"],                    # Ophthalmology
    "Hernia Repair":             ["SUR", "GAS"],             # Surgery / Gastro
    "Appendectomy":              ["SUR"],
    "Angioplasty with Stent":    ["CAR"],
    # Extra procedures (future-proofing)
    "Gallbladder Removal":       ["SUR", "GAS"],
    "TURP - Prostate":           ["URO"],
    "Kidney Stone PCNL":         ["URO"],
    "Neurosurgery - Brain Tumor":["NEU", "NSU"],
    "Spinal Surgery":            ["ORT", "NEU"],
    "Hysterectomy":              ["GYN", "OBS"],
    "CABG - Bypass Surgery":     ["CAR", "CTS"],
    "Chemotherapy":              ["ONC"],
    "Radiation Therapy":         ["ONC"],
}

# Tier priority order — yahi 5 best dikhane hain (ek ek)
TIER_PRIORITY = [
    "Tertiary Corporate",
    "Advanced Multispecialty",
    "Boutique Super-Specialty",
    "Standard Secondary General",
    "Small Day-Care Clinics",
]

def haversine(la1, lo1, la2, lo2):
    R = 6371
    la1, lo1, la2, lo2 = map(radians, [la1, lo1, la2, lo2])
    a = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1-a)), 1)

def _valid_coord(lat, lng):
    try:
        lf, lgf = float(lat), float(lng)
        if lf == 0.0 or lgf == 0.0: return False
        defaults = [(30.3165,78.0322),(21.1458,79.0882),
                    (28.6139,77.2090),(19.0760,72.8777)]
        for d in defaults:
            if abs(lf-d[0])<0.001 and abs(lgf-d[1])<0.001: return False
        return True
    except: return False
def _speciality_match_score(unified_specialities: str, required_codes: list) -> float:
    """
    Returns 0.0 to 1.0.
    - 1.0  = sab required codes mil gaye
    - 0.5  = koi ek match mila
    - 0.0  = kuch nahi mila
    """
    if not required_codes or not unified_specialities:
        return 0.3   # data nahi hai, neutral score
    
    hosp_codes = set(unified_specialities.upper().split("|"))
    matched = sum(1 for code in required_codes if code in hosp_codes)
    
    if matched == len(required_codes):
        return 1.0
    elif matched > 0:
        return 0.5 + (0.3 * matched / len(required_codes))
    return 0.0


def _distance_score(dist_km) -> float:
    """
    Distance ko 0.0-1.0 mein convert karo.
    < 3 km  → 1.0 (perfect)
    3-10 km → linear decay 1.0 → 0.5
    10-20km → linear decay 0.5 → 0.2
    > 20 km → 0.1
    None    → 0.3 (unknown, neutral)
    """
    if dist_km is None:
        return 0.3
    if dist_km < 3:
        return 1.0
    elif dist_km < 10:
        return 1.0 - (0.5 * (dist_km - 3) / 7)
    elif dist_km < 20:
        return 0.5 - (0.3 * (dist_km - 10) / 10)
    return 0.1


def _rating_score(rating: float, review_count: int) -> float:
    """
    Google rating + review count ko milake score banao.
    Rating zyada important hai, reviews credibility dete hain.
    """
    # Rating component (0 to 0.7)
    if rating >= 4.5:
        r_score = 0.70
    elif rating >= 4.0:
        r_score = 0.55
    elif rating >= 3.5:
        r_score = 0.35
    elif rating >= 3.0:
        r_score = 0.20
    elif rating > 0:
        r_score = 0.10
    else:
        r_score = 0.0

    # Review count component (0 to 0.3) — credibility
    if review_count >= 2000:
        rc_score = 0.30
    elif review_count >= 1000:
        rc_score = 0.22
    elif review_count >= 500:
        rc_score = 0.15
    elif review_count >= 100:
        rc_score = 0.08
    elif review_count > 0:
        rc_score = 0.03
    else:
        rc_score = 0.0

    return round(r_score + rc_score, 2)


def _conf_v2(row, required_speciality_codes: list, dist_km) -> dict:
    """
    NEW Confidence Score v2.0
    
    Weights:
      Speciality match   → 40% (PRIMARY)
      Google Rating      → 20%
      Review Count       → 10%
      Distance           → 15%
      Phone available    →  5%
      Website available  →  5%
      Accreditation      →  5%
    
    Returns dict with score + label + breakdown.
    """
    # 1. Speciality match (0.0 – 0.40)
    unified = str(row.get("unified_specialities_updated", "") or 
                  row.get("unified_specialities", "")).strip()
    spec_raw = _speciality_match_score(unified, required_speciality_codes)
    spec_component = round(spec_raw * 0.40, 3)

    # 2. Google Rating (0.0 – 0.20)
    rating = float(row.get("google_rating", 0) or 0)
    reviews_raw = row.get("google_reviews", 0)
    try:
        reviews = int(float(reviews_raw)) if reviews_raw == reviews_raw else 0  # NaN check
    except (ValueError, TypeError):
        reviews = 0
    if rating >= 4.5:
        rating_component = 0.20
    elif rating >= 4.0:
        rating_component = 0.15
    elif rating >= 3.5:
        rating_component = 0.10
    elif rating >= 3.0:
        rating_component = 0.05
    else:
        rating_component = 0.0

    # 3. Review Count (0.0 – 0.10)
    if reviews >= 2000:
        reviews_component = 0.10
    elif reviews >= 1000:
        reviews_component = 0.07
    elif reviews >= 500:
        reviews_component = 0.05
    elif reviews >= 100:
        reviews_component = 0.03
    elif reviews > 0:
        reviews_component = 0.01
    else:
        reviews_component = 0.0

    # 4. Distance (0.0 – 0.15)
    dist_raw = _distance_score(dist_km)
    dist_component = round(dist_raw * 0.15, 3)

    # 5. Phone available (0.0 – 0.05) — reviews_phone preferred, phone as fallback
    _BAD = ["NA", "nan", "", "None", "N/A"]
    phone = str(row.get("reviews_phone") or "").strip()
    if not phone or phone in _BAD:
        phone = str(row.get("phone", "NA") or "NA").strip()
    phone_component = 0.05 if phone not in _BAD else 0.0

    # 6. Website available (0.0 – 0.05) — reviews_website preferred, website as fallback
    website = str(row.get("reviews_website") or "").strip()
    if not website or website in _BAD:
        website = str(row.get("website", "NA") or "NA").strip()
    website_component = 0.05 if website not in _BAD else 0.0

    # 7. Accreditation (0.0 – 0.05)
    accred = str(row.get("accreditation_type", "") or "").strip()
    match_type = str(row.get("match_type", "") or "").strip()
    if "NABH" in accred or "NABH+PMJAY" in match_type:
        accred_component = 0.05
    elif "Accredited" in accred:
        accred_component = 0.04
    elif "Certified" in accred or "PMJAY" in match_type:
        accred_component = 0.02
    else:
        accred_component = 0.0

    total = (spec_component + rating_component + reviews_component +
             dist_component + phone_component + website_component + accred_component)
    total = min(round(total, 2), 1.0)

    # Label
    if total >= 0.75:
        label = "High"
    elif total >= 0.50:
        label = "Medium"
    elif total >= 0.30:
        label = "Low"
    else:
        label = "Very Low"

    # Speciality match explanation
    if spec_raw == 1.0:
        spec_label = "Full match"
    elif spec_raw >= 0.5:
        spec_label = "Partial match"
    else:
        spec_label = "No speciality match"

    return {
        "score": total,
        "label": label,
        "breakdown": {
            "speciality_match":  {"score": spec_component, "raw": spec_raw, "note": spec_label},
            "google_rating":     {"score": rating_component, "value": rating},
            "review_count":      {"score": reviews_component, "count": reviews},
            "distance":          {"score": dist_component, "km": dist_km},
            "phone_available":   {"score": phone_component},
            "website_available": {"score": website_component},
            "accreditation":     {"score": accred_component, "type": accred or "None"},
        }
    }


def estimate_for_hospitals(
    procedure: str,
    city: str,
    user_lat: float,
    user_lng: float,
    room_type: str = "SEMI_PRIVATE",
    comorbidities: list = [],
    sub_procedure: str = None,
    variant_hint: str = None,
    hospital_types: list = None,
    age:           int  = None,
    gender:        str  = None,
) -> dict:
    """
    Naya hospital suggestion engine.
    
    Logic:
    1. City ke hospitals filter karo
    2. Procedure ke liye required speciality codes nikalo
    3. Speciality match score calculate karo — NO match wale hospitals zyada
       neeche jaate hain (sirf backup mein aate hain agar tier fill karna ho)
    4. Har tier (5 tiers) se BEST hospital nikalo (speciality score + rating)
    5. Distance tiebreaker hai, primary sort nahi
    6. Confidence score v2 attach karo
    
    Returns 5 hospitals — 1 per tier (diversity guaranteed).
    """
    df = _load_hosp()
    hospitals = df[df["final_hub"].str.upper() == city.strip().upper()].copy()
    # Filter by hospital_types if provided (match against 'New_Types' column)
    if hospital_types:
        hospitals = hospitals[hospitals["New_Types"].isin(hospital_types)].copy()
    if hospitals.empty:
        return {"error": f"No hospitals found for city: {city} with types: {hospital_types}"}

    required_codes = PROCEDURE_SPECIALITY_MAP.get(procedure, [])
    all_results = []

    for _, row in hospitals.iterrows():
        tier = str(row.get("New_Types", "Standard Secondary General")).strip()
        if not tier or tier in ["nan", "", "Not Matched"]:
            tier = "Standard Secondary General"

        # Distance calculate karo — reviews_lat/lng (Places API) preferred over lat/lng
        dist = None
        lat  = row.get("reviews_lat") if _valid_coord(row.get("reviews_lat"), row.get("reviews_lng")) \
               else row.get("lat")
        lng  = row.get("reviews_lng") if _valid_coord(row.get("reviews_lat"), row.get("reviews_lng")) \
               else row.get("lng")
        if _valid_coord(lat, lng):
            try:
                dist = haversine(user_lat, user_lng, float(lat), float(lng))
            except:
                dist = None

        # Confidence score v2
        conf = _conf_v2(row, required_codes, dist)

        # Cost estimate
        cost = estimate(
            procedure=procedure, 
            hospital_tier=tier, 
            city=city, 
            room_type=room_type,
            comorbidities=comorbidities, 
            sub_procedure=sub_procedure, 
            variant_hint=variant_hint,
            age=age,
            gender=gender
        )

        # Speciality match raw (for filtering)
        unified = str(row.get("unified_specialities_updated", "") or
                      row.get("unified_specialities", "")).strip()
        spec_raw = _speciality_match_score(unified, required_codes)

        all_results.append({
            "hospital_name":   str(row.get("name", "")).strip(),
            "address":         str(row.get("address", "")).strip(),
            "hospital_tier":   tier,
            "nabh_status":     str(row.get("accreditation_type", "NA")),
            "google_rating":   float(row.get("google_rating", 0) or 0),
            "google_reviews":  int(float(row.get("google_reviews", 0) or 0)) 
                   if str(row.get("google_reviews", "")) not in ["nan", "", "None"] 
                   else 0,
            "specialities":    unified,
            "website":         _best(row, "reviews_website", "website"),
            "phone":           _best(row, "reviews_phone",   "phone"),
            "distance_km":     dist,
            "speciality_match": spec_raw,      # 0.0 / 0.5 / 1.0
            "confidence":      conf,
            "cost_estimate":   cost,
        })

    # ─── TIER DIVERSITY SELECTION ─────────────────────────────────────────────
    # Har tier se best hospital nikalo.
    # "Best" = speciality match priority, phir confidence score, phir distance.
    
    def hospital_sort_key(h):
        """
        Sort: speciality match DESC → confidence score DESC → distance ASC
        """
        dist = h["distance_km"]
        dist_val = dist if (dist is not None and not math.isnan(dist)) else 999.0
        return (
            -h["speciality_match"],          # Higher = better (negate for DESC)
            -h["confidence"]["score"],        # Higher = better (negate for DESC)
            dist_val,                         # Lower = better (natural ASC)
        )

    final_5 = []
    used_hospital_names = set()

    for tier in TIER_PRIORITY:
        # Is tier ke hospitals filter karo
        tier_hospitals = [h for h in all_results if h["hospital_tier"] == tier
                          and h["hospital_name"] not in used_hospital_names]
        
        if not tier_hospitals:
            continue
        
        # Best pick karo (speciality first, then confidence, then distance)
        tier_hospitals.sort(key=hospital_sort_key)
        best = tier_hospitals[0]
        
        final_5.append(best)
        used_hospital_names.add(best["hospital_name"])

    # Agar 5 se kam mila (city mein sab tiers nahi hain), fallback: 
    # remaining best hospitals jo abhi tak include nahi hue
    if len(final_5) < 5:
        remaining = [h for h in all_results 
                     if h["hospital_name"] not in used_hospital_names]
        remaining.sort(key=hospital_sort_key)
        for h in remaining:
            if len(final_5) >= 5:
                break
            final_5.append(h)
            used_hospital_names.add(h["hospital_name"])

    result = {
        "procedure":        procedure,
        "city":             city,
        "room_type":        room_type,
        "comorbidities":    comorbidities,
        "required_specialities": required_codes,
        "total_found":      len(all_results),
        "top_hospitals":    final_5,
        "selection_note":   (
            "1 hospital per tier shown (Tertiary → Advanced Multispecialty → "
            "Boutique → Standard → Day-Care). Ranked by: speciality match (40%) "
            "→ confidence score → distance."
        ),
    }
    return clean_nans(result)


# =============================================================================
# SECTION 7 — HOSPITAL LOOKUP BY NAME
# =============================================================================

def get_hospital_by_name(name: str) -> dict:
    """
    Look up a hospital by (partial, case-insensitive) name and return its details.
    Returns the first match with all columns as a dict, or None if not found.
    """
    df = _load_hosp()
    if not name or not isinstance(name, str):
        return None
    # Case-insensitive, partial match (use 'name' column from CSV)
    matches = df[df["name"].str.contains(name, case=False, na=False)]
    if matches.empty:
        return None
    return clean_nans(matches.iloc[0].to_dict())



# =============================================================================
# SECTION 9 — PERSONALIZED INPUT ESTIMATOR
# This replaces bulk validation export with user-input driven estimate flow.
# =============================================================================

def get_master_options() -> dict:
    df = _load_proc().copy()
    df.columns = df.columns.str.strip()

    procedure_map = {}
    for proc in sorted(df["Procedure"].dropna().astype(str).str.strip().unique()):
        proc_rows = df[df["Procedure"].astype(str).str.strip() == proc].copy()
        sub_map = {}
        for sub in sorted(proc_rows["Sub-Procedure"].fillna("").astype(str).str.strip().unique()):
            sub_rows = proc_rows[proc_rows["Sub-Procedure"].fillna("").astype(str).str.strip() == sub].copy()
            variants = sorted(sub_rows["Clinical Variant"].fillna("").astype(str).str.strip().unique().tolist())
            tech_types = sorted(sub_rows["Tech Type"].fillna("").astype(str).str.strip().unique().tolist())
            sub_map[sub] = {
                "variants": variants,
                "tech_types": tech_types,
            }
        procedure_map[proc] = sub_map

    return {
        "procedures": procedure_map,
        "hospital_tiers": sorted(list({k[0] for k in TIER_VAR_MULT.keys()})),
        "cities": sorted(list(CITY_TIER.keys())),
        "room_types": list(ROOM_TYPE_MULT_ON_BASE.keys()),
        "comorbidities": sorted(list(COMORBIDITY.keys())),
    }


def print_master_options():
    opts = get_master_options()
    print("\n================ AVAILABLE INPUTS ================")
    print("\nHospital Tiers:")
    for x in opts["hospital_tiers"]:
        print(" -", x)

    print("\nCities:")
    for x in opts["cities"]:
        print(" -", x)

    print("\nRoom Types:")
    for x in opts["room_types"]:
        print(" -", x)

    print("\nComorbidities:")
    for x in opts["comorbidities"]:
        print(" -", x)

    print("\nProcedures / Sub-Procedures / Variants:")
    for proc, subs in opts["procedures"].items():
        print(f"\n>> {proc}")
        for sub, meta in subs.items():
            print(f"   - Sub-Procedure: {sub or '[blank]'}")
            print(f"     Variants: {', '.join(meta['variants']) if meta['variants'] else 'N/A'}")
            print(f"     Tech Types: {', '.join(meta['tech_types']) if meta['tech_types'] else 'N/A'}")


def validate_user_inputs(
    procedure: str,
    sub_procedure: str = None,
    variant: str = None,
    hospital_tier: str = None,
    city: str = None,
    room_type: str = None,
    comorbidities: list = None,
) -> dict:
    opts = get_master_options()
    errors = []
    comorbidities = comorbidities or []

    if procedure not in opts["procedures"]:
        errors.append(f"Invalid procedure: {procedure}")
    else:
        allowed_subs = opts["procedures"][procedure]
        if sub_procedure is not None and sub_procedure not in allowed_subs:
            errors.append(f"Invalid sub_procedure for '{procedure}': {sub_procedure}")
        elif sub_procedure is not None and variant is not None:
            allowed_variants = allowed_subs[sub_procedure]["variants"]
            if variant not in allowed_variants:
                errors.append(f"Invalid variant for '{procedure}' / '{sub_procedure}': {variant}")

    if hospital_tier not in opts["hospital_tiers"]:
        errors.append(f"Invalid hospital_tier: {hospital_tier}")

    if city is None or city.strip().lower() not in opts["cities"]:
        errors.append(f"Invalid city: {city}")

    if room_type not in opts["room_types"]:
        errors.append(f"Invalid room_type: {room_type}")

    bad_comorb = [c for c in comorbidities if c not in opts["comorbidities"]]
    if bad_comorb:
        errors.append(f"Invalid comorbidities: {bad_comorb}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
    }


def get_personalized_bill(
    procedure: str,
    sub_procedure: str,
    variant: str,
    hospital_tier: str,
    city: str,
    room_type: str,
    comorbidities: list = None,
    age:           int  = None,
    gender:        str  = None,
    save_json: bool = False,
    json_path: str = "personalized_bill_output.json",
):
    comorbidities = comorbidities or []
    city = city.strip().lower()
    room_type = room_type.strip().upper()

    check = validate_user_inputs(
        procedure=procedure,
        sub_procedure=sub_procedure,
        variant=variant,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        comorbidities=comorbidities,
    )

    if not check["ok"]:
        return {
            "ok": False,
            "errors": check["errors"],
            "message": "Input validation failed"
        }

    result = estimate(
        procedure=procedure,
        sub_procedure=sub_procedure,
        variant_hint=variant,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        comorbidities=comorbidities,
        age=age,
        gender=gender,
    )

    if save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "ok": True,
        "inputs": {
            "procedure": procedure,
            "sub_procedure": sub_procedure,
            "variant": variant,
            "hospital_tier": hospital_tier,
            "city": city,
            "room_type": room_type,
            "comorbidities": comorbidities,
            "age": age,
            "gender": gender,
        },
        "result": result,
    }


def print_bill_result(response: dict):
    if not response.get("ok"):
        print("\n❌ INPUT ERROR")
        for e in response.get("errors", []):
            print(" -", e)
        return

    r = response["result"]
    print("\n================ ESTIMATE RESULT ================")
    print("Procedure      :", r.get("procedure"))
    print("Sub-Procedure  :", r.get("sub_procedure"))
    print("Variant        :", r.get("variant"))
    print("Tech Type      :", r.get("tech_type"))
    print("Hospital Tier  :", r.get("hospital_tier"))
    print("City           :", r.get("city"), f"({r.get('city_tier', '')})")
    print("Room Type      :", r.get("room_type"))
    print("Comorbidities  :", ", ".join(r.get("comorbidities", [])) if r.get("comorbidities") else "NONE")
    print("Grand Total    :", r.get("grand_total", {}).get("formatted"))
    #print("Loan Relevant  :", r.get("loan_relevant_amount", {}).get("formatted"))
    print("Recommended Loan:", r.get("loan_relevant_amount", {}).get("recommended_loan"))

    print("\n---- Summary ----")
    s = r.get("summary", {}) or {}
    print("Procedure/OT   :", s.get("procedure_ot"))
    print("Hospital Stay  :", s.get("hospital_stay"))
    print("Medication     :", s.get("medication"))
    print("ICU/Ventilator :", s.get("icu_ventilator"))
    print("Implant        :", s.get("implant"))

    print("Tech           :", s.get("tech"))
    print("Room GST       :", s.get("gst_room"))
    print("Contingency    :", s.get("contingency"))

    print("\n---- Full Bill JSON ----")
    print(json.dumps(r, ensure_ascii=False, indent=2))


# =============================================================================
# QUICK EXAMPLE
# =============================================================================
# 1) Print all available options:
# print_master_options()
#
# 2) Get one estimate:
# resp = get_personalized_bill(
#      procedure="Knee Replacement",
#      sub_procedure="TKR Bilateral",
#     variant="Standard CoCr - Both Knees Simultaneous",
#      hospital_tier="Advanced Multispecialty",
#      city="Delhi",
#      room_type="GENERAL",
#      comorbidities=["diabetes"],
#      save_json=True,
#      json_path="delhi_knee_replacement_bill.json",
#  )
# print_bill_result(resp)