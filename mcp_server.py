"""
=============================================================================
MCP SERVER — HealthcareCostEngine  v2.0
Poonawalla Fincorp | Agents Assemble Hackathon 2025
=============================================================================
Tools (13 total):
  ── Already present (fixed/upgraded) ──
  1.  calculate_procedure_cost      — itemized bill  [+ age, gender, budget_inr]
  2.  calculate_insurance_gap       — FHIR coverage gap
  3.  get_procedure_pathway         — clinical care steps
  4.  lookup_hospital               — hospital info  [+ enriched review data]

  ── New tools (Priority 1) ──
  5.  find_specialty_hospitals      — VADER review-ranked hospital list by city+specialty
  6.  get_top_doctors               — top 3 specialty doctors from patient reviews

  ── New tools (Priority 2) ──
  7.  find_hospitals_for_procedure  — procedure-ranked hospitals across all tiers + budget filter
  8.  get_hospital_review_scores    — raw VADER specialty scores for a named hospital
  9.  get_valid_options             — all valid procedures / tiers / cities / comorbidities

  ── New tools (Priority 3) ──
  10. get_all_procedure_variants    — all sub-procedures, variants, LOS, base costs
  11. get_procedure_card            — raw rate-card data for a procedure

  ── New tools (Priority 4) ──
  12. estimate_all_variants_cost    — cost comparison across all variants for one procedure

  ── New tools (Priority 5) ──
  13. validate_inputs               — validate params before calling cost tools

Run:  python mcp_server.py
Then: ngrok http 8000   →  paste the https URL into your agent config
=============================================================================
"""

from mcp.server.fastmcp import FastMCP
from check import (
    get_personalized_bill,
    get_hospital_by_name,
    get_clinical_pathway,
    get_hospital_specialty_scores,
    get_all_variants,
    get_card,
    get_master_options,
    validate_user_inputs,
    estimate_for_hospitals,
    estimate_all_variants,
    clean_nans,
    haversine,
    CITY_TIER,
)
import json

mcp = FastMCP("HealthcareCostEngine")

# Fix for ngrok Invalid Host header error
import mcp.server.sse as _sse_module
_original_init = _sse_module.SseServerTransport.__init__

def _patched_init(self, endpoint, *args, **kwargs):
    _original_init(self, endpoint, *args, **kwargs)
    self._allow_all_hosts = True

_sse_module.SseServerTransport.__init__ = _patched_init


# =============================================================================
# SHARED HELPERS (copied from chatbot.py so MCP server is self-contained)
# =============================================================================

CITY_COORDS = {
    "delhi":     (28.6139, 77.2090),
    "mumbai":    (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "indore":    (22.7196, 75.8577),
    "bhopal":    (23.2599, 77.4126),
    "nagpur":    (21.1458, 79.0882),
    "jaipur":    (26.9124, 75.7873),
    "lucknow":   (26.8467, 80.9462),
    "dehradun":  (30.3165, 78.0322),
}

SPEC_LABEL = {
    "ORT": "Orthopedic",     "CAR": "Cardiology",        "OPH": "Ophthalmology",
    "NEU": "Neurology",      "GAS": "Gastroenterology",  "URO": "Urology",
    "ONC": "Oncology",       "GYN": "Gynecology",        "CTS": "Cardiac Surgery",
    "SUR": "Surgery",        "NPH": "Nephrology",        "PUL": "Pulmonology",
    "MED": "General Medicine","EDO": "Endocrinology",    "EMD": "Emergency",
}

SPEC_REVIEW_FALLBACK = {
    "CTS": "CAR",
    "PUL": "MED",
    "EDO": "MED",
    "NPH": "URO",
}

_CONTACT_BAD = {"", "NA", "N/A", "nan", "None", "none", "NaN"}

def _resolve(row_or_dict, primary_col: str, fallback_col: str, default: str = "N/A") -> str:
    val = str(row_or_dict.get(primary_col) or "").strip()
    if not val or val in _CONTACT_BAD:
        val = str(row_or_dict.get(fallback_col) or "").strip()
    if not val or val in _CONTACT_BAD:
        val = default
    return val

_HOSP_SKIP = {
    "hospital", "clinic", "multispecialty", "multi", "pvt", "ltd", "the",
    "of", "and", "a", "unit", "centre", "center", "care", "health",
    "medical", "research", "speciality", "specialty", "super", "institute",
}

def _norm_name(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())

def enrich_hospital(hospital_name: str, spec_code: str, scores_db) -> dict:
    """Pull review intelligence for one hospital + specialty."""
    hosp_scores = scores_db(hospital_name)
    spec_data   = hosp_scores.get(spec_code, {}) if spec_code else {}

    if not spec_data.get("mention_count") and spec_code in SPEC_REVIEW_FALLBACK:
        fallback_code = SPEC_REVIEW_FALLBACK[spec_code]
        fb_data = hosp_scores.get(fallback_code, {})
        if fb_data.get("mention_count", 0) > 0:
            spec_data = fb_data

    spec_score    = spec_data.get("specialty_score", 0) or 0
    sentiment     = spec_data.get("sentiment", 0) or 0
    mention_count = spec_data.get("mention_count", 0) or 0
    doctors       = spec_data.get("doctors", [])

    if mention_count == 0:
        match_quality = "No review data"
    elif spec_score >= 0.4:
        match_quality = "Strong"
    elif spec_score >= 0.1:
        match_quality = "Moderate"
    else:
        match_quality = "Weak"

    label = SPEC_LABEL.get(spec_code, spec_code)
    strength_tags = []
    if spec_score >= 0.4:
        strength_tags.append(f"Highly praised for {label} in reviews")
    elif spec_score >= 0.1:
        strength_tags.append(f"Generally positive {label} reviews")

    risk_flags = []
    emd = hosp_scores.get("EMD", {})
    if emd.get("sentiment", 1) < 0 and emd.get("mention_count", 0) >= 2:
        risk_flags.append("ICU / emergency complaints noted in reviews")
    if sentiment < 0 and mention_count >= 2:
        risk_flags.append(f"Mixed reviews for {label}")

    top_docs = []
    for d in sorted(doctors, key=lambda x: (-x["mentions"], -x["avg_sentiment"])):
        if d["mentions"] >= 1 and d["avg_sentiment"] >= 0:
            top_docs.append({
                "name":            d["name"],
                "specialty_label": label,
                "mentions":        d["mentions"],
                "sentiment":       d["avg_sentiment"],
            })

    return {
        "spec_score":              spec_score,
        "mention_count":           mention_count,
        "specialty_match_quality": match_quality,
        "strength_tags":           strength_tags,
        "risk_flags":              risk_flags,
        "top_doctors":             top_docs[:2],
    }


def get_top_doctors_for_specialty(hospitals: list, spec_code: str) -> list:
    """
    Aggregate top 3 doctors across ranked hospitals for a specialty.
    Deduplicates by prefix + last-name matching.
    """
    label = SPEC_LABEL.get(spec_code, spec_code)
    seen  = {}

    for h in hospitals:
        hosp_scores = get_hospital_specialty_scores(h["name"])
        spec_data   = hosp_scores.get(spec_code, {}) if spec_code else {}
        all_docs    = spec_data.get("doctors", [])
        for doc in all_docs:
            if doc.get("avg_sentiment", 0) < 0:
                continue
            key      = doc["name"].lower().strip()
            mentions = doc.get("mentions", 1)
            sentiment= doc.get("avg_sentiment", 0)
            if key not in seen or mentions > seen[key]["mentions"]:
                seen[key] = {
                    "name":            doc["name"],
                    "specialty_label": label,
                    "mentions":        mentions,
                    "sentiment":       round(sentiment, 3),
                    "hospital":        h["name"],
                    "hospital_tier":   h.get("tier", ""),
                    "hospital_rating": h.get("rating", 0),
                    "hospital_phone":  h.get("phone", ""),
                    "match_quality":   h.get("specialty_match_quality", ""),
                    "reason": (
                        f"Most mentioned {label} specialist in patient reviews"
                        if mentions >= 3
                        else f"Mentioned {mentions}x in patient reviews for {label}"
                    ),
                }

    # Dedup pass 1: prefix match
    all_keys  = list(seen.keys())
    to_remove = set()
    for k in all_keys:
        for other in all_keys:
            if k != other and other.startswith(k + " "):
                to_remove.add(k)

    # Dedup pass 2: last-name match
    def last_name(key: str) -> str:
        parts = key.strip().split()
        return parts[-1] if parts else key

    last_name_map = {}
    for k, v in seen.items():
        if k in to_remove:
            continue
        ln = last_name(k)
        if ln not in last_name_map:
            last_name_map[ln] = k
        else:
            existing_key = last_name_map[ln]
            if (v["mentions"] > seen[existing_key]["mentions"] or
                    (v["mentions"] == seen[existing_key]["mentions"] and len(k) > len(existing_key))):
                to_remove.add(existing_key)
                last_name_map[ln] = k
            else:
                to_remove.add(k)

    deduped     = {k: v for k, v in seen.items() if k not in to_remove}
    sorted_docs = sorted(deduped.values(), key=lambda d: (-d["mentions"], -d["sentiment"]))
    return sorted_docs[:3]


def _get_hospitals_by_specialty(
    specialty_code: str, city: str,
    user_lat=None, user_lng=None, limit=5
) -> list:
    """
    Filter hospitals by specialty + city, score with 6-factor composite,
    return ranked list with enriched VADER review intelligence.
    """
    try:
        from check import _load_hosp, haversine, _valid_coord, _best
        df = _load_hosp()

        city_cl = city.strip().lower() if city else ""
        if not city_cl:
            return []

        city_df = df[df["city"].str.lower().str.strip() == city_cl] if "city" in df.columns else df
        if city_df.empty:
            city_df = df[df["city"].str.lower().str.contains(city_cl, na=False)]
        if city_df.empty:
            return []

        spec_code_up = specialty_code.upper() if specialty_code else ""
        
        # Pre-filter based on specialty code
        if spec_code_up:
            city_df = city_df[city_df['unified_specialities_updated'].str.contains(spec_code_up, na=False, case=False)].copy()

        if city_df.empty:
            return []

        lat, lng = (
            (float(user_lat), float(user_lng))
            if user_lat and user_lng
            else CITY_COORDS.get(city_cl, (22.7196, 75.8577))
        )
        eff_code = SPEC_REVIEW_FALLBACK.get(spec_code_up, spec_code_up) if spec_code_up else ""

        seen_names = set()
        hospitals  = []
        for _, row in city_df.iterrows():
            try:
                raw_name = str(row.get("hospital_name", row.get("name", "N/A"))).strip()
                norm = _norm_name(raw_name)
                if norm in seen_names:
                    continue
                seen_names.add(norm)

                h_lat = float(row.get("reviews_lat") or row.get("lat") or 0)
                h_lng = float(row.get("reviews_lng") or row.get("lng") or 0)
                dist  = round(haversine(lat, lng, h_lat, h_lng), 1) if h_lat and h_lng else None
                if dist and dist > 100:
                    dist = None

                mas_raw = float(row.get("market_alignment_score", 0) or 0)
                hospitals.append({
                    "name":        raw_name,
                    "tier":        str(row.get("New_Types", row.get("hospital_tier", "N/A"))).strip(),
                    "rating":      float(row.get("google_rating", 0) or 0),
                    "reviews":     int(row.get("google_reviews", 0) or 0),
                    "distance_km": dist,
                    "phone":       _resolve(row, "reviews_phone",   "phone"),
                    "website":     _resolve(row, "reviews_website", "website"),
                    "address":     str(row.get("address", "N/A")).strip(),
                    "nabh":        str(row.get("accreditation_type", "N/A")).strip(),
                    "specialities":str(row.get("unified_specialities_updated", row.get("unified_specialities", ""))).strip(),
                    "mas":         mas_raw,
                })
            except Exception:
                continue

        specialty_scores_db = get_hospital_specialty_scores

        for h in hospitals:
            enriched = enrich_hospital(h["name"], spec_code_up, specialty_scores_db)
            if enriched["spec_score"] <= 0 and eff_code and eff_code != spec_code_up:
                enriched_fb = enrich_hospital(h["name"], eff_code, specialty_scores_db)
                if enriched_fb["spec_score"] > 0:
                    enriched = enriched_fb

            spec_str       = h["specialities"].upper()
            has_official   = (spec_code_up in spec_str) or (eff_code and eff_code in spec_str)
            mas_norm       = h["mas"] / 10.0 if has_official else 0.0
            rating_norm    = h["rating"] / 5.0 if h["rating"] else 0
            dist_score     = max(0.0, 1.0 - (h["distance_km"] / 30.0)) if h["distance_km"] and h["distance_km"] > 0 else 0.5
            nabh_bonus     = 0.08 if (h.get("nabh") and h["nabh"] not in ["N/A", "None", "", "nan"]) else 0.0
            doc_quality    = max(0, enriched["top_doctors"][0]["sentiment"]) * 0.5 if enriched["top_doctors"] else 0.0
            risk_penalty   = len(enriched["risk_flags"]) * 0.05

            mq         = enriched["specialty_match_quality"]
            spec_score = enriched["spec_score"]

            if spec_score <= 0.1 and has_official and h["mas"] >= 5.0:
                if h["mas"] >= 8.0:
                    mq = "Market Aligned"
                elif mq == "No review data":
                    mq = "Market Aligned (Secondary)"
                spec_score = mas_norm * 0.85
                label = SPEC_LABEL.get(spec_code_up, spec_code_up)
                enriched["strength_tags"].append(f"Top-rated for {label} based on market data")

            mq_boost = {
                "Strong": 0.20, "Moderate": 0.10, "Weak": 0.02,
                "Market Aligned": 0.15, "Market Aligned (Secondary)": 0.08,
                "No review data": 0.0,
            }.get(mq, 0.0)

            h["composite_score"]         = (
                rating_norm * 0.20 + spec_score * 0.35 + dist_score * 0.15 +
                nabh_bonus + doc_quality * 0.10 + max(0, 0.10 - risk_penalty) +
                mq_boost + (mas_norm * 0.15)
            )
            h["specialty_match_quality"] = mq
            h["strength_tags"]           = enriched["strength_tags"]
            h["risk_flags"]              = enriched["risk_flags"]
            h["top_doctors"]             = enriched["top_doctors"]
            h["mention_count"]           = enriched["mention_count"]
            h["specialty_sentiment"]     = round(spec_score, 3)
            h["review_doctors"]          = [d["name"] for d in enriched["top_doctors"]]

        reviewed = [h for h in hospitals
                    if h["specialty_match_quality"] in ("Strong", "Moderate", "Weak")
                    and h["specialty_sentiment"] > 0]
        market   = [h for h in hospitals
                    if h["specialty_match_quality"] in (
                        "Market Aligned", "Market Aligned (Secondary)", "No review data")]

        reviewed.sort(key=lambda h: -h["composite_score"])
        market.sort(key=lambda h: -h["composite_score"])

        top_2, used = [], set()
        for h in reviewed:
            if len(top_2) >= 2: break
            if h["name"] not in used:
                h["_source"] = "reviews"
                top_2.append(h); used.add(h["name"])

        top_3 = []
        for h in market + reviewed:
            if len(top_3) >= (limit - len(top_2)): break
            if h["name"] not in used:
                h["_source"] = h.get("_source", "market")
                top_3.append(h); used.add(h["name"])

        return top_2 + top_3

    except Exception:
        return []


# =============================================================================
# TOOL 1 — Procedure Cost Calculator  [UPDATED: + age, gender, budget_inr]
# =============================================================================

@mcp.tool()
def calculate_procedure_cost(
    procedure: str,
    city: str,
    hospital_tier: str,
    room_type: str = "SEMI_PRIVATE",
    sub_procedure: str = None,
    variant: str = None,
    comorbidities: list = None,
    age: int = None,
    gender: str = None,
    budget_inr: int = None,
) -> str:
    """
    Calculates a detailed, itemized medical bill for a healthcare procedure
    in India. Returns cost split into two buckets:
      - Bucket A (Variable): Surgery package, room charges, ICU, medications
      - Bucket B (Fixed/Capped): NPPA-regulated implants and technology fees

    Also accepts age and gender for personalized risk-adjusted cost, and
    budget_inr to check affordability (within / borderline / over).

    Args:
        procedure:Exact procedure name. Options:
                    "Knee Replacement", "Hip Replacement",
                    "CABG - Bypass Surgery", "Cataract Surgery",
                    "Hernia Repair", "Appendectomy",
                    "Angioplasty with Stent", "Gallbladder Removal",
                    "TURP - Prostate", "Kidney Stone PCNL",
                    "Neurosurgery - Brain Tumor", "Spinal Surgery",
                    "Chemotherapy", "Radiation Therapy",
                    "Diagnostics", "Dialysis per Session",
                    "Piles Surgery", "Varicose Veins"
        city:          City name (delhi, mumbai, indore, bhopal, nagpur,
                       jaipur, lucknow, dehradun, bangalore)
        hospital_tier: One of:
                       "Tertiary Corporate",
                       "Advanced Multispecialty",
                       "Boutique Super-Specialty",
                       "Standard Secondary General",
                       "Small Day-Care Clinics",
                       "Government"
        room_type:     "GENERAL" | "SEMI_PRIVATE" | "PRIVATE"  (default: SEMI_PRIVATE)
        sub_procedure: Optional sub-procedure name (from rate cards)
        variant:       Optional variant (e.g. "Bilateral", "Robotic")
        comorbidities: Optional list e.g. ["diabetes", "hypertension"]
        age:           Patient age in years — adjusts for paediatric/senior/elderly risk
        gender:        "male" | "female" | None — triggers gender-specific clinical flags
        budget_inr:    Optional budget in INR (plain integer). Returns within/borderline/over status.
    """
    comorbidities = comorbidities or []

    bill_data = get_personalized_bill(
        procedure=procedure,
        sub_procedure=sub_procedure,
        variant=variant,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        comorbidities=comorbidities,
        age=age,
        gender=gender,
        save_json=False,
    )

    if not bill_data.get("ok", True) and bill_data.get("errors"):
        return json.dumps({
            "success": False,
            "errors": bill_data["errors"],
            "message": "Input validation failed. Check procedure name, city, and hospital_tier spelling.",
        }, indent=2)

    # bill_data structure: {"ok": True, "inputs": {...}, "result": {...}}
    r = bill_data.get("result", bill_data)  # fallback for direct estimate() calls

    summary     = r.get("summary", {})
    grand_total = r.get("grand_total", {})

    def _val(field, key="max"):
        v = summary.get(field, 0)
        if isinstance(v, dict):
            return v.get(key, 0) or 0
        return v or 0

    bucket_a = (
        _val("procedure_ot") +
        _val("hospital_stay") +
        _val("medication") +
        _val("icu_ventilator") +
        _val("gst_room")
    )
    bucket_b = _val("implant") + _val("tech")

    formatted = {
        "success":       True,
        "procedure":     r.get("procedure"),
        "sub_procedure": r.get("sub_procedure"),
        "variant":       r.get("variant"),
        "hospital_tier": r.get("hospital_tier"),
        "city":          r.get("city"),
        "room_type":     r.get("room_type"),
        "age_band":      r.get("age_band"),
        "gender_flag":   r.get("gender_flag"),
        "icd10":         r.get("icd10", {}),
        "Bucket_A_Variable_Base": {
            "description":  "Surgery package + room + ICU + medications (tier-adjusted)",
            "amount_inr":   round(bucket_a),
        },
        "Bucket_B_Fixed_Capped_Base": {
            "description":  "NPPA-regulated implants + technology fees (fixed, no tier markup)",
            "amount_inr":   round(bucket_b),
        },
        "Grand_Total": {
            "min_inr":   grand_total.get("min", 0),
            "max_inr":   grand_total.get("max", 0),
            "formatted": grand_total.get("formatted", "N/A"),
        },
        "Loan_Recommendation": r.get("loan_relevant_amount", {}),
        "disclaimer":    r.get("disclaimer"),
    }

    # ── Budget check ──────────────────────────────────────────────────────────
    if budget_inr:
        cost_min = grand_total.get("min", 0)
        cost_max = grand_total.get("max", 0)
        if cost_max <= budget_inr:
            budget_status  = "within"
            budget_message = f"✅ Estimated max ₹{cost_max:,} is within your budget of ₹{budget_inr:,}."
        elif cost_min <= budget_inr < cost_max:
            budget_status  = "borderline"
            budget_message = (
                f"⚠️ Minimum estimate ₹{cost_min:,} is within budget, "
                f"but max ₹{cost_max:,} may exceed ₹{budget_inr:,}. "
                "Consider a General room to reduce costs."
            )
        else:
            budget_status  = "over_budget"
            budget_message = (
                f"❌ Estimated ₹{cost_min:,}–₹{cost_max:,} exceeds budget ₹{budget_inr:,}. "
                "Try a lower-tier hospital or Government facility."
            )
        formatted["budget_check"] = {
            "budget_inr":    budget_inr,
            "status":        budget_status,
            "message":       budget_message,
        }

    return json.dumps(formatted, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 2 — Insurance Gap Calculator (FHIR / SHARP Context)
# =============================================================================

@mcp.tool()
def calculate_insurance_gap(
    patient_id: str,
    total_cost_inr: float,
    coverage_tier: str = "standard",
) -> str:
    """
    Simulates a FHIR Coverage resource lookup and calculates the patient's
    out-of-pocket gap after insurance. Designed for SHARP Context integration.

    IMPORTANT: Uses synthetic/de-identified data only. No real PHI.

    Args:
        patient_id:      Synthetic patient identifier (e.g. "P001", "PAT-123")
        total_cost_inr:  Total estimated procedure cost in INR
        coverage_tier:   "basic" (2L) | "standard" (4L) | "premium" (10L)
    """
    SYNTHETIC_PLANS = {
        "basic": {
            "plan_name": "PMJAY / Government Health Scheme (Synthetic)",
            "insurer": "National Health Authority (Simulated)",
            "coverage_limit_inr": 200000,
            "deductible_inr": 0,
            "copay_percent": 0,
            "fhir_resource_id": f"Coverage/syn-{patient_id}-basic",
        },
        "standard": {
            "plan_name": "Corporate Group Mediclaim - Standard (Synthetic)",
            "insurer": "Star Health / HDFC ERGO (Simulated)",
            "coverage_limit_inr": 400000,
            "deductible_inr": 5000,
            "copay_percent": 10,
            "fhir_resource_id": f"Coverage/syn-{patient_id}-std",
        },
        "premium": {
            "plan_name": "Super Top-Up Policy (Synthetic)",
            "insurer": "Niva Bupa / ICICI Lombard (Simulated)",
            "coverage_limit_inr": 1000000,
            "deductible_inr": 10000,
            "copay_percent": 5,
            "fhir_resource_id": f"Coverage/syn-{patient_id}-prem",
        },
    }

    plan             = SYNTHETIC_PLANS.get(coverage_tier.lower(), SYNTHETIC_PLANS["standard"])
    coverage_limit   = plan["coverage_limit_inr"]
    deductible       = plan["deductible_inr"]
    copay_pct        = plan["copay_percent"] / 100.0
    after_deductible = max(0.0, total_cost_inr - deductible)
    insurer_eligible = min(after_deductible, coverage_limit)
    insurer_pays     = round(insurer_eligible * (1 - copay_pct))
    financial_gap    = max(0, round(total_cost_inr - insurer_pays))

    if financial_gap <= 0:
        recommendation = "Procedure fully covered by insurance. No medical loan needed."
        loan_suggestion = None
    elif financial_gap <= 100000:
        recommendation = (
            f"Patient gap of Rs.{financial_gap:,} is manageable. "
            "A small personal loan or EMI plan from Poonawalla Fincorp is recommended."
        )
        loan_suggestion = {
            "suggested_loan_inr": round(financial_gap * 1.10 / 5000) * 5000,
            "note":    "10% buffer added for contingency",
            "product": "Poonawalla Fincorp Personal Loan / Medical Loan",
        }
    else:
        recommendation = (
            f"Significant gap of Rs.{financial_gap:,} detected. "
            "Triggering Poonawalla Fincorp medical loan workflow is strongly recommended."
        )
        loan_suggestion = {
            "suggested_loan_inr": round(financial_gap * 1.10 / 10000) * 10000,
            "note":    "10% buffer added for contingency + rounding",
            "product": "Poonawalla Fincorp Medical Loan",
            "action":  "TRIGGER_LOAN_WORKFLOW",
        }

    return json.dumps({
        "success":    True,
        "patient_id": patient_id,
        "fhir_coverage": {
            "resourceType":      "Coverage",
            "id":                plan["fhir_resource_id"],
            "status":            "active",
            "beneficiary":       f"Patient/{patient_id}",
            "payor":             plan["insurer"],
            "plan_name":         plan["plan_name"],
            "coverage_limit_inr":coverage_limit,
            "deductible_inr":    deductible,
            "copay_percent":     plan["copay_percent"],
            "note": "SYNTHETIC DATA - for demonstration only. Not real PHI.",
        },
        "financial_summary": {
            "total_estimated_cost_inr":  round(total_cost_inr),
            "patient_deductible_inr":    deductible,
            "insurer_pays_inr":          insurer_pays,
            "patient_out_of_pocket_inr": financial_gap,
            "coverage_sufficient":       financial_gap == 0,
        },
        "recommendation": recommendation,
        "loan_suggestion": loan_suggestion,
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 3 — Clinical Pathway Lookup
# =============================================================================

@mcp.tool()
def get_procedure_pathway(
    procedure: str,
    sub_procedure: str = None,
) -> str:
    """
    Returns the full step-by-step clinical care pathway for a procedure —
    from pre-op preparation through ICU, ward, and discharge.

    Args:
        procedure:     e.g. "Knee Replacement", "CABG - Bypass Surgery"
        sub_procedure: Optional sub-procedure for more specific pathway
    """
    steps = get_clinical_pathway(procedure, sub_procedure)

    if not steps:
        return json.dumps({
            "success": False,
            "message": f"No clinical pathway found for '{procedure}'.",
        }, indent=2)

    phases = list(dict.fromkeys(s["phase"] for s in steps if s.get("phase")))

    return json.dumps({
        "success":          True,
        "procedure":        procedure,
        "total_steps":      len(steps),
        "phases":           phases,
        "timeline_summary": f"{steps[0].get('timeline','')} -> {steps[-1].get('timeline','')}",
        "pathway":          steps,
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 4 — Hospital Lookup by Name  [UPDATED: + enriched review data]
# =============================================================================

@mcp.tool()
def lookup_hospital(hospital_name: str, specialty_code: str = None) -> str:
    """
    Looks up a hospital by name and returns its tier, specialities,
    accreditation, rating, contact details — AND enriched VADER review
    intelligence (specialty scores, top doctors, risk flags) when
    specialty_code is provided.

    Args:
        hospital_name:  Full or partial name e.g. "Apollo", "Medanta"
        specialty_code: Optional specialty code for review enrichment
                        e.g. "ORT", "CAR", "OPH", "NEU", "GAS", "URO",
                             "ONC", "GYN", "CTS", "SUR", "NPH", "EDO"
    """
    result = get_hospital_by_name(hospital_name)

    if not result:
        return json.dumps({
            "success": False,
            "message": f"Hospital '{hospital_name}' not found. Try a shorter search term.",
        }, indent=2)

    hospital_info = {
        "name":          result.get("name"),
        "address":       result.get("address"),
        "city":          result.get("city"),
        "tier":          result.get("hospital_type") or result.get("New_Types"),
        "specialities":  result.get("unified_specialities_updated") or result.get("unified_specialities"),
        "accreditation": result.get("accreditation_type"),
        "google_rating": result.get("google_rating"),
        "google_reviews":result.get("google_reviews"),
        "phone":         _resolve(result, "reviews_phone",   "phone"),
        "website":       _resolve(result, "reviews_website", "website"),
    }

    # ── Enriched review intelligence ─────────────────────────────────────────
    review_intelligence = None
    if specialty_code:
        enriched = enrich_hospital(
            hospital_name=result.get("name", hospital_name),
            spec_code=specialty_code.upper(),
            scores_db=get_hospital_specialty_scores,
        )
        review_intelligence = {
            "specialty_code":         specialty_code.upper(),
            "specialty_label":        SPEC_LABEL.get(specialty_code.upper(), specialty_code),
            "specialty_score":        enriched["spec_score"],
            "mention_count":          enriched["mention_count"],
            "match_quality":          enriched["specialty_match_quality"],
            "strength_tags":          enriched["strength_tags"],
            "risk_flags":             enriched["risk_flags"],
            "top_doctors_from_reviews": enriched["top_doctors"],
        }

    return json.dumps({
        "success":             True,
        "hospital":            hospital_info,
        "review_intelligence": review_intelligence,
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 5 — Find Specialty Hospitals  [NEW — Priority 1]
# =============================================================================

@mcp.tool()
def find_specialty_hospitals(
    city: str,
    specialty_code: str,
    user_lat: float = None,
    user_lng: float = None,
    limit: int = 5,
) -> str:
    """
    Returns top hospitals in a city for a given specialty, ranked by a
    6-factor composite score: VADER review score (35%), Google rating (20%),
    distance (15%), NABH accreditation (8%), doctor quality (10%),
    risk penalty (10%). Top 2 slots are filled by patient review data;
    remaining 3 by market alignment score.

    This is the core HOSPITAL_SUGGEST and GENERAL_HEALTH_ADVICE engine.

    Args:
        city:           City name — delhi, mumbai, bangalore, indore, bhopal,
                        nagpur, jaipur, lucknow, dehradun
        specialty_code: Specialty code — ORT (Orthopedic), CAR (Cardiology),
                        OPH (Ophthalmology), NEU (Neurology), GAS (Gastro),
                        URO (Urology), ONC (Oncology), GYN (Gynecology),
                        CTS (Cardiac Surgery), SUR (Surgery), NPH (Nephrology),
                        PUL (Pulmonology), MED (General Medicine),
                        EDO (Endocrinology), EMD (Emergency)
        user_lat:       User latitude for distance scoring (optional)
        user_lng:       User longitude for distance scoring (optional)
        limit:          Max hospitals to return (default 5)
    """
    if not city or not specialty_code:
        return json.dumps({
            "success": False,
            "message": "Both city and specialty_code are required.",
        }, indent=2)

    hospitals = _get_hospitals_by_specialty(
        specialty_code=specialty_code,
        city=city,
        user_lat=user_lat,
        user_lng=user_lng,
        limit=limit,
    )

    if not hospitals:
        return json.dumps({
            "success": False,
            "city":    city,
            "specialty_code": specialty_code,
            "message": f"No hospitals found for {specialty_code} in {city}.",
        }, indent=2)

    # Aggregate top doctors across all ranked hospitals
    top_doctors = get_top_doctors_for_specialty(hospitals, specialty_code.upper())

    return json.dumps(clean_nans({
        "success":           True,
        "city":              city,
        "specialty_code":    specialty_code.upper(),
        "specialty_label":   SPEC_LABEL.get(specialty_code.upper(), specialty_code),
        "total_returned":    len(hospitals),
        "hospitals":         hospitals,
        "recommended_doctors": top_doctors,
        "ranking_method": (
            "Top 2: VADER patient review score (35%) + Google rating (20%) + distance (15%) "
            "+ NABH (8%) + doctor quality (10%) + risk penalty (10%). "
            "Bottom 3: Market Alignment Score fallback."
        ),
    }), indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 6 — Get Top Doctors  [NEW — Priority 1]
# =============================================================================

@mcp.tool()
def get_top_doctors(
    city: str,
    specialty_code: str,
    user_lat: float = None,
    user_lng: float = None,
) -> str:
    """
    Returns up to 3 top doctors for a specialty in a city, extracted from
    patient reviews (VADER sentiment). Each doctor entry includes:
    name, hospital, tier, phone, mention count, sentiment score, and
    a plain-language reason for the recommendation.

    Deduplicates by prefix and last-name matching to avoid duplicates
    like "Dr. Somani" and "Dr. Vinod Somani" appearing separately.

    Args:
        city:           City name (delhi, mumbai, bangalore, etc.)
        specialty_code: Specialty code — ORT, CAR, OPH, NEU, GAS, URO,
                        ONC, GYN, CTS, SUR, NPH, PUL, MED, EDO, EMD
        user_lat:       Optional user latitude
        user_lng:       Optional user longitude
    """
    hospitals = _get_hospitals_by_specialty(
        specialty_code=specialty_code,
        city=city,
        user_lat=user_lat,
        user_lng=user_lng,
        limit=10,  # wider pool for doctor extraction
    )

    if not hospitals:
        return json.dumps({
            "success": False,
            "message": f"No hospitals found for {specialty_code} in {city} to extract doctors from.",
        }, indent=2)

    doctors = get_top_doctors_for_specialty(hospitals, specialty_code.upper())

    if not doctors:
        return json.dumps({
            "success":        False,
            "city":           city,
            "specialty_code": specialty_code.upper(),
            "specialty_label":SPEC_LABEL.get(specialty_code.upper(), specialty_code),
            "message": "No doctors found in patient reviews for this specialty and city.",
        }, indent=2)

    return json.dumps({
        "success":        True,
        "city":           city,
        "specialty_code": specialty_code.upper(),
        "specialty_label":SPEC_LABEL.get(specialty_code.upper(), specialty_code),
        "total_found":    len(doctors),
        "doctors":        doctors,
        "data_source":    "Extracted from patient reviews using VADER sentiment analysis",
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 7 — Find Hospitals for Procedure  [NEW — Priority 2]
# =============================================================================

@mcp.tool()
def find_hospitals_for_procedure(
    procedure: str,
    city: str,
    room_type: str = "SEMI_PRIVATE",
    comorbidities: list = None,
    age: int = None,
    gender: str = None,
    budget_inr: int = None,
    user_lat: float = None,
    user_lng: float = None,
) -> str:
    """
    Returns procedure-ranked hospitals across all tiers in a city with:
    - Cost estimate per hospital
    - Distance, NABH status, Google rating
    - Confidence score v2 (speciality match 40% + rating + distance + accreditation)
    - Optional budget filter (returns hospitals where cost_min <= budget)

    Use this when the user has a specific procedure in mind and wants to
    compare costs and options across different hospital tiers.

    Args:
        procedure:     Exact procedure name e.g. "Knee Replacement",
                       "CABG - Bypass Surgery", "Cataract Surgery"
        city:          City name (delhi, mumbai, bangalore, indore, etc.)
        room_type:     "GENERAL" | "SEMI_PRIVATE" | "PRIVATE" (default: SEMI_PRIVATE)
        comorbidities: Optional list e.g. ["diabetes", "hypertension"]
        age:           Patient age for risk-adjusted cost
        gender:        "male" | "female" | None
        budget_inr:    Optional budget in INR — filters to affordable hospitals
        user_lat:      User latitude for distance calculation
        user_lng:      User longitude for distance calculation
    """
    comorbidities = comorbidities or []

    lat, lng = CITY_COORDS.get(city.strip().lower(), (22.7196, 75.8577))
    eff_lat = float(user_lat) if user_lat else lat
    eff_lng = float(user_lng) if user_lng else lng

    hosp_result = estimate_for_hospitals(
        procedure=procedure,
        city=city,
        user_lat=eff_lat,
        user_lng=eff_lng,
        room_type=room_type,
        comorbidities=comorbidities,
        age=age,
        gender=gender,
    )

    if "error" in hosp_result:
        return json.dumps({
            "success": False,
            "message": hosp_result["error"],
        }, indent=2)

    hospitals_simple = []
    for h in hosp_result.get("top_hospitals", []):
        grand = h["cost_estimate"].get("grand_total", {})
        hospitals_simple.append({
            "name":             h["hospital_name"],
            "address":          h.get("address", "N/A"),
            "tier":             h["hospital_tier"],
            "google_rating":    h["google_rating"],
            "google_reviews":   h.get("google_reviews", 0),
            "distance_km":      h["distance_km"],
            "cost_range":       grand.get("formatted", "N/A"),
            "cost_min_inr":     grand.get("min", 0),
            "cost_max_inr":     grand.get("max", 0),
            "phone":            h["phone"],
            "website":          h.get("website", "N/A"),
            "nabh":             h["nabh_status"],
            "confidence_score": h["confidence"]["score"],
            "confidence_label": h["confidence"]["label"],
            "speciality_match": h["speciality_match"],
        })

    # ── Budget filter ─────────────────────────────────────────────────────────
    budget_filter_applied  = False
    budget_filter_message  = None
    if budget_inr:
        affordable = [h for h in hospitals_simple if h.get("cost_min_inr", 0) <= budget_inr]
        if len(affordable) >= 2:
            hospitals_simple       = affordable
            budget_filter_applied  = True
            budget_filter_message  = (
                f"Showing {len(affordable)} hospital(s) within ₹{budget_inr:,} budget."
            )
        else:
            budget_filter_message = (
                f"No hospital found within ₹{budget_inr:,} budget. "
                "Showing all options — consider Government or Standard tier."
            )

    return json.dumps(clean_nans({
        "success":              True,
        "procedure":            procedure,
        "city":                 city,
        "room_type":            room_type,
        "total_found":          hosp_result.get("total_found", len(hospitals_simple)),
        "budget_filter_applied":budget_filter_applied,
        "budget_filter_message":budget_filter_message,
        "hospitals":            hospitals_simple,
        "selection_note":       hosp_result.get("selection_note", ""),
    }), indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 8 — Get Hospital Review Scores  [NEW — Priority 2]
# =============================================================================

@mcp.tool()
def get_hospital_review_scores(
    hospital_name: str,
    specialty_code: str = None,
) -> str:
    """
    Returns raw VADER specialty sentiment scores for a named hospital.
    Provides: specialty_score, mention_count, top doctors, risk flags per
    specialty code. If specialty_code is provided, also returns enriched
    single-specialty intelligence with strength tags and risk flags.

    Use this when you need the full review intelligence for a specific
    hospital before making a recommendation or explanation.

    Args:
        hospital_name:  Full or partial hospital name
        specialty_code: Optional — return enriched view for one specialty
                        (ORT, CAR, OPH, NEU, GAS, URO, ONC, GYN, CTS, SUR,
                         NPH, PUL, MED, EDO, EMD)
    """
    scores = get_hospital_specialty_scores(hospital_name)

    if not scores:
        return json.dumps({
            "success": False,
            "message": f"No review data found for '{hospital_name}'.",
        }, indent=2)

    # Build summary across all specialties
    all_specialties = []
    for code, data in scores.items():
        all_specialties.append({
            "specialty_code":   code,
            "specialty_label":  SPEC_LABEL.get(code, code),
            "specialty_score":  data.get("specialty_score", 0),
            "mention_count":    data.get("mention_count", 0),
            "sentiment":        data.get("sentiment", 0),
            "top_doctors":      [d["name"] for d in data.get("doctors", [])[:3]],
        })
    all_specialties.sort(key=lambda x: -x["specialty_score"])

    # Enriched single-specialty view
    enriched_view = None
    if specialty_code:
        enriched = enrich_hospital(
            hospital_name=hospital_name,
            spec_code=specialty_code.upper(),
            scores_db=get_hospital_specialty_scores,
        )
        enriched_view = {
            "specialty_code":  specialty_code.upper(),
            "specialty_label": SPEC_LABEL.get(specialty_code.upper(), specialty_code),
            **enriched,
        }

    return json.dumps({
        "success":          True,
        "hospital_name":    hospital_name,
        "all_specialties":  all_specialties,
        "enriched_specialty": enriched_view,
        "data_source":      "VADER sentiment analysis on patient reviews",
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 9 — Get Valid Options  [NEW — Priority 2]
# =============================================================================

@mcp.tool()
def get_valid_options() -> str:
    """
    Returns all valid input values for use with other tools:
      - All procedures with their sub-procedures, clinical variants, and tech types
      - Valid hospital tiers
      - Valid cities
      - Valid room types
      - Valid comorbidity codes

    Call this FIRST when you are unsure of exact procedure names, tiers, or
    cities — it prevents hallucinated input values that would cause other
    tools to fail validation.
    """
    opts = get_master_options()
    return json.dumps({
        "success":        True,
        "procedures":     opts["procedures"],
        "hospital_tiers": opts["hospital_tiers"],
        "cities":         opts["cities"],
        "room_types":     opts["room_types"],
        "comorbidities":  opts["comorbidities"],
        "specialty_codes": {
            code: label for code, label in SPEC_LABEL.items()
        },
        "usage_note": (
            "Always use exact strings from this list when calling "
            "calculate_procedure_cost, find_hospitals_for_procedure, "
            "find_specialty_hospitals, or validate_inputs."
        ),
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 10 — Get All Procedure Variants  [NEW — Priority 3]
# =============================================================================

@mcp.tool()
def get_all_procedure_variants(procedure: str) -> str:
    """
    Lists all sub-procedures, clinical variants, technology types, LOS days,
    ICU days, and base costs for a given procedure. Use this before calling
    calculate_procedure_cost to pick the right variant.

    For example, "Knee Replacement" has variants:
    TKR Unilateral / TKR Bilateral / Robotic-Assisted, each with different
    base costs and LOS.

    Args:
        procedure: Procedure name (use get_valid_options to see all procedures)
    """
    variants = get_all_variants(procedure)

    if not variants:
        return json.dumps({
            "success": False,
            "message": f"No variants found for '{procedure}'. Use get_valid_options() to see all procedures.",
        }, indent=2)

    return json.dumps({
        "success":    True,
        "procedure":  procedure,
        "total":      len(variants),
        "variants":   variants,
        "usage_note": (
            "Pass sub_procedure and variant to calculate_procedure_cost "
            "for a more precise cost estimate."
        ),
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 11 — Get Procedure Card  [NEW — Priority 3]
# =============================================================================

@mcp.tool()
def get_procedure_card(
    procedure: str,
    sub_procedure: str = None,
    variant_hint: str = None,
) -> str:
    """
    Returns raw rate-card data for a procedure: base cost, implant cost,
    tech cost, LOS days, ICU days, room charges per type, ICD-10 codes,
    and clinical notes. Useful when agents need raw pricing data before
    applying tier/city multipliers themselves.

    Args:
        procedure:     Exact procedure name
        sub_procedure: Optional sub-procedure name
        variant_hint:  Optional variant keyword (e.g. "Robotic", "Bilateral")
    """
    card = get_card(procedure, sub_procedure, variant_hint)

    if not card:
        return json.dumps({
            "success": False,
            "message": f"No rate card found for '{procedure}'. Use get_valid_options() to see all procedures.",
        }, indent=2)

    return json.dumps({
        "success": True,
        "card":    card,
        "notes": {
            "base":    "Variable column — tier multipliers apply to this",
            "implant": "Fixed (NPPA capped) — no tier markup",
            "tech":    "Fixed (device cost) — no tier markup",
            "total":   "BASE + IMPLANT + TECH reference (no GST or contingency)",
        },
    }, indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 12 — Estimate All Variants Cost  [NEW — Priority 4]
# =============================================================================

@mcp.tool()
def estimate_all_variants_cost(
    procedure: str,
    hospital_tier: str,
    city: str,
    room_type: str = "SEMI_PRIVATE",
    comorbidities: list = None,
    age: int = None,
    gender: str = None,
) -> str:
    """
    Runs cost estimates across ALL sub-procedure variants for a single
    procedure + tier + city combination. Returns a comparison list sorted
    by cost (ascending).

    Use this when the user asks "what's the cheapest variant of X" or
    "compare options for Knee Replacement."

    Args:
        procedure:     Exact procedure name (e.g. "Knee Replacement")
        hospital_tier: Hospital tier (e.g. "Advanced Multispecialty")
        city:          City name (e.g. "delhi")
        room_type:     "GENERAL" | "SEMI_PRIVATE" | "PRIVATE"
        comorbidities: Optional list of conditions
        age:           Patient age
        gender:        "male" | "female" | None
    """
    comorbidities = comorbidities or []

    results = estimate_all_variants(
        procedure=procedure,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        comorbidities=comorbidities,
        age=age,
        gender=gender,
    )

    if not results or (isinstance(results, dict) and "error" in results):
        return json.dumps({
            "success": False,
            "message": (
                results.get("error", f"No variants found for '{procedure}'.")
                if isinstance(results, dict) else f"No variants found for '{procedure}'."
            ),
        }, indent=2)

    # Sort by cost min ascending
    if isinstance(results, list):
        results_sorted = sorted(
            results,
            key=lambda x: x.get("grand_total", {}).get("min", 0) if isinstance(x, dict) else 0
        )
    else:
        results_sorted = results

    return json.dumps(clean_nans({
        "success":      True,
        "procedure":    procedure,
        "hospital_tier":hospital_tier,
        "city":         city,
        "room_type":    room_type,
        "total_variants": len(results_sorted) if isinstance(results_sorted, list) else 1,
        "variants_by_cost": results_sorted,
        "usage_note": "Sorted cheapest to most expensive. Pass sub_procedure + variant to calculate_procedure_cost for the full itemized bill.",
    }), indent=2, ensure_ascii=False)


# =============================================================================
# TOOL 13 — Validate Inputs  [NEW — Priority 5]
# =============================================================================

@mcp.tool()
def validate_inputs(
    procedure: str,
    hospital_tier: str,
    city: str,
    room_type: str = "SEMI_PRIVATE",
    sub_procedure: str = None,
    variant: str = None,
    comorbidities: list = None,
) -> str:
    """
    Validates procedure, sub-procedure, variant, hospital tier, city,
    room type, and comorbidities before calling cost tools. Returns a
    structured list of errors (if any). Call this before
    calculate_procedure_cost to ensure inputs are valid and avoid
    confusing error messages.

    Args:
        procedure:     Procedure name to validate
        hospital_tier: Hospital tier to validate
        city:          City to validate
        room_type:     Room type to validate (default: SEMI_PRIVATE)
        sub_procedure: Optional sub-procedure to validate
        variant:       Optional variant to validate against sub-procedure
        comorbidities: Optional list of comorbidities to validate
    """
    comorbidities = comorbidities or []

    result = validate_user_inputs(
        procedure=procedure,
        sub_procedure=sub_procedure,
        variant=variant,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        comorbidities=comorbidities,
    )

    opts = get_master_options()

    if result["ok"]:
        return json.dumps({
            "success": True,
            "valid":   True,
            "message": "All inputs are valid. You can now call calculate_procedure_cost.",
            "validated": {
                "procedure":     procedure,
                "sub_procedure": sub_procedure,
                "variant":       variant,
                "hospital_tier": hospital_tier,
                "city":          city,
                "room_type":     room_type,
                "comorbidities": comorbidities,
            },
        }, indent=2)
    else:
        return json.dumps({
            "success":          False,
            "valid":            False,
            "errors":           result["errors"],
            "fix_suggestions": {
                "valid_procedures": list(opts["procedures"].keys()),
                "valid_tiers":      opts["hospital_tiers"],
                "valid_cities":     opts["cities"],
                "valid_room_types": opts["room_types"],
                "valid_comorbidities": opts["comorbidities"],
            },
            "message": (
                f"{len(result['errors'])} validation error(s) found. "
                "Fix them using the suggestions above, then retry."
            ),
        }, indent=2)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.routing import Mount

    port = int(os.environ.get("PORT", 8000))
    print(f"Healthcare MCP Server v2.0 starting on port {port}...")
    print("Transport: SSE")
    print(f"Listening on http://0.0.0.0:{port}")
    print("All patient data is SYNTHETIC - no real PHI used")
    print("-" * 60)

    class FixHostMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.scope["headers"] = [
                (k, v) for k, v in request.scope["headers"]
                if k.lower() != b"host"
            ] + [(b"host", b"localhost")]
            return await call_next(request)

    sse_app = mcp.sse_app()
    final_app = Starlette(routes=[Mount("/", app=sse_app)])
    final_app.add_middleware(FixHostMiddleware)

    uvicorn.run(final_app, host="0.0.0.0", port=port)
