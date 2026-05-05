"""
=============================================================================
HEALTHCARE CHATBOT — POONAWALLA FINCORP
=============================================================================
Features:
  - Intent classification (Symptom / Procedure Cost / Hospital / General)
  - In-memory session management (conversation history)
  - Backend integration (check.py) — ZERO hallucination on costs/hospitals
  - Structured response format
  - Hinglish support
=============================================================================
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import uvicorn
from datetime import datetime

from llm_utils import SYSTEM_PROMPT, call_llm

from check import (
    estimate,
    estimate_for_hospitals,
    get_personalized_bill,
    get_master_options,
    get_clinical_pathway,
    get_hospital_by_name,
    get_hospital_specialty_scores,
)


# =============================================================================
# IN-MEMORY SESSION STORE
# =============================================================================
sessions: dict = {}

def get_or_create_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "context": {
                "procedure": None,
                "city": None,
                "room_type": "SEMI_PRIVATE",
                "comorbidities": [],
                "user_lat": None,
                "user_lng": None,
                "budget_inr": None,
                "age": None,
                "gender": None,
            },
            "created_at": datetime.now().isoformat(),
        }
    return sessions[session_id]

def update_context(session: dict, llm_data: dict):
    """Merge new LLM extracted values into session context (only overwrite non-null)"""
    ctx = session["context"]
    for key in ["procedure", "city", "room_type", "user_lat", "user_lng", "budget_inr", "age", "gender"]:
        val = llm_data.get(key)
        if val and str(val).lower() not in ["null", "none", ""]:
            ctx[key] = val
    if llm_data.get("comorbidities"):
        ctx["comorbidities"] = llm_data["comorbidities"]
    specialty_code = llm_data.get("specialty_code")
    if specialty_code and str(specialty_code).lower() not in ["null", "none", ""]:
        ctx["specialty_code"] = specialty_code


# =============================================================================
# BACKEND DISPATCHER
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

# ── Specialty label map ────────────────────────────────────────────────────
SPEC_LABEL = {
    "ORT": "Orthopedic",     "CAR": "Cardiology",        "OPH": "Ophthalmology",
    "NEU": "Neurology",      "GAS": "Gastroenterology",  "URO": "Urology",
    "ONC": "Oncology",       "GYN": "Gynecology",        "CTS": "Cardiac Surgery",
    "SUR": "Surgery",        "NPH": "Nephrology",        "PUL": "Pulmonology",
    "MED": "General Medicine","EDO": "Endocrinology",    "EMD": "Emergency",
}

# ── Review-score fallback map ───────────────────────────────────────────────
# When specialty_scores.json has no/sparse data for a code, use this proxy code
# instead of falling back to ALL city hospitals (which lets dental clinics appear).
# Audit-verified across all 9 cities (bangalore/bhopal/dehradun/delhi/indore/
# jaipur/lucknow/mumbai/nagpur).
SPEC_REVIEW_FALLBACK = {
    "CTS": "CAR",   # Cardiac Surgery  → Cardiology (NONE: indore/lucknow/dehradun)
    "PUL": "MED",   # Pulmonology      → General Medicine (NONE: indore)
    "EDO": "MED",   # Endocrinology    → General Medicine (NONE: bhopal/dehradun)
    "ONC": "CAR",    # Oncology         → Surgery (NONE: dehradun)
    "NPH": "URO",   # Nephrology       → Urology (NONE: nagpur)
}

def _norm_name(name: str) -> str:
    """Normalize hospital name for dedup."""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())

_CONTACT_BAD = {"", "NA", "N/A", "nan", "None", "none", "NaN"}

def _resolve(row_or_dict, primary_col: str, fallback_col: str, default: str = "N/A") -> str:
    """
    Read contact info with priority: primary_col (e.g. reviews_phone) first,
    fallback_col (e.g. phone) second, default third.
    Works on both pandas Series rows and plain dicts.
    """
    val = str(row_or_dict.get(primary_col) or "").strip()
    if not val or val in _CONTACT_BAD:
        val = str(row_or_dict.get(fallback_col) or "").strip()
    if not val or val in _CONTACT_BAD:
        val = default
    return val


# Generic words to skip during partial-name matching
_HOSP_SKIP = {
    "hospital", "clinic", "multispecialty", "multi", "pvt", "ltd", "the",
    "of", "and", "a", "unit", "centre", "center", "care", "health",
    "medical", "research", "speciality", "specialty", "super", "institute",
}

def resolve_hospital_reference(message: str, ctx: dict):
    """
    If the user references a previously suggested hospital by:
      - number/ordinal: "1", "first", "2nd", "hospital 3"
      - partial name:   "arthros", "jain", "sahaj"
    Returns the matching hospital dict from ctx["last_suggested_hospitals"], or None.
    """
    last = ctx.get("last_suggested_hospitals", [])
    if not last:
        return None

    msg = message.lower().strip()

    # Ordinal / number references
    ORDINALS = {
        "1": 0, "first": 0, "1st": 0,
        "2": 1, "second": 1, "2nd": 1,
        "3": 2, "third": 2, "3rd": 2,
        "4": 3, "fourth": 3, "4th": 3,
        "5": 4, "fifth": 4, "5th": 4,
    }
    for word in msg.split():
        if word in ORDINALS:
            idx = ORDINALS[word]
            if idx < len(last):
                return last[idx]

    # Partial name match — score by significant keyword hits
    best_score, best_hosp = 0, None
    for h in last:
        h_words = [
            w for w in h["name"].lower().split()
            if w not in _HOSP_SKIP and len(w) > 2
        ]
        score = sum(1 for w in h_words if w in msg)
        if score > best_score:
            best_score = score
            best_hosp = h

    return best_hosp if best_score > 0 else None

def enrich_hospital(hospital_name: str, spec_code: str, scores_db) -> dict:
    """Pull review intelligence for one hospital + specialty.
    Falls back to SPEC_REVIEW_FALLBACK code if primary has no data.
    """
    hosp_scores = scores_db(hospital_name)
    # Primary lookup
    spec_data   = hosp_scores.get(spec_code, {}) if spec_code else {}
    # Fallback if primary has no meaningful data
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


def build_reasons(h: dict, spec_code: str) -> list:
    """Generate plain-language reasons why this hospital was ranked here."""
    reasons = []
    label = SPEC_LABEL.get(spec_code, "")
    mq    = h.get("specialty_match_quality", "")

    if mq == "Strong":
        reasons.append(f"Highly reviewed for {label} — review score: {h.get('specialty_sentiment', 0)}")
    elif mq == "Moderate":
        mc = h.get("mention_count", 0)
        reasons.append(f"Decent {label} review signal ({mc} patient mention{'s' if mc != 1 else ''})")
    elif mq == "Weak":
        reasons.append(f"Limited {label} review data — ranked primarily by Google rating")
    elif mq == "No review data":
        reasons.append("Ranked by Google rating — no specialty review data available")

    if h.get("top_doctors"):
        d = h["top_doctors"][0]
        reasons.append(f"{d['name']} ({d['specialty_label']}) mentioned {d['mentions']}x in patient reviews")

    if h.get("nabh") and h["nabh"] not in ["N/A", "None", "", "nan"]:
        reasons.append("NABH accredited hospital")

    dist = h.get("distance_km")
    if dist and dist < 100:
        reasons.append(f"{dist} km from your location")

    for r in h.get("risk_flags", []):
        reasons.append(f"⚠️ Note: {r}")

    return reasons


def get_top_doctors_for_specialty(hospitals: list, spec_code: str) -> list:
    """
    Aggregate top doctors across all ranked hospitals for a specialty.
    Pulls directly from specialty_scores.json via get_hospital_specialty_scores.
    Deduplicates by both prefix AND last-name matching.
    Returns up to 3 doctors with hospital, rating, phone, sentiment.
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
                    "reason":          (
                        f"Most mentioned {label} specialist in patient reviews"
                        if mentions >= 3
                        else f"Mentioned {mentions}x in patient reviews for {label}"
                    ),
                }

    # ── Dedup pass 1: prefix match ("Dr. Vinay" inside "Dr. Vinay Tantuway") ──
    all_keys   = list(seen.keys())
    to_remove  = set()
    for k in all_keys:
        for other in all_keys:
            if k != other and other.startswith(k + " "):
                to_remove.add(k)

    # ── Dedup pass 2: last-name match ("Dr. Somani" == "Dr. Vinod Somani") ──
    def last_name(key: str) -> str:
        parts = key.strip().split()
        return parts[-1] if parts else key

    last_name_map = {}   # last_name -> key with most mentions
    for k, v in seen.items():
        if k in to_remove:
            continue
        ln = last_name(k)
        if ln not in last_name_map:
            last_name_map[ln] = k
        else:
            existing_key = last_name_map[ln]
            # keep the entry with more mentions; on tie, keep longer (more specific) name
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
    return ranked list with enriched review intelligence.

    FIX 1: Deduplication by normalized name
    FIX 2: Strict specialty filter with fallback
    FIX 3: Distance clamping (>100km = bad GPS data)
    FIX 4: Priority ordering: Strong > Moderate > Weak > No data
    """
    try:
        from check import _load_hosp, haversine, clean_nans
        df = _load_hosp()

        city_cl = city.strip().lower() if city else ""
        if not city_cl:
            return []

        # Filter by city
        city_df = df[df["city"].str.lower().str.strip() == city_cl] if "city" in df.columns else df
        if city_df.empty:
            city_df = df[df["city"].str.lower().str.contains(city_cl, na=False)]
        if city_df.empty:
            return []

        # Use specialty_scores.json (review data) as the ONLY specialty filter.
        # unified_specialities column is unreliable (missing codes like CTS, OPH, GYN).
        # Strategy: score ALL city hospitals, then surface those with spec_score > 0 first.
        spec_code_up = specialty_code.upper() if specialty_code else ""

        # No pre-filter — let composite scoring sort everything
        filtered_df = city_df


        # Reference coords
        lat, lng = (
            (float(user_lat), float(user_lng))
            if user_lat and user_lng
            else CITY_COORDS.get(city_cl, (22.7196, 75.8577))
        )

        eff_code = SPEC_REVIEW_FALLBACK.get(spec_code_up, spec_code_up) if spec_code_up else ""

        # FIX 1: Build hospitals list with deduplication
        seen_names = set()
        hospitals = []
        for _, row in filtered_df.iterrows():
            try:
                raw_name = str(row.get("hospital_name", row.get("name", "N/A"))).strip()
                norm = _norm_name(raw_name)
                if norm in seen_names:
                    continue
                seen_names.add(norm)

                h_lat = float(row.get("reviews_lat") or row.get("lat") or 0)
                h_lng = float(row.get("reviews_lng") or row.get("lng") or 0)
                dist  = round(haversine(lat, lng, h_lat, h_lng), 1) if h_lat and h_lng else None

                # FIX 3: clamp bad GPS data
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

        # ── 6-factor composite scoring ──────────────────────────────────────
        specialty_scores_db = get_hospital_specialty_scores

        for h in hospitals:
            enriched = enrich_hospital(h["name"], spec_code_up, specialty_scores_db)
            
            # If primary review is weak/zero, try fallback review code
            if enriched["spec_score"] <= 0 and eff_code and eff_code != spec_code_up:
                enriched_fb = enrich_hospital(h["name"], eff_code, specialty_scores_db)
                if enriched_fb["spec_score"] > 0:
                    enriched = enriched_fb

            spec_str = h["specialities"].upper()
            has_official_spec = (spec_code_up in spec_str) or (eff_code and eff_code in spec_str)
            mas_norm = h["mas"] / 10.0 if has_official_spec else 0.0

            rating_norm = h["rating"] / 5.0 if h["rating"] else 0

            if h["distance_km"] and h["distance_km"] > 0:
                dist_score = max(0.0, 1.0 - (h["distance_km"] / 30.0))
            else:
                dist_score = 0.5

            nabh_bonus  = 0.08 if (h.get("nabh") and h["nabh"] not in ["N/A", "None", "", "nan"]) else 0.0
            doc_quality = 0.0
            if enriched["top_doctors"]:
                doc_quality = max(0, enriched["top_doctors"][0]["sentiment"]) * 0.5

            risk_penalty = len(enriched["risk_flags"]) * 0.05

            mq = enriched["specialty_match_quality"]
            spec_score = enriched["spec_score"]

            # ── Blend Review Data with Database MAS Data ──
            if spec_score <= 0.1 and has_official_spec and h["mas"] >= 5.0:
                # Upgrade visually if it's a highly aligned hospital with official specialty
                if h["mas"] >= 8.0:
                    mq = "Market Aligned"
                elif mq == "No review data":
                    mq = "Market Aligned (Secondary)"
                
                # Use MAS as proxy for spec_score
                spec_score = mas_norm * 0.85
                
                # Add a strength tag explaining why it's listed
                label = SPEC_LABEL.get(spec_code_up, spec_code_up)
                enriched["strength_tags"].append(f"Top-rated for {label} based on market data")

            # FIX 4: priority tier multiplier ensures Strong > Moderate > Weak > No data
            mq_boost = {"Strong": 0.20, "Moderate": 0.10, "Weak": 0.02, "Market Aligned": 0.15, "Market Aligned (Secondary)": 0.08, "No review data": 0.0}.get(mq, 0.0)

            h["composite_score"] = (
                rating_norm                    * 0.20 +
                spec_score                     * 0.35 +
                dist_score                     * 0.15 +
                nabh_bonus                           +
                doc_quality                    * 0.10 +
                max(0, 0.10 - risk_penalty)          +
                mq_boost                             +
                (mas_norm * 0.15)  # General bonus for top-tier hospitals
            )

            h["specialty_match_quality"] = mq
            h["strength_tags"]           = enriched["strength_tags"]
            h["risk_flags"]              = enriched["risk_flags"]
            h["top_doctors"]             = enriched["top_doctors"]
            h["mention_count"]           = enriched["mention_count"]
            h["specialty_sentiment"]     = round(spec_score, 3)
            h["review_doctors"]          = [d["name"] for d in enriched["top_doctors"]]

        # NEW — Top 2 VADER, Bottom 3 MAS
        reviewed = [h for h in hospitals 
                    if h["specialty_match_quality"] in ("Strong", "Moderate", "Weak")
                    and h["specialty_sentiment"] > 0]
        market   = [h for h in hospitals 
                    if h["specialty_match_quality"] in ("Market Aligned", 
                                                         "Market Aligned (Secondary)", 
                                                         "No review data")]

        reviewed.sort(key=lambda h: -h["composite_score"])
        market.sort(key=lambda h: -h["composite_score"])

        top_2 = []
        used = set()
        for h in reviewed:
            if len(top_2) >= 2: break
            if h["name"] not in used:
                h["_source"] = "reviews"
                top_2.append(h)
                used.add(h["name"])

        top_3 = []
        for h in market + reviewed:   # market first, then reviewed leftovers
            if len(top_3) >= (limit - len(top_2)): break
            if h["name"] not in used:
                h["_source"] = h.get("_source", "market")
                top_3.append(h)
                used.add(h["name"])

        top = top_2 + top_3
        
        for h in top:
            h["reasons"] = build_reasons(h, spec_code_up)

        return top

    except Exception as e:
        return []


def call_backend(backend_type: str, ctx: dict) -> dict:
    """
    Call check.py functions based on what's needed.
    Returns structured backend data — ZERO hallucination.
    """
    procedure          = ctx.get("procedure")
    city               = ctx.get("city")
    room_type          = ctx.get("room_type", "SEMI_PRIVATE")
    comorbidities      = ctx.get("comorbidities", [])
    age                = ctx.get("age")
    gender             = ctx.get("gender")
    user_lat           = ctx.get("user_lat")
    user_lng           = ctx.get("user_lng")
    hospital_type_filter = ctx.get("hospital_type_filter")
    specialty_code     = ctx.get("specialty_code")

    result = {}
    
    tier_to_use = hospital_type_filter if hospital_type_filter else "Advanced Multispecialty"

    if backend_type in ["cost", "both"]:
        bill = get_personalized_bill(
            procedure=procedure, sub_procedure=None, variant=None,
            hospital_tier=tier_to_use,
            city=city, room_type=room_type, comorbidities=comorbidities,
            age=age, gender=gender,
        )
        if bill.get("ok"):
            r = bill["result"]
            result["cost"] = {
                "grand_total":      r["grand_total"]["formatted"],
                "recommended_loan": r["loan_relevant_amount"]["recommended_loan"],
                "summary":          r.get("summary", {}),
                "clinical_notes":   r.get("clinical_notes", ""),
                "disclaimer":       r.get("disclaimer", ""),
                # ── Assumption context (shown on UI so user knows what was estimated) ──
                "estimate_context": {
                    "procedure":     procedure or "N/A",
                    "city":          (city or "N/A").title(),
                    "room_type":     room_type or "SEMI_PRIVATE",
                    "hospital_tier": tier_to_use,
                },
            }
            result["full_result"] = r

            # ── Budget check ───────────────────────────────────────────────────
            budget = ctx.get("budget_inr")
            if budget:
                cost_min = r["grand_total"].get("min", 0)
                cost_max = r["grand_total"].get("max", 0)
                if cost_max <= budget:
                    status = "within"
                    msg    = f"✅ {tier_to_use} ka estimate ₹{cost_max:,} tak hai, jo aapke budget ₹{budget:,} mein hai."
                elif cost_min <= budget < cost_max:
                    status = "borderline"
                    msg    = (
                        f"⚠️ {tier_to_use} ka minimum estimate ₹{cost_min:,} budget mein hai, "
                        f"lekin max ₹{cost_max:,} tak ja sakta hai. "
                        "General room consider karein."
                    )
                else:
                    status = "over"
                    msg    = (
                        f"❌ {tier_to_use} mein estimated cost ₹{cost_min:,}–₹{cost_max:,} hai, "
                        f"jo aapke budget ₹{budget:,} se zyada hai. "
                        "Neeche ke tier ka hospital ya Government hospital try karein."
                    )
                result["cost"]["budget_check"] = {
                    "budget_inr": budget,
                    "status":     status,
                    "message":    msg,
                }
        else:
            result["cost_error"] = bill.get("errors", ["Unknown error"])

    if backend_type in ["hospitals", "both"]:
        hospital_types = [hospital_type_filter] if hospital_type_filter else None
        lat, lng = CITY_COORDS.get(city.lower(), (22.7196, 75.8577))
        hosp_result = estimate_for_hospitals(
            procedure=procedure, city=city,
            user_lat=float(user_lat) if user_lat else lat,
            user_lng=float(user_lng) if user_lng else lng,
            room_type=room_type, comorbidities=comorbidities,
            hospital_types=hospital_types,
            age=age, gender=gender,
        )
        if "error" not in hosp_result:
            available_types = list({h["hospital_tier"] for h in hosp_result.get("top_hospitals", [])})
            hospitals_simple = []
            for h in hosp_result.get("top_hospitals", []):
                grand = h["cost_estimate"].get("grand_total", {})
                hospitals_simple.append({
                    "name":             h["hospital_name"],
                    "tier":             h["hospital_tier"],
                    "rating":           h["google_rating"],
                    "distance_km":      h["distance_km"],
                    "cost_range":       grand.get("formatted", "N/A"),
                    "cost_min":         grand.get("min", 0),
                    "phone":            h["phone"],
                    "nabh":             h["nabh_status"],
                    "confidence":       h["confidence"]["label"],
                    "speciality_match": h["speciality_match"],
                })
            result["hospitals"]       = hospitals_simple
            result["available_types"] = available_types
        else:
            result["hospital_error"] = hosp_result.get("error")

        # ── Budget filter on hospital list ──────────────────────────────────────
        budget = ctx.get("budget_inr")
        if budget and result.get("hospitals"):
            affordable = [h for h in result["hospitals"] if h.get("cost_min", 0) <= budget]
            if len(affordable) >= 2:
                result["hospitals"] = affordable
                result["budget_filter_applied"] = True
                result["budget_filter_message"] = (
                    f"₹{budget:,} budget ke andar {len(affordable)} hospital(s) dikha rahe hain."
                )
            else:
                result["budget_filter_applied"] = False
                result["budget_filter_message"] = (
                    f"₹{budget:,} mein koi suitable hospital nahi mila. "
                    "Sabhi options dikha rahe hain — Government hospital ya Standard tier try karein."
                )

    if backend_type == "specialty_hospitals" and city:
        hospitals = _get_hospitals_by_specialty(
            specialty_code=specialty_code, city=city,
            user_lat=user_lat, user_lng=user_lng,
        )
        result["specialty_hospitals"] = hospitals
        # Aggregate top doctors across all hospitals
        if hospitals:
            result["recommended_doctors"] = get_top_doctors_for_specialty(
                hospitals, specialty_code.upper() if specialty_code else ""
            )

    if procedure and backend_type in ["cost", "both"]:
        pathway = get_clinical_pathway(procedure)
        if pathway:
            result["pathway_steps"] = len(pathway)
            result["key_phases"]    = list({s["phase"] for s in pathway if s["phase"]})

    return result


# =============================================================================
# RESPONSE FORMATTER
# =============================================================================

def format_response(llm_data: dict, backend_data: dict, session_id: str,
                    ctx: dict = None) -> dict:
    """
    Combine LLM reply + backend data into final structured response.
    Cards in structured_data are rendered by the HTML frontend.
    The text reply carries only the conversational intro.
    """
    reply_parts = []
    intent      = llm_data.get("intent", "")
    ctx         = ctx or {}

    # Use ctx city as fallback when LLM returns null city on follow-ups
    display_city = (llm_data.get("city") or ctx.get("city") or "").title()

    conv_reply = llm_data.get("conversational_reply", "")
    if conv_reply:
        reply_parts.append(conv_reply)

    # Pathway info (text only — no card)
    if "pathway_steps" in backend_data:
        phases = ", ".join(backend_data.get("key_phases", []))
        reply_parts.append(
            f"\n📍 **Clinical Pathway:** {backend_data['pathway_steps']} steps ({phases})"
        )

    # Disclaimer
    if backend_data or intent in ("GENERAL_HEALTH_ADVICE", "HOSPITAL_SUGGEST"):
        reply_parts.append(
            "\n⚠️ _Yeh ek advisory hai. Diagnosis ke liye doctor se zaroor milein. "
            "Kisi bhi emergency mein 112 ya nazdiki hospital jaayein._"
        )

    follow_up = llm_data.get("follow_up_question")
    if follow_up:
        reply_parts.append(f"\n❓ {follow_up}")

    response = {
        "session_id":   session_id,
        "intent":       intent,
        "procedure":    llm_data.get("procedure") or ctx.get("procedure"),
        "city":         display_city,
        "backend_used": bool(backend_data),
        "reply":        "\n".join(reply_parts),
        "structured_data": {
            "cost":                 backend_data.get("cost"),
            "hospitals":            backend_data.get("hospitals"),
            "specialty_hospitals":  backend_data.get("specialty_hospitals"),
            "recommended_doctors":  backend_data.get("recommended_doctors"),
            "doctor_only_mode":     backend_data.get("doctor_only_mode", False),
            "avoid_mode":           backend_data.get("avoid_mode", False),
            "health_advice":        llm_data.get("health_advice") if intent == "GENERAL_HEALTH_ADVICE" else None,
            "pathway_steps":        backend_data.get("pathway_steps"),
            "pathway_summary":      backend_data.get("pathway_summary"),
            "full_pathway":         backend_data.get("full_pathway"),
            "budget_filter_message": backend_data.get("budget_filter_message"),
        },
        "missing_info": llm_data.get("missing_info", []),
    }
    if "full_result" in backend_data:
        response["full_result"] = backend_data["full_result"]
    return response


# =============================================================================
# HOSPITAL LOOKUP BY NAME
# =============================================================================

def handle_hospital_lookup(llm_data: dict) -> dict:
    hospital_name = llm_data.get("hospital_name") or llm_data.get("hospital") or llm_data.get("name")
    if not hospital_name:
        return None
    details = get_hospital_by_name(hospital_name)
    if not details:
        return {
            "reply": f"Sorry, I couldn't find any hospital matching '{hospital_name}'. Please check the name and try again.",
            "structured_data": None
        }

    phone   = _resolve(details, "reviews_phone",   "phone")
    website = _resolve(details, "reviews_website", "website")

    reply  = f"\U0001f3e5 **{details.get('hospital_name', details.get('name', hospital_name))}**\n"
    reply += f"Address: {details.get('address', 'N/A')}\n"
    reply += f"Tier: {details.get('New_Types', details.get('hospital_tier', 'N/A'))}\n"
    reply += f"Rating: \u2b50{details.get('google_rating', 'N/A')} ({details.get('google_reviews', 'N/A')} reviews)\n"
    reply += f"Phone: {phone}\n"
    reply += f"Website: {website}\n"
    reply += f"NABH: {details.get('accreditation_type', 'N/A')}\n"
    return {"reply": reply, "structured_data": details}
