"""
preprocess_reviews.py — FREE specialty sentiment + doctor extraction
=====================================================================
FIXES vs original:
  FIX 1 — "back" removed from ORT (82% false positive rate)
           Replaced with specific medical phrases only.
  FIX 4 — Confidence weighting added.
           Low-mention scores shrunk toward neutral instead of
           displayed at face value (only 5 reviews = high variance).

Everything else is identical to the original version.

Output: specialty_scores.json
{
  "hospital name": {
    "ORT": {
      "sentiment": 0.82, "mention_count": 3, "focus_score": 0.6,
      "confidence": 1.0, "specialty_score": 0.49,
      "doctors": [{"name": "Dr. Ankit Sharma", "mentions": 3, "avg_sentiment": 0.91}]
    }, ...
  }
}
Run: python preprocess_reviews.py [reviews.jsonl] [output.json]
"""

import json, re
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SPECIALTY_KEYWORDS = {
    "CAR": [
        "cardiology","cardiologist","cardiac","heart","angioplasty","angiography",
        "echocardiogram","ecg","ekg","bypass","heart attack","heart failure","pacemaker",
        "stent","arrhythmia","atrial fibrillation","coronary","myocardial","heart disease",
        "heart surgery","open heart","cabg","valve replacement","heart block","tachycardia",
        "bradycardia","chest pain",
    ],
    "ORT": [
        # FIX 1 — "back" removed (82% false positive: "came back", "few months back")
        # Replaced with specific medical-context phrases:
        "back pain","lower back","back problem","back surgery","back treatment","back specialist",
        # Everything else unchanged:
        "ortho","orthopedic","orthopaedic","bone","joint","fracture",
        "knee","hip","shoulder","spine","acl","ligament",
        "meniscus","arthroscopy","arthroplasty",
        "replacement surgery","knee replacement","hip replacement","shoulder replacement",
        "sports injury","sports medicine","physiotherapy for bone",
        "physiotherapy for joint","tendon","cartilage","scoliosis",
        "slip disc","disc herniation","rotator cuff","ankle","wrist fracture",
    ],
    "OPH": [
        "ophthalmology","ophthalmologist","eye","cataract","glaucoma",
        "retina","cornea","lasik","laser eye","vision","eyesight",
        "squint","macular","eye surgery","eye drops","eye care",
        "lens implant","iol","vitreous","optical","spectacles",
        "phacoemulsification","intraocular",
    ],
    "NEU": [
        "neurology","neurologist","neuro","brain","stroke","epilepsy",
        "seizure","headache","migraine","parkinson","alzheimer",
        "dementia","multiple sclerosis","nerve","spinal cord","tremor",
        "memory","neurosurgery","brain surgery","brain tumor",
        "meningitis","neuropathy","cerebral",
    ],
    "GAS": [
        "gastro","gastroenterology","gastroenterologist","stomach",
        "liver","hepatitis","cirrhosis","colonoscopy","endoscopy",
        "ibs","crohn","ulcer","acidity","gerd","acid reflux",
        "gallbladder","gallstone","pancreas","pancreatitis",
        "bowel","intestine","colon","digestion","digestive",
        "appendix","appendectomy","hernia",
    ],
    "URO": [
        "urology","urologist","kidney","urinary","bladder","prostate",
        "kidney stone","renal","dialysis","uti","urinary tract",
        "kidney failure","lithotripsy","urethra","cystoscopy","incontinence",
    ],
    "ONC": [
        "oncology","oncologist","cancer","tumor","tumour","chemotherapy",
        "chemo","radiation","radiotherapy","biopsy","malignant",
        "breast cancer","lung cancer","blood cancer","leukemia",
        "lymphoma","ovarian cancer","cervical cancer","colon cancer",
        "prostate cancer","skin cancer","palliative",
    ],
    "GYN": [
        "gynecology","gynaecology","gynecologist","gynaecologist",
        "obstetrics","delivery","c-section","caesarean","cesarean",
        "maternity","pregnancy","prenatal","postnatal","neonatal",
        "newborn","ivf","fertility","menstrual","uterus","ovary",
        "hysterectomy","fibroid","pcod","pcos","infertility",
        "labor","labour","birth","antenatal",
    ],
    "CTS": [
        "cardiothoracic","thoracic","chest surgery","lung surgery",
        "heart surgery","valve surgery","aortic","aorta","cabg",
        "bypass surgery","thoracoscopy","lobectomy","pneumonectomy",
    ],
    "SUR": [
        "surgery","surgeon","operation","laparoscopy","laparoscopic",
        "hernia repair","appendectomy","appendix removed",
        "general surgery","day surgery","minimally invasive",
        "post-op","post operative","surgical",
    ],
    "NPH": [
        "nephrology","nephrologist","kidney disease","ckd",
        "chronic kidney","renal failure","kidney transplant",
        "dialysis","haemodialysis","hemodialysis",
    ],
    "PUL": [
        "pulmonology","pulmonologist","lung","respiratory","asthma",
        "copd","bronchitis","pneumonia","tuberculosis","tb",
        "breathing","spirometry","sleep apnea","sleep apnoea",
        "bronchoscopy","cough","chest infection",
    ],
    "MED": [
        "general medicine","physician","internal medicine","fever",
        "diabetes management","sugar control","hypertension control",
        "blood pressure","general checkup","routine checkup",
        "blood test","medical examination","opd","out patient",
        "general physician","medicine department",
    ],
    "EDO": [
        "endocrinology","endocrinologist","diabetes","diabetic",
        "thyroid","insulin","hormone","adrenal","pituitary",
        "hba1c","sugar","hypothyroid","hyperthyroid","metabolic",
    ],
    "EMD": [
        "emergency","casualty","trauma","icu","intensive care",
        "critical care","life support","ambulance","resuscitation","cpr",
    ],
}

_COMPILED = {
    code: re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")(?!\w)",
        re.IGNORECASE
    )
    for code, kws in SPECIALTY_KEYWORDS.items()
}

_DR_PATTERN = re.compile(
    r"(?:Dr\.?|Doctor\.?|Prof\.?)\s+([A-Z][a-zA-Z-]{2,20}(?:\s+[A-Z][a-zA-Z-]{2,20}){0,2})"
)

_NOT_NAMES = {
    "the","a","an","is","was","has","have","said","told","advised","their",
    "this","that","who","which","when","team","staff","all","our","your",
    "his","her","my","we","you","it","its",
}

_SUFFIX_STRIP = re.compile(
    r"\s+(?:sir|mam|madam|ji|sahab|bhai|didi|md)$", re.IGNORECASE
)

def _clean_name(raw):
    name = _SUFFIX_STRIP.sub("", raw.strip())
    parts = name.split()
    if not parts or parts[0].lower() in _NOT_NAMES:
        return None
    return "Dr. " + " ".join(p.capitalize() for p in parts)

def _extract_doctors(text):
    seen, result = set(), []
    for m in _DR_PATTERN.finditer(text):
        name = _clean_name(m.group(1))
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result

def _norm(name):
    return re.sub(r"[^a-z ]", "", name.lower()).strip()

def _dedup_doctors(doctors):
    """Drop 'Dr. Vaishal' when 'Dr. Vaishal Kenia' also exists."""
    all_names = [d["name"].lower() for d in doctors]
    return [
        d for d in doctors
        if not any(
            other.startswith(d["name"].lower() + " ")
            for other in all_names
            if other != d["name"].lower()
        )
    ]

_vader = SentimentIntensityAnalyzer()

def _sentiment(text):
    # Unchanged from original — pure VADER, no star blending
    return _vader.polarity_scores(text)["compound"]

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]

def _confidence(mention_count):
    # FIX 4 — shrink low-evidence scores toward neutral
    # 0 mentions → 0.0 | 1 mention → 0.33 | 2 → 0.67 | 3+ → 1.0
    return round(min(mention_count / 3.0, 1.0), 4)


def score_hospital(hospital_name, reviews):
    total = len(reviews)
    spec_data = {c: {"mention_count": 0, "sentiments": []} for c in SPECIALTY_KEYWORDS}
    doctor_data = defaultdict(lambda: defaultdict(lambda: {"display": "", "mentions": 0, "sentiments": []}))

    for review in reviews:
        text = review.get("text", "") or ""
        star_boost = (int(review.get("rating", 3)) - 3) * 0.15   # unchanged from original
        review_doctors = _extract_doctors(text)
        review_codes = set()

        for sent in _sentences(text):
            sent_score = min(1.0, max(-1.0, _sentiment(sent) + star_boost))
            sent_doctors = _extract_doctors(sent)
            for code, pat in _COMPILED.items():
                if pat.search(sent):
                    spec_data[code]["sentiments"].append(sent_score)
                    review_codes.add(code)
                    for doc in sent_doctors:
                        key = _norm(doc)
                        rec = doctor_data[code][key]
                        if not rec["display"]:
                            rec["display"] = doc
                        rec["mentions"] += 1
                        rec["sentiments"].append(sent_score)

        for code in review_codes:
            spec_data[code]["mention_count"] += 1

        rev_score = min(1.0, max(-1.0, _sentiment(text) + star_boost))
        for doc in review_doctors:
            key = _norm(doc)
            for code in review_codes:
                rec = doctor_data[code][key]
                if not rec["display"]:
                    rec["display"] = doc
                if rec["mentions"] == 0:
                    rec["mentions"] += 1
                    rec["sentiments"].append(rev_score)

    results = {}
    for code in SPECIALTY_KEYWORDS:
        mc   = spec_data[code]["mention_count"]
        sents = spec_data[code]["sentiments"]

        avg_sent = round(sum(sents) / len(sents), 4) if sents else 0.0
        focus    = round(mc / total, 4) if total > 0 else 0.0
        conf     = _confidence(mc)                     # FIX 4

        raw_score = round(avg_sent * focus, 4)
        # FIX 4 — specialty_score is now confidence-adjusted
        specialty_score = round(raw_score * conf, 4)

        doctors = []
        for key, rec in doctor_data[code].items():
            if not rec["mentions"]:
                continue
            ds = rec["sentiments"]
            doctors.append({
                "name": rec["display"],
                "mentions": rec["mentions"],
                "avg_sentiment": round(sum(ds) / len(ds), 3) if ds else 0.0,
            })
        doctors.sort(key=lambda d: (-d["mentions"], -d["avg_sentiment"]))
        doctors = _dedup_doctors(doctors)

        results[code] = {
            "sentiment":       avg_sent,
            "mention_count":   mc,
            "focus_score":     focus,
            "confidence":      conf,
            "specialty_score": specialty_score,   # use this for ranking
            "doctors":         doctors,
        }

    return results


def run(reviews_path="reviews_output.jsonl", output_path="specialty_scores.json"):
    print("Loading reviews...")
    all_scores = {}
    skipped = processed = 0

    with open(reviews_path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} hospitals...")
    for i, line in enumerate(lines, 1):
        try:
            h = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if h.get("status") != "OK" or not h.get("reviews"):
            skipped += 1
            continue
        all_scores[h["input_name"].strip().lower()] = score_hospital(
            h["google_name"], h["reviews"]
        )
        processed += 1
        if i % 200 == 0:
            print(f"  {i}/{len(lines)} done...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = len(json.dumps(all_scores)) / 1024
    print(f"\nDone! {processed} hospitals, {skipped} skipped.")
    print(f"Output: {output_path}  ({size_kb:.0f} KB)\n")

    # ── Sanity checks ──
    print("--- FIX 1 CHECK: Arthros (ORT high, OPH zero) ---")
    for name, scores in all_scores.items():
        if "arthros" in name:
            print(f"  {name}")
            for code in ["ORT", "OPH"]:
                s = scores[code]
                print(f"  {code}: score={s['specialty_score']}  conf={s['confidence']}  "
                      f"mentions={s['mention_count']}  doctors={[d['name'] for d in s['doctors']]}")
            break

    print("\n--- FIX 1 CHECK: Eye hospital (OPH high, ORT zero) ---")
    for name, scores in all_scores.items():
        if ("eye" in name or "vision" in name) and "hospital" in name:
            print(f"  {name}")
            for code in ["OPH", "ORT"]:
                s = scores[code]
                print(f"  {code}: score={s['specialty_score']}  conf={s['confidence']}  "
                      f"mentions={s['mention_count']}  doctors={[d['name'] for d in s['doctors']]}")
            break

    print("\n--- FIX 4 CHECK: 1-mention hospital (score must be < raw) ---")
    for name, scores in all_scores.items():
        for code in ["ORT", "CAR", "OPH"]:
            s = scores[code]
            if s["mention_count"] == 1:
                raw = round(s["sentiment"] * s["focus_score"], 4)
                print(f"  {name} [{code}]: raw={raw}  conf={s['confidence']}  "
                      f"adjusted_score={s['specialty_score']}")
                break
        else:
            continue
        break

    return all_scores


if __name__ == "__main__":
    import os, sys
    rp = sys.argv[1] if len(sys.argv) > 1 else "reviews_output.jsonl"
    op = sys.argv[2] if len(sys.argv) > 2 else "specialty_scores.json"
    if not os.path.exists(rp):
        print(f"ERROR: {rp} not found.")
        sys.exit(1)
    run(reviews_path=rp, output_path=op)
