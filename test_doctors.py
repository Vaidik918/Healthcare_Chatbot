"""Full end-to-end simulation of doctor suggestion for Indore ORT"""
from check import get_hospital_specialty_scores

# Simulate the hospitals that _get_hospitals_by_specialty would return
mock_hospitals = [
    {'name': 'arthros multi speciality hospital', 'tier': 'Boutique Super-Specialty',
     'specialty_match_quality': 'Strong', 'specialty_sentiment': 0.759},
    {'name': 'arthros clinic (a unit of arthrogen medical (opc) pvt. ltd.)', 'tier': 'Standard Secondary General',
     'specialty_match_quality': 'Strong', 'specialty_sentiment': 0.62},
    {'name': 'jain multispecialty hospital', 'tier': 'Standard Secondary General',
     'specialty_match_quality': 'Strong', 'specialty_sentiment': 0.4},
    {'name': 'sahaj hospitals pvt. ltd.', 'tier': 'Standard Secondary General',
     'specialty_match_quality': 'Weak', 'specialty_sentiment': 0.055},
    {'name': 'varma union hospital pvt ltd', 'tier': 'Advanced Multispecialty',
     'specialty_match_quality': 'No review data', 'specialty_sentiment': 0.0},
]

spec_code = 'ORT'
label = 'Orthopedic'
seen = {}

for h in mock_hospitals:
    hosp_scores = get_hospital_specialty_scores(h['name'])
    spec_data   = hosp_scores.get(spec_code, {})
    all_docs    = spec_data.get('doctors', [])
    for doc in all_docs:
        if doc.get('avg_sentiment', 0) < 0:
            continue
        key      = doc['name'].lower().strip()
        mentions = doc.get('mentions', 1)
        sentiment= doc.get('avg_sentiment', 0)
        if key not in seen or mentions > seen[key]['mentions']:
            seen[key] = {
                'name':           doc['name'],
                'specialty_label':label,
                'mentions':       mentions,
                'sentiment':      round(sentiment, 3),
                'hospital':       h['name'],
                'hospital_tier':  h.get('tier',''),
                'confidence':     'High' if mentions>=3 else ('Medium' if mentions>=2 else 'Low'),
                'reason':         f"Most mentioned {label} specialist" if mentions>=3 else f"Mentioned {mentions}x in patient reviews for {label}",
            }

# Prefix dedup
all_keys = list(seen.keys())
to_remove = set()
for k in all_keys:
    for other in all_keys:
        if k != other and other.startswith(k + " "):
            to_remove.add(k)
deduped = {k: v for k, v in seen.items() if k not in to_remove}

sorted_docs = sorted(deduped.values(), key=lambda d: (-d['mentions'], -d['sentiment']))

print("=== DOCTOR RECOMMENDATION CARD — Orthopedic, Indore ===")
for i, d in enumerate(sorted_docs[:3], 1):
    print(f"\n{i}. {d['name']}")
    print(f"   Specialty: {d['specialty_label']}")
    print(f"   Hospital:  {d['hospital']} ({d['hospital_tier']})")
    print(f"   Mentions:  {d['mentions']}x | Sentiment: {d['sentiment']}")
    print(f"   Confidence:{d['confidence']}")
    print(f"   Reason:    {d['reason']}")
