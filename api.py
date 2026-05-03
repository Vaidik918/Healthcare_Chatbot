from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uuid
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from chatbot import (
    get_or_create_session,
    update_context,
    call_backend,
    format_response,
    handle_hospital_lookup,
    resolve_hospital_reference,
    SYSTEM_PROMPT,
    call_llm,
    CITY_COORDS
)


from check import (
    estimate,
    estimate_for_hospitals,
    get_personalized_bill,
    get_master_options,
    get_clinical_pathway,
    get_hospital_by_name,
    haversine
)



app = FastAPI(title="Healthcare Cost Estimator API — Poonawalla Fincorp")

# Allow CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serve the HealthBot HTML UI.
    Open http://localhost:8000 in your browser instead of opening healthbot.html directly.
    This avoids the file:// CORS restriction when calling the API.
    """
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthbot.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/options")
def get_options():
    return get_master_options()


@app.get("/estimate")
def get_estimate(
    procedure: str,
    hospital_tier: str,
    city: str,
    room_type: str = "SEMI_PRIVATE",
    sub_procedure: Optional[str] = None,
    variant: Optional[str] = None,
    comorbidities: Optional[List[str]] = Query(default=[])
):
    result = estimate(
        procedure=procedure,
        hospital_tier=hospital_tier,
        city=city,
        room_type=room_type,
        sub_procedure=sub_procedure,
        variant_hint=variant,
        comorbidities=comorbidities,
    )
    return result


@app.post("/personalized-bill")
def personalized_bill(payload: dict):
    result = get_personalized_bill(
        procedure=payload.get("procedure"),
        sub_procedure=payload.get("sub_procedure"),
        variant=payload.get("variant"),
        hospital_tier=payload.get("hospital_tier"),
        city=payload.get("city"),
        room_type=payload.get("room_type"),
        comorbidities=payload.get("comorbidities", []),
        save_json=False,
    )
    return result


@app.get("/pathway")
def clinical_pathway(
    procedure: str,
    sub_procedure: Optional[str] = None,
):
    steps = get_clinical_pathway(procedure, sub_procedure)
    if not steps:
        return {"error": f"No pathway found for: {procedure}"}
    return {
        "procedure": procedure,
        "sub_procedure": sub_procedure,
        "total_steps": len(steps),
        "pathway": steps,
    }


@app.get("/hospitals")

def hospitals_estimate(
    procedure: str,
    city: str,
    user_lat: float,
    user_lng: float,
    room_type: str = "SEMI_PRIVATE",
    sub_procedure: Optional[str] = None,
    variant: Optional[str] = None,
    comorbidities: Optional[List[str]] = Query(default=[]),
    hospital_types: Optional[str] = None,  # comma-separated list
):
    # Parse hospital_types as a list if provided
    types_list = [t.strip() for t in hospital_types.split(",") if t.strip()] if hospital_types else None
    result = estimate_for_hospitals(
        procedure=procedure,
        city=city,
        user_lat=user_lat,
        user_lng=user_lng,
        room_type=room_type,
        sub_procedure=sub_procedure,
        variant_hint=variant,
        comorbidities=comorbidities,
        hospital_types=types_list,
    )
    return result




class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Pass None for new session
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None

@app.post("/chat")

def chat(req: ChatRequest):
    """
    Main chatbot endpoint (session-aware, multi-turn).
    - Pass session_id to continue a conversation
    - Omit session_id (or pass null) to start new conversation
    - Returns session_id in response — save it for next message!
    """
    try:
        session_id = req.session_id or str(uuid.uuid4())
        session = get_or_create_session(session_id)

        # If user_lat/user_lng are provided, update session context
        if req.user_lat is not None and req.user_lng is not None:
            session["context"]["user_lat"] = req.user_lat
            session["context"]["user_lng"] = req.user_lng

        # ── Resolve hospital reference from last suggested list ──────────────
        # (before LLM so it gets the resolved name in context)
        ctx = session["context"]
        resolved_h = resolve_hospital_reference(req.message, ctx)
        if resolved_h:
            # Inject resolved name & tier into ctx so LLM + backend can use them
            ctx["referred_hospital"]      = resolved_h["name"]
            ctx["referred_hospital_tier"] = resolved_h.get("tier", "")
            # Augment the message so LLM knows exactly which hospital is meant
            augmented_msg = (
                f"[Referring to hospital: {resolved_h['name']} "
                f"(Tier: {resolved_h.get('tier','')})]. "
                + req.message
            )
        else:
            ctx.pop("referred_hospital", None)
            ctx.pop("referred_hospital_tier", None)
            augmented_msg = req.message

        # ── Auto-infer city from GPS if not already known ──────────────
        inferred_city_msg = ""
        if not ctx.get("city") and req.user_lat and req.user_lng:
            best_city, best_dist = None, 999999
            for c_name, coords in CITY_COORDS.items():
                try:
                    d = haversine(req.user_lat, req.user_lng, coords[0], coords[1])
                    if d < best_dist:
                        best_dist = d
                        best_city = c_name
                except:
                    pass
            if best_dist < 100:  # within 100km
                ctx["city"] = best_city
                inferred_city_msg = f"[System: User's GPS location is near {best_city.title()}. Automatically use '{best_city}' as their city for hospitals/costs.] "

        if inferred_city_msg:
            augmented_msg = inferred_city_msg + augmented_msg

        # System prompt + last 2 messages (1 turn) + new message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(session["history"][-4:])   # last 2 turns for better context
        messages.append({"role": "user", "content": augmented_msg})

        llm_data = call_llm(messages)
        if "error" in llm_data:
            return {
                "session_id": session_id,
                "reply": "Sorry, kuch technical issue ho gaya. Please dobara try karein.",
                "error": llm_data["error"],
            }

        update_context(session, llm_data)
        ctx = session["context"]

        # If hospital was resolved from last suggestions, override LLM hospital_name
        if ctx.get("referred_hospital"):
            llm_data["hospital_name"] = llm_data.get("hospital_name") or ctx["referred_hospital"]
            # For cost estimates: use resolved hospital's tier
            if ctx.get("referred_hospital_tier") and llm_data.get("intent") in (
                "PROCEDURE_COST", "HOSPITAL_LOOKUP_BY_NAME"
            ):
                ctx["referred_hospital_tier_for_cost"] = ctx["referred_hospital_tier"]

        # ── Bill-intent override ──────────────────────────────────────────────
        # "varma ka bill" / "arthros bill breakdown" → LLM returns HOSPITAL_LOOKUP
        # Redirect to PROCEDURE_COST if a procedure is known and billing keyword present
        _BILL_KW = ["bill", "cost", "estimate", "breakdown", "kitna", "price",
                    "charges", "fees", "amount", "total", "kharch", "lagega"]
        _is_bill = any(w in req.message.lower() for w in _BILL_KW)

        if llm_data.get("intent") == "HOSPITAL_LOOKUP_BY_NAME" and _is_bill and ctx.get("procedure"):
            llm_data["intent"]          = "PROCEDURE_COST"
            llm_data["backend_trigger"] = True
            llm_data["backend_type"]    = "cost"
            if ctx.get("referred_hospital_tier"):
                ctx["hospital_type_filter"] = ctx["referred_hospital_tier"]

        # Special handling for hospital lookup by name intent (pure lookup, no billing)
        elif llm_data.get("intent") == "HOSPITAL_LOOKUP_BY_NAME":
            try:
                lookup_result = handle_hospital_lookup(llm_data)
                if not lookup_result or not lookup_result.get("reply"):
                    reply = "Sorry, I couldn't find any hospital matching your query. Please check the name and try again."
                    structured_data = None
                else:
                    reply = lookup_result["reply"]
                    structured_data = lookup_result["structured_data"]
            except Exception as e:
                reply = f"Sorry, an error occurred while looking up the hospital: {str(e)}"
                structured_data = None
            response = {
                "session_id": session_id,
                "intent": llm_data.get("intent"),
                "reply": reply,
                "structured_data": structured_data,
                "missing_info": llm_data.get("missing_info", []),
            }
            session["history"].append({"role": "user", "content": req.message})
            session["history"].append({"role": "assistant", "content": response["reply"]})
            return response

        # Hospital type clarification logic
        # If last response asked for type, and user now provides a type, filter and re-run
        last_reply = session["history"][-1]["content"] if session["history"] else ""
        available_types = ctx.get("available_types")
        user_message_lower = req.message.strip().lower()
        clarified_type = None
        if available_types:
            for t in available_types:
                if t.lower() in user_message_lower:
                    clarified_type = t
                    break
        if clarified_type:
            ctx["hospital_type_filter"] = clarified_type
            backend_type = "hospitals"
            try:
                backend_data = call_backend(backend_type, ctx)
            except Exception as e:
                backend_data = {"backend_error": str(e)}
            response = format_response(llm_data, backend_data, session_id, ctx=ctx)
            response["reply"] = f"Showing only {clarified_type} hospitals:\n" + response["reply"]
            session["history"].append({"role": "user", "content": req.message})
            session["history"].append({"role": "assistant", "content": response["reply"]})
            ctx.pop("available_types", None)
            return response

        # Detect doctor-focused queries → hide hospital card, show only doctor card
        _msg_lower = req.message.lower()
        _doctor_keywords = ["doctor", "doctors", "doc ", "physician", "specialist", "surgeon"]
        _doctor_only = any(w in _msg_lower for w in _doctor_keywords)

        # ── CLINICAL_PATHWAY routing ─────────────────────────────────────────────────────────────────────────────
        # Triggered when user asks about procedure steps, pre-op prep, recovery time,
        # or anything about the clinical process. get_clinical_pathway() reads from
        # RATE_CARDS.xlsx Sheet 2 ("Clinical Pathways") and returns structured step data.
        if llm_data.get("intent") == "CLINICAL_PATHWAY":
            procedure  = llm_data.get("procedure") or ctx.get("procedure")
            sub_proc   = llm_data.get("sub_procedure") or ctx.get("sub_procedure")

            if not procedure:
                # Procedure unknown — ask for it
                response = {
                    "session_id":      session_id,
                    "intent":          "CLINICAL_PATHWAY",
                    "reply":           (
                        "Zaroor! Kaunse procedure ka clinical pathway chahiye? "
                        "Jaise Knee Replacement, Bypass Surgery, Cataract Surgery, ya koi aur?"
                    ),
                    "structured_data": {},
                    "missing_info":    ["procedure"],
                }
                session["history"].append({"role": "user",      "content": req.message})
                session["history"].append({"role": "assistant", "content": response["reply"]})
                return response

            steps = get_clinical_pathway(procedure, sub_proc)

            if steps:
                phases     = list(dict.fromkeys(s["phase"] for s in steps if s.get("phase")))
                first_time = steps[0].get("timeline", "")
                last_time  = steps[-1].get("timeline", "")
                summary    = f"{len(steps)} steps · {len(phases)} phases · {first_time} → {last_time}"

                pathway_backend_data = {
                    "pathway_steps":   len(steps),
                    "key_phases":      phases,
                    "pathway_summary": summary,
                    "full_pathway":    steps,
                }
                conv_reply = (
                    llm_data.get("conversational_reply")
                    or (
                        f"{procedure} ka poora clinical pathway yahan hai. "
                        "Har step mein kya hoga, kaun karega, aur timeline — sab clear hai."
                    )
                )
            else:
                pathway_backend_data = {
                    "pathway_error": f"'{procedure}' ka pathway abhi available nahi hai."
                }
                conv_reply = (
                    f"Sorry, {procedure} ka detailed pathway abhi hamare database mein nahi hai. "
                    "Aap directly hospital se pre-admission counselling le sakte hain."
                )

            response = format_response(llm_data, pathway_backend_data, session_id, ctx=ctx)
            response["reply"] = conv_reply
            session["history"].append({"role": "user",      "content": req.message})
            session["history"].append({"role": "assistant", "content": response["reply"]})
            return response
        # ── END CLINICAL_PATHWAY ─────────────────────────────────────────────────────────────────────────────

        backend_data = {}
        intent = llm_data.get("intent", "")
        backend_type = llm_data.get("backend_type", "none")

        # GENERAL_HEALTH_ADVICE: trigger specialty hospital lookup
        if intent == "GENERAL_HEALTH_ADVICE" and ctx.get("specialty_code"):
            try:
                backend_data = call_backend("specialty_hospitals", ctx)
            except Exception as e:
                backend_data = {"backend_error": str(e)}

        # AVOID HOSPITAL query — surface hospitals with risk_flags
        elif any(w in _msg_lower for w in ["avoid", "worst", "bad hospital", "not recommended"]):
            try:
                avoid_ctx = dict(ctx)
                if not avoid_ctx.get("specialty_code"):
                    avoid_ctx["specialty_code"] = "EMD"
                backend_data = call_backend("specialty_hospitals", avoid_ctx)
                all_h = backend_data.get("specialty_hospitals", [])
                risky = [h for h in all_h if h.get("risk_flags")]
                if risky:
                    backend_data["specialty_hospitals"] = risky
                    backend_data["avoid_mode"] = True
            except Exception as e:
                backend_data = {"backend_error": str(e)}

        # HOSPITAL_SUGGEST specialty-based (with OR without city — doctor-only handles no-city)
        elif intent == "HOSPITAL_SUGGEST" and backend_type == "specialty_hospitals" and ctx.get("specialty_code"):
            try:
                backend_data = call_backend("specialty_hospitals", ctx)
            except Exception as e:
                backend_data = {"backend_error": str(e)}

        # Doctor-only fallback: user asked for doctors, specialty known, no explicit hospital intent
        elif _doctor_only and ctx.get("specialty_code"):
            try:
                backend_data = call_backend("specialty_hospitals", ctx)
            except Exception as e:
                backend_data = {"backend_error": str(e)}

        # HOSPITAL_SUGGEST / PROCEDURE_COST with a specific procedure
        elif llm_data.get("backend_trigger") and ctx.get("procedure") and ctx.get("city"):
            try:
                # ── Budget present → always show cost + hospitals together ──────────
                # LLM returns backend_type="cost" for PROCEDURE_COST, but when user
                # has stated a budget they clearly want to see hospital options too.
                effective_backend_type = backend_type if backend_type != "none" else "both"
                if ctx.get("budget_inr") and effective_backend_type == "cost":
                    effective_backend_type = "both"
                backend_data = call_backend(effective_backend_type, ctx)
            except Exception as e:
                backend_data = {"backend_error": str(e)}

        # Set doctor_only_mode if query was about doctors (not hospitals)
        if _doctor_only and backend_data.get("recommended_doctors"):
            backend_data["doctor_only_mode"] = True

        # If hospitals are present and multiple types, prompt for clarification
        if backend_data.get("hospitals") and backend_data.get("available_types") and len(backend_data["available_types"]) > 1:
            ctx["available_types"] = backend_data["available_types"]
            types_str = ", ".join(backend_data["available_types"])
            follow_up = f"Which type of hospital do you want? ({types_str})"
        else:
            ctx.pop("available_types", None)
            follow_up = llm_data.get("follow_up_question")

        # Store last suggested hospitals for future cross-referencing by number/name
        spec_hosps = backend_data.get("specialty_hospitals", [])
        if spec_hosps:
            ctx["last_suggested_hospitals"] = spec_hosps
        elif backend_data.get("hospitals"):
            ctx["last_suggested_hospitals"] = [
                {"name": h["name"], "tier": h["tier"],
                 "rating": h.get("rating", 0), "phone": h.get("phone", ""),
                 "distance_km": h.get("distance_km")}
                for h in backend_data["hospitals"]
            ]

        # Use resolved hospital tier for cost if available
        if ctx.get("referred_hospital_tier_for_cost"):
            ctx["hospital_type_filter"] = ctx.pop("referred_hospital_tier_for_cost")

        response = format_response(llm_data, backend_data, session_id, ctx=ctx)
        
        # Ensure follow-up question from API logic is appended to the reply since UI only renders reply
        if follow_up:
            response["follow_up_question"] = follow_up
            if follow_up not in response["reply"]:
                response["reply"] += f"\n\n❓ {follow_up}"

        session["history"].append({"role": "user", "content": req.message})
        session["history"].append({"role": "assistant", "content": response["reply"]})

        return response
    except Exception as e:
        # Always return a JSON error with CORS headers
        return {
            "session_id": getattr(req, "session_id", None),
            "reply": "Sorry, an unexpected error occurred. Please try again.",
            "error": str(e),
        }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)