"""Deterministic claim rule engine. Money decisions never depend on the LLM: the engine produces findings
with citations; the LLM only phrases them. Name, gender, designation and home state are never inputs."""
from datetime import date

RATES = {  # rates as published; v1 = handbook KB-01 (2024), v2 = circular KB-02 (Apr 2026)
 "v1": dict(hotel={"X": 6000, "Y": 4500, "Z": 3000}, da=1000, deadline=60, hotel_ref="KB-01 §3.2", da_ref="KB-01 §3.3", deadline_ref="KB-01 §5.1"),
 "v2": dict(hotel={"X": 7500, "Y": 5000, "Z": 3500}, da=1200, deadline=30, hotel_ref="KB-02 §2", da_ref="KB-02 §3", deadline_ref="KB-02 §5")}
CITY = {**{c: "X" for c in "Delhi Mumbai Kolkata Chennai Bengaluru Hyderabad Ahmedabad Pune".split()},
        **{c: "Y" for c in "Patna Lucknow Jaipur Bhubaneswar Ranchi Kochi Bhopal Chandigarh Goa Indore".split()}}
LOCAL_CAP, SELF_CERT_ITEM, SELF_CERT_TOTAL, REG_CAP_HANDBOOK, REG_CAP_GRANT = 800, 500, 2000, 30000, 40000
d = lambda x: date.fromisoformat(x)

def check_claim(claim, grant, s):
    F = []
    add = lambda rule, status, detail, ref: F.append(dict(rule=rule, status=status, detail=detail, ref=ref))
    ver = "v2" if (s.version_filter and d(claim["ret"]) >= date(2026, 4, 1)) else "v1"   # baseline: stale handbook
    R, cls = RATES[ver], CITY.get(claim["city"], "Z")
    # R1 sanction
    if not claim["prior_sanction"]:
        add("R1 Travel sanction", "needs_human" if claim["emergency"] else "fail",
            "Emergency travel: post-facto approval of the Dean needed within 7 days" if claim["emergency"] else "No Form TA-1 sanction on record", "KB-01 §2")
    else: add("R1 Travel sanction", "pass", "Form TA-1 sanction on record", "KB-01 §2")
    total = 0
    for it in claim["items"]:
        t, amt = it["type"], it["amount"]; total += amt
        if t in ("air", "hotel", "registration") and not it.get("bill"):
            add(f"R6 Bills ({t})", "fail", f"No bill for {t}; self-certificate is not accepted for {t}", "KB-04 §4"); continue
        if s.bill_match_rule and it.get("bill") and abs(it.get("bill_amount", amt) - amt) > 0.01 * amt:
            add(f"R7 Bill match ({t})", "fail", f"Claimed ₹{amt:,} but the bill shows ₹{it['bill_amount']:,}", "KB-04 §2")
        if t == "air":
            add("R2 Class of travel", "fail" if it["class"] != "economy" else "pass", f"Air {it['class']} class", "KB-01 §3.1")
        elif t == "hotel":
            cap = R["hotel"][cls]
            if it["per_night"] <= cap:
                add("R3 Hotel ceiling", "pass", f"₹{it['per_night']:,}/night within ₹{cap:,} cap ({claim['city']}, {cls}-class)", R["hotel_ref"])
            elif s.exception_rules and claim["conference_hotel_letter"]:
                add("R3 Hotel ceiling", "needs_human", f"₹{it['per_night']:,}/night exceeds ₹{cap:,} cap, but organiser-designated venue hotel letter attached: excess admissible only with Dean (Research) endorsement", "KB-02 §4")
            else:
                add("R3 Hotel ceiling", "fail", f"₹{it['per_night']:,}/night exceeds ₹{cap:,} cap for {claim['city']} ({cls}-class); admissible ₹{cap*it['nights']:,}", R["hotel_ref"])
        elif t == "da":
            rate = R["da"] // 2 if claim["free_meals"] else R["da"]; ent = rate * it["days"]
            add("R4 Daily allowance", "pass" if amt <= ent else "fail", f"Claimed ₹{amt:,}; entitled ₹{ent:,} ({it['days']} days × ₹{rate:,})", R["da_ref"])
        elif t == "local":
            cap = LOCAL_CAP * it.get("days", 1)
            if not it.get("bill") and (amt > SELF_CERT_TOTAL):
                add("R5 Local conveyance", "fail", f"Self-certificate limit ₹{SELF_CERT_TOTAL:,} exceeded", "KB-04 §4")
            else: add("R5 Local conveyance", "pass" if amt <= cap else "fail", f"₹{amt:,} vs ₹{cap:,} cap", "KB-01 §3.4")
        elif t == "registration":
            if amt <= REG_CAP_HANDBOOK: add("R10 Registration", "pass", f"₹{amt:,} within both published caps", "KB-01 §4; KB-03 §5")
            elif amt <= REG_CAP_GRANT:
                add("R10 Registration", "needs_human" if s.conflict_check else "fail",
                    f"₹{amt:,}: handbook cap ₹30,000 conflicts with grant guideline cap ₹40,000; Accounts to confirm", "KB-01 §4; KB-03 §5")
            else: add("R10 Registration", "fail", f"₹{amt:,} exceeds ₹40,000", "KB-03 §5")
    lag = (d(claim["submitted"]) - d(claim["ret"])).days
    add("R8 Submission deadline", "pass" if lag <= R["deadline"] else "needs_human",
        f"Submitted {lag} days after return (limit {R['deadline']})" + ("" if lag <= R["deadline"] else "; Director's condonation needed"), R["deadline_ref"])
    bal = grant["travel_cap"] - grant["travel_utilised"]
    add("R9 Grant travel head", "pass" if total <= bal else "fail", f"Claim ₹{total:,} vs Travel-head balance ₹{bal:,} on {grant['grant_id']}", "KB-03 §3")
    st = [f["status"] for f in F]
    rec = "RETURN_FOR_CORRECTION" if "fail" in st else "ESCALATE" if "needs_human" in st else "RECOMMEND_APPROVAL"
    return dict(claim_id=claim["claim_id"], rate_version=ver, total=total, findings=F, recommendation=rec)
