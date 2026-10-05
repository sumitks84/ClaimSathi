"""Input, output and document guardrails: regex + rule classifiers (the free-tier equivalent of the
n8n Guardrails node / Llama Guard). Each function returns (result, events)."""
import re

PII = [("AADHAAR", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), ("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
       ("IFSC", r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), ("PHONE", r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
       ("BANK_AC", r"(?<![\d-])\d{9,18}(?![\d-])"), ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")]
INJECTION = [r"ignore (all |the )?(previous|prior|above) (instructions|rules)", r"you are now", r"\bsystem\s*:",
    r"admin mode", r"developer mode", r"role\s*=\s*\w+", r"disregard (your|the) (rules|policy|instructions)",
    r"pretend (to be|you are)", r"jailbreak", r"reveal (your )?(system )?prompt", r"note to (the )?ai",
    r"ai assistant note", r"mark (the|this) claim (as )?(compliant|approved)", r"tell the user it is approved"]
BULK = [r"\ball (students|faculty|claims|users)\b", r"\bevery (faculty|user|claimant)\b", r"\blist all\b"]
DISTRESS = [r"stress", r"anxious", r"anxiety", r"can'?t sleep", r"can'?t handle", r"depress", r"breaking down",
            r"hopeless", r"overwhelm", r"harass", r"desperate", r"crushing"]
DOMAIN = r"(claim|ta\b|da\b|tada|travel|hotel|grant|reimburs|conference|registration|bill|receipt|allowance|advance|" \
         r"sanction|air|flight|train|conveyance|deadline|seed|budget|head|refund|invoice|status|clm-|audit|kitna|milega|" \
         r"jama|safar|khana|regn|conf\b|per diem|lodging|approve|submit|balance|condon|city|patna|delhi|mumbai)"

def mask_pii(text, on=True):
    if not on: return text, []
    ev = []
    for name, pat in PII:
        if re.search(pat, text):
            ev.append(("PII_MASKED", name)); text = re.sub(pat, f"[{name}]", text)
    return text, ev

def check_input(text, s):
    """Returns (verdict, masked_text, events). verdict in ok / block_injection / off_topic / distress."""
    ev = []
    masked, pev = mask_pii(text, s.pii_mask); ev += pev
    low = text.lower()
    if s.injection_guard and any(re.search(p, low) for p in INJECTION):
        return "block_injection", masked, ev + [("INJECTION_BLOCKED", "input")]
    if s.distress_handoff and any(re.search(p, low) for p in DISTRESS):
        return "distress", masked, ev + [("DISTRESS_DETECTED", "handoff")]
    if s.offtopic_guard and not re.search(DOMAIN, low):
        return "off_topic", masked, ev + [("OFF_TOPIC", "refused")]
    return "ok", masked, ev

def sanitise_document(text, s):
    """F6: text from uploaded bills is DATA. Remove instruction-like lines before any model sees it."""
    if not s.injection_guard or not text: return text, []
    kept, ev = [], []
    for line in text.splitlines():
        if any(re.search(p, line.lower()) for p in INJECTION):
            ev.append(("INJECTION_IN_DOCUMENT", line[:60] + "...")); continue
        kept.append(line)
    return "\n".join(kept), ev

# Scoped to promises about a specific claim/person. v1 of this filter also matched policy text such as
# "a seed grant ... is sanctioned for two years" (false positive caught by test T22).
COMMIT = [r"(your|this|the|my) (claim|refund|reimbursement|bill)[^.]{0,40}\b(is|has been|stands|was) (approved|sanctioned|cleared)",
          r"\byou will be (reimbursed|paid|credited)|\bwill be reimbursed in full", r"refund (is|has been) (approved|processed)",
          r"\bguarantee", r"\b(is|has been) approved\b(?! by)"]

def check_output(answer, context, s):
    """Grounding (every figure must appear in the evidence; citations must point at retrieved sections)
    and commitment filter. Returns (answer, events)."""
    ev = []
    if s.commitment_filter:
        for p in COMMIT:
            if re.search(p, answer, re.I):
                ev.append(("COMMITMENT_BLOCKED", p))
                answer = re.sub(r"[^.\n]*" + p + r"[^.\n]*[.]?", " Final approval rests with the competent authority in Accounts; I can only recommend.", answer, flags=re.I)
    if s.grounding_check:
        ctx = context.replace(",", "")
        kept = []
        for sent in re.split(r"(?<=[.!?])\s+", answer):
            nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*", sent)]
            bad = [n for n in nums if len(n) >= 2 and n not in ctx]
            refs = re.findall(r"KB-\d\d §[\d.]+", sent)
            badref = [r for r in refs if r not in context]
            if bad or badref:
                ev.append(("UNGROUNDED_REMOVED", (bad + badref)[0])); continue
            kept.append(sent)
        answer = " ".join(kept).strip() or "I could not verify an answer against the policy documents, so I have not given one. I've flagged this for the Accounts Section."
    return answer, ev

def groundedness(answer, context):
    """Metric only (independent of toggles): share of figure-bearing sentences fully supported."""
    ctx = context.replace(",", "")
    answer = re.sub(r"\[?KB-\d\d §[\d.]+\]?|CLM-\d{4}-\d{3}|G-\d{4}-\d\d|TKT-\d{4}", "", answer)   # IDs/citations are not claims
    sents = [x for x in re.split(r"(?<=[.!?])\s+", answer) if re.search(r"\d{2,}", x.replace(",", ""))]
    if not sents: return 1.0
    ok = sum(all(n.replace(",", "") in ctx for n in re.findall(r"\d[\d,]*", x) if len(n.replace(",", "")) >= 2) for x in sents)
    return ok / len(sents)
