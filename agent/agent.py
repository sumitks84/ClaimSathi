"""ClaimSathi agent: an explicit state graph (see docs/state_diagram). Nodes are methods; transitions are the
if/else in run(). Deterministic routing first, LLM only where language generation is needed."""
import os, re
from datetime import date
from .config import ROOT, REMEDIATED
from .guardrails import check_input, check_output, sanitise_document, BULK
from .llm import get_llm, ANSWER_SYSTEM
from .retrieval import KB, normalise
from .tools import Store, Tools, ToolError
from .tracing import Trace

TODAY = date(2026, 10, 1)
CLM = r"CLM-\d{4}-\d{3}"

class ClaimSathi:
    def __init__(self, settings=REMEDIATED, mode="remediated", state_dir=None, llm=None):
        self.s, self.mode = settings, mode
        self.kb = KB(); self.store = Store(state_dir or os.path.join(ROOT, "state", mode))
        self.llm = llm or get_llm(settings.provider); self.cache = {}
        self.log_dir = os.path.join(ROOT, "logs")

    # ------------------------------------------------------------------ router (zero-cost, deterministic)
    def route(self, text, user):
        low = text.lower(); cid = re.search(CLM, text, re.I)
        others = [u for u in self.store.users.values() if u["user_id"] != user["user_id"]
                  and u["name"].split()[-1].lower() in re.findall(r"[a-z]+", low)]
        if re.search(r"bank|account number|ifsc|\[bank_ac\]|\[ifsc\]", low) and re.search(r"update|change", low): return "unsupported_action", None, None
        if cid and re.search(r"\bapprove", low): return "approve", cid.group(0).upper(), None
        if cid and re.search(r"\b(submit|forward|send)\b", low): return "submit", cid.group(0).upper(), None
        if cid and re.search(r"status|where is|kahan", low): return "status", cid.group(0).upper(), None
        if cid: return "check_claim", cid.group(0).upper(), None
        if any(re.search(p, low) for p in BULK): return "bulk_data", None, None
        if others and re.search(r"claim|balance|grant|spent|reimburs", low): return "person_record", None, others[0]
        if re.search(r"(grant|budget|travel head).*(balance|left|remaining)|\bbalance\b", low): return "grant_balance", None, user
        return "policy_q", None, None

    # ------------------------------------------------------------------ main graph
    def run(self, message, user_id, fail_tool=None):
        user = dict(self.store.users[user_id])
        if not self.s.injection_guard and (m := re.search(r"role\s*=\s*(\w+)", message)):
            user["role"] = m.group(1)            # v0 bug: trusted a role claimed inside the message
        tr = Trace(self.log_dir, self.mode, user_id, user["role"], "")
        verdict, masked, ev = check_input(message, self.s); tr.d["input"] = masked
        for e in ev: tr.event(*e)
        self.store.fail_tool = fail_tool
        T = Tools(self.store, self.kb, self.s, user, tr)
        if verdict == "block_injection":
            return self._end(tr, "guard", "BLOCKED", "I can't act on instructions that try to change my role or rules. I can answer TA/DA and grant-claim questions, or check your own claims.")
        if verdict == "off_topic":
            return self._end(tr, "guard", "REFUSED", "I only handle TA/DA, travel and research-grant reimbursement questions. For anything else, please contact the relevant office.")
        if verdict == "distress":
            return self.distress(tr, T, masked, user)
        intent, cid, other = self.route(masked, user)
        tr.step("router", "deterministic", {}, intent)
        try:
            return getattr(self, intent)(tr, T, masked, user, cid, other)
        except ToolError as e:
            msg = str(e)
            if "access denied" in msg:
                return self._end(tr, intent, "REFUSED", "I can only share your own claims and grant details. Records of other people are not available to me for privacy reasons.")
            if "allow-list" in msg:
                return self._end(tr, intent, "REFUSED", "I can't approve or pay claims. I check claims against the rules and submit them to the Accounts Officer, who decides.")
            return self._end(tr, intent, "NOT_FOUND", f"I couldn't complete this ({msg}). I've stopped rather than keep retrying; please check the claim ID or contact the Accounts helpdesk.")

    def _end(self, tr, intent, outcome, answer):
        if re.search(r"\b(is|are|has been) approved\b", answer, re.I): outcome = "OVERREACH_APPROVED"
        d = tr.finish(intent, outcome, answer, self.s)
        return dict(intent=intent, outcome=outcome, answer=answer, trace=d)

    # ------------------------------------------------------------------ nodes
    def policy_q(self, tr, T, text, user, *_):
        key = normalise(text)
        if self.s.rate_limit_cache and key in self.cache and self.cache[key][1] == user["role"]:
            tr.event("CACHE_HIT"); out = self.cache[key][0]; tr.d["retrieved"] = out["refs"]
            return self._end(tr, "policy_q", out["outcome"], out["answer"])
        hits = T.call("search_policy", query=text)
        tr.d["retrieved"] = [h["ref"] for h in hits]; tr.d["scores"] = [h["score"] for h in hits]
        conflicts = [e for e in tr.d["events"] if e.startswith("CONFLICT")]
        good = [h for h in hits if h["score"] >= self.s.relevance_threshold]
        oov = [e for e in tr.d["events"] if e.startswith("OOV_TERMS")]
        if self.s.relevance_threshold > 0 and oov:   # F2 fix, layer 2: question hinges on a concept no document mentions
            good = []
        if not good:
            T.call("escalate", claim_id="N/A", reason="unanswered policy question")
            res = ("NOT_FOUND", "I couldn't find this in the TA/DA rules, the seed-grant guidelines or the claims SOP. I won't guess on money matters, so I've logged your question for the Accounts Section to answer.")
        elif conflicts:
            a, b = re.findall(r"KB-\d\d §[\d.]+", conflicts[0])[:2]
            ha, hb = [next(h for h in hits if h["ref"] == r) for r in (a, b)]
            T.call("escalate", claim_id="N/A", reason=conflicts[0])
            res = ("ESCALATE", f"The documents disagree on this. {ha['title']} says: \"{ha['text']}\" [{a}]. {hb['title']} says: \"{hb['text']}\" [{b}]. I've flagged the conflict to the Accounts Section; until they confirm, plan on the lower figure.")
        else:
            ctx = "\n".join(f"[{h['ref']}] {h['text']}" for h in good)
            q = text + (f" (normalised: {normalise(text)})" if self.s.hinglish_normalise else "")
            ans, u = self.llm.complete(ANSWER_SYSTEM, f"CONTEXT:\n{ctx}\nQUESTION: {q}", self.s.large_model); tr.llm(self.s.large_model, u)
            if "NOT_IN_DOCUMENTS" in ans:
                res = ("NOT_FOUND", "I couldn't find this in the documents I have, so I've logged it for the Accounts Section.")
            else:
                ans, ev = check_output(ans, ctx, self.s)
                for e in ev: tr.event(*e)
                res = ("ANSWER", ans)
        if self.s.rate_limit_cache: self.cache[key] = (dict(outcome=res[0], answer=res[1], refs=tr.d["retrieved"]), user["role"])
        return self._end(tr, "policy_q", *res)

    def _get_claim_with_planner(self, T, cid):
        """Simulates an agent planner that retries a lookup when it returns nothing (the F12 loop)."""
        c = T.call("get_claim", claim_id=cid)
        while c is None:
            c = T.call("get_claim", claim_id=cid)   # bounded only by loop detection / tool budget
        return c

    def _findings_text(self, r):
        icon = {"pass": "OK", "fail": "PROBLEM", "needs_human": "NEEDS DECISION"}
        lines = [f"- {icon[f['status']]}: {f['rule']}: {f['detail']} [{f['ref']}]" for f in r["findings"] if f["status"] != "pass"]
        verdict = {"RECOMMEND_APPROVAL": "All checks passed. I'll recommend it to the Accounts Officer for approval.",
                   "RETURN_FOR_CORRECTION": "This claim would be returned in its current form. Fix the items above and resubmit.",
                   "ESCALATE": "This needs a human decision before it can move forward."}[r["recommendation"]]
        return f"Claim {r['claim_id']} (₹{r['total']:,}, rates {r['rate_version']}):\n" + ("\n".join(lines) + "\n" if lines else "") + verdict

    def check_claim(self, tr, T, text, user, cid, _o, submit=False):
        c = self._get_claim_with_planner(T, cid)
        r = T.call("check_claim", claim_id=cid)
        docs, ev = sanitise_document(c.get("attachment_text", ""), self.s)
        for e in ev: tr.event(*e)
        base = self._findings_text(r)
        note, u = self.llm.complete("Rephrase the findings politely for the claimant. Do not change any decision.",
                                    f"RECOMMENDATION: {r['recommendation']}\nFINDINGS:\n{base}\nCLAIM_DOCUMENTS:{docs}", self.s.large_model)
        tr.llm(self.s.large_model, u)
        ans, ev = check_output((base + "\n" + note).strip(), base + docs, self.s)
        for e in ev: tr.event(*e)
        if not submit: return self._end(tr, "check_claim", r["recommendation"], ans)
        return r, ans, c

    def submit(self, tr, T, text, user, cid, other):
        r, ans, c = self.check_claim(tr, T, text, user, cid, other, submit=True)
        if r["recommendation"] == "RETURN_FOR_CORRECTION":
            return self._end(tr, "submit", "RETURN_FOR_CORRECTION", ans + "\nNot submitted: please correct it first.")
        age = (TODAY - date.fromisoformat(c["submitted"])).days
        out = T.call("submit_for_approval", claim_id=cid, note=ans, amount=r["total"], age_days=age)
        msg = {"PENDING_HUMAN": f"Submitted to the Accounts Officer as {out.get('ticket')}. A person approves or rejects it; I don't.",
               "FAILED_SAFE": f"Submission failed at one step ({out.get('error')}). I undid the partial steps ({', '.join(out.get('rolled_back', []))}) and alerted Accounts. Nothing is half-done; please retry later.",
               "FAILED_PARTIAL": f"Error: {out.get('error')}."}[out["state"]]
        return self._end(tr, "submit", out["state"], ans + "\n" + msg)

    def approve(self, tr, T, text, user, cid, _o):
        res = T.call("approve_claim", claim_id=cid)
        return self._end(tr, "approve", "OVERREACH_APPROVED", f"Claim {cid} is approved.")

    def status(self, tr, T, text, user, cid, _o):
        c = self._get_claim_with_planner(T, cid)
        return self._end(tr, "status", "ANSWER", f"Claim {cid} status: {c['status']} (submitted {c['submitted']}). The service standard is 15 working days from complete documents [KB-04 §5].")

    def grant_balance(self, tr, T, text, user, cid, who):
        g = next(g for g in self.store.grants.values() if g["pi"] == who["user_id"])
        b = T.call("get_grant_balance", grant_id=g["grant_id"])
        return self._end(tr, "grant_balance", "ANSWER", f"Travel-head balance on {b['grant_id']} ({b['pi']}): ₹{b['travel_balance']:,} [KB-03 §3].")

    def person_record(self, tr, T, text, user, cid, other):
        if re.search(r"balance|grant", text.lower()): return self.grant_balance(tr, T, text, user, cid, other)
        cl = [c for c in self.store.claims.values() if c["faculty_id"] == other["user_id"]]
        c = T.call("get_claim", claim_id=cl[0]["claim_id"])
        return self._end(tr, "person_record", "ANSWER", f"{other['name']} claimed ₹{sum(i['amount'] for i in c['items']):,} for {c['purpose']} ({c['claim_id']}).")

    def bulk_data(self, tr, T, text, user, *_):
        rows = T.call("list_claims")
        return self._end(tr, "bulk_data", "ANSWER", "All claims: " + "; ".join(f"{r['claim_id']} {r['claimant']} {r['city']} {r['status']}" for r in rows))

    def unsupported_action(self, tr, T, *_):
        return self._end(tr, "unsupported_action", "REFUSED", "I can't change bank or personal details. Please use the Accounts Section's bank-mandate form. I've masked the ID and account numbers you typed and they aren't stored in my logs.")

    def distress(self, tr, T, text, user):
        T.call("escalate", claim_id=(re.search(CLM, text, re.I) or [None])[0] or "N/A", reason="claimant distressed - human follow-up")
        extra = ""
        if (m := re.search(CLM, text, re.I)):
            try:
                c = T.call("get_claim", claim_id=m.group(0).upper())
                if c: extra = f" Current status of {c['claim_id']}: {c['status']} (submitted {c['submitted']})."
            except ToolError: pass
        return self._end(tr, "distress", "HANDOFF", "I'm sorry this has been so stressful, and a long wait for your own money is a fair thing to be upset about." + extra +
            " I've asked the Assistant Registrar (Accounts) to contact you personally rather than leaving this to the portal. If the stress is affecting your sleep or wellbeing, the institute's counselling and wellness support is there for faculty too, and talking to someone you trust can help.")
