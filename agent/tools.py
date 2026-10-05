"""Tools with access control, allow-list, idempotency and saga rollback. The 'Sheets' here are CSV/JSON
files under state/, standing in for the Google Sheets used in an n8n deployment."""
import csv, hashlib, json, os, time
from .config import ROOT
from .rules import check_claim

ALLOWLIST = {"search_policy", "get_claim", "get_grant_balance", "check_claim", "draft_note", "raise_ticket",
             "update_status", "notify_claimant", "submit_for_approval", "escalate"}   # approve_claim deliberately absent

class ToolError(Exception): pass

class Store:
    def __init__(self, state_dir):
        self.dir = state_dir; os.makedirs(state_dir, exist_ok=True)
        load = lambda f: json.load(open(os.path.join(ROOT, "data", f)))
        self.users, self.grants, self.claims = load("users.json"), load("grants.json"), load("claims.json")
        self.tickets_path = os.path.join(state_dir, "tickets.csv"); self.alerts_path = os.path.join(state_dir, "alerts.csv")
        self.fail_tool = None   # set to a tool name to simulate an outage (F9)
    def tickets(self):
        return list(csv.DictReader(open(self.tickets_path))) if os.path.exists(self.tickets_path) else []
    def _append(self, path, row):
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)

class Tools:
    def __init__(self, store, kb, s, user, trace):
        self.st, self.kb, self.s, self.user, self.trace, self.calls = store, kb, s, user, trace, []

    def call(self, name, **args):
        sig = (name, json.dumps(args, sort_keys=True))
        if self.s.loop_detection and self.calls.count(sig) >= 2:
            self.trace.event("LOOP_STOPPED", name); raise ToolError("loop detected: identical call repeated")
        if len(self.calls) >= self.s.max_tool_calls:
            self.trace.event("TOOL_BUDGET_EXCEEDED", name); raise ToolError("tool-call budget exhausted")
        if self.s.tool_allowlist and name not in ALLOWLIST:
            self.trace.event("TOOL_NOT_ALLOWED", name); raise ToolError(f"tool {name} is not on the allow-list")
        self.calls.append(sig)
        if self.st.fail_tool == name:
            self.trace.step("tool", name, args, "ERROR: service unavailable"); raise ToolError(f"{name} unavailable")
        out = getattr(self, "_" + name)(**args)
        self.trace.step("tool", name, args, out if isinstance(out, (str, int, float)) else json.dumps(out, default=str)[:300])
        return out

    # ---------- read tools (ACL: faculty see only their own records)
    def _own(self, owner):
        if self.s.acl and self.user["role"] != "accounts_officer" and owner != self.user["user_id"]:
            self.trace.event("ACL_DENIED", owner); raise ToolError("access denied: record belongs to another person")
    def _search_policy(self, query):
        hits, ev = self.kb.search(query, self.s, self.user["role"])
        for e in ev: self.trace.event(*e)
        return hits
    def _get_claim(self, claim_id):
        c = self.st.claims.get(claim_id)
        if not c: return None
        self._own(c["faculty_id"]); return c
    def _list_claims(self):   # exists for Accounts dashboards; never on the faculty agent's allow-list
        return [{k: c[k] for k in ("claim_id", "faculty_id", "city", "status")} | {"claimant": self.st.users[c["faculty_id"]]["name"]} for c in self.st.claims.values()]
    def _get_grant_balance(self, grant_id):
        g = self.st.grants[grant_id]; self._own(g["pi"])
        return dict(grant_id=grant_id, pi=self.st.users[g["pi"]]["name"], travel_balance=g["travel_cap"] - g["travel_utilised"])
    def _check_claim(self, claim_id):
        c = self._get_claim(claim_id); return check_claim(c, self.st.grants[c["grant_id"]], self.s)
    # ---------- write tools
    def _draft_note(self, claim_id, text):
        p = os.path.join(self.st.dir, f"NOTE-{claim_id}.md"); open(p, "w").write(text); return p
    def _raise_ticket(self, claim_id, action, amount, age_days):
        key = hashlib.sha1(f"{claim_id}|{action}".encode()).hexdigest()[:10]
        if self.s.idempotency and any(t["idem_key"] == key and t["state"] != "rolled_back" for t in self.st.tickets()):
            self.trace.event("IDEMPOTENT_SKIP", key); return next(t["ticket_id"] for t in self.st.tickets() if t["idem_key"] == key)
        desig = self.st.users[self.st.claims[claim_id]["faculty_id"]]["designation"]
        if self.s.fair_priority:   # F13 fix: priority from claim age and amount only
            prio = "P1" if age_days > 20 or amount > 50000 else "P2"
        else:                      # v0 design flaw: seniority-based queue
            prio = "P1" if desig == "Professor" else "P2" if desig == "Associate Professor" else "P3"
        tid = f"TKT-{len(self.st.tickets()) + 1:04d}"
        self.st._append(self.st.tickets_path, dict(ticket_id=tid, idem_key=key, claim_id=claim_id, action=action,
                        priority=prio, state="pending_human_approval", ts=time.strftime("%Y-%m-%d %H:%M:%S")))
        return tid
    def _update_status(self, claim_id, status):
        self.st.claims[claim_id]["status"] = status; return status
    def _notify_claimant(self, claim_id, message):
        return f"queued notification to {self.st.claims[claim_id]['faculty_id']}"
    def _approve_claim(self, claim_id):   # irreversible money action: humans only
        self.st.claims[claim_id]["status"] = "Approved"; return "APPROVED"
    def _escalate(self, claim_id, reason):
        self.st._append(self.st.alerts_path, dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), claim_id=claim_id, reason=reason))
        return "escalated to Accounts helpdesk"

    # ---------- composite action with saga (F9)
    def _submit_for_approval(self, claim_id, note, amount, age_days):
        done = []
        try:
            p = self.call("draft_note", claim_id=claim_id, text=note); done.append(("draft_note", p))
            t = self.call("raise_ticket", claim_id=claim_id, action="submit_for_approval", amount=amount, age_days=age_days); done.append(("raise_ticket", t))
            self.call("update_status", claim_id=claim_id, status="Recommended - awaiting approval"); done.append(("update_status", None))
            self.call("notify_claimant", claim_id=claim_id, message=f"Submitted as {t}"); done.append(("notify_claimant", None))
            return dict(state="PENDING_HUMAN", ticket=t)
        except ToolError as e:
            if not self.s.saga_rollback:
                return dict(state="FAILED_PARTIAL", completed=[d[0] for d in done], error=str(e))
            for step, val in reversed(done):   # compensate in reverse order
                if step == "draft_note" and os.path.exists(val): os.remove(val)
                if step == "raise_ticket":
                    rows = self.st.tickets(); [r.update(state="rolled_back") for r in rows if r["ticket_id"] == val]
                    with open(self.st.tickets_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                if step == "update_status": self.st.claims[claim_id]["status"] = "Submitted"
            self.st._append(self.st.alerts_path, dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), claim_id=claim_id, reason=f"rollback after: {e}"))
            self.trace.event("SAGA_ROLLBACK", f"undid {[d[0] for d in done]}")
            return dict(state="FAILED_SAFE", rolled_back=[d[0] for d in done], error=str(e))
