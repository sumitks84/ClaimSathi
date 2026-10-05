"""Runs the test set against BASELINE (v0, guards off) and REMEDIATED (v1) agents, plus the load test (F15).
Writes eval/results_<mode>.csv, eval/summary.json and per-failure trace files in eval/failure_traces/."""
import csv, json, os, re, shutil, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import ClaimSathi
from agent.config import BASELINE, REMEDIATED, ROOT
from agent.guardrails import groundedness

HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda t: t.lower().replace(",", "")

def run_mode(name, settings):
    shutil.rmtree(os.path.join(ROOT, "state", name), ignore_errors=True)
    for f in (f"traces_{name}.jsonl", f"runs_{name}.csv"):
        p = os.path.join(ROOT, "logs", f); os.path.exists(p) and os.remove(p)
    agent = ClaimSathi(settings, name); rows = []
    for t in csv.DictReader(open(os.path.join(HERE, "testset.csv"))):
        r = agent.run(t["message"], t["user_id"], fail_tool=t["fail_tool"] or None); tr = r["trace"]
        exp_src = [x for x in t["expected_sources"].split(";") if x]
        hit = (any(x in tr["retrieved"] for x in exp_src)) if exp_src else None
        ok_out = r["outcome"] == t["expected_outcome"]
        ok_in = all(norm(x) in norm(r["answer"]) for x in t["must_contain"].split("|") if x)
        ok_not = not any(norm(x) in norm(r["answer"]) for x in t["must_not_contain"].split("|") if x)
        ctx = " ".join(c["text"] for c in agent.kb.chunks if c["ref"] in tr["retrieved"]) + " " + json.dumps(tr["steps"])
        rows.append(dict(id=t["id"], category=t["category"], failure_ref=t["failure_ref"], user=t["user_id"], message=t["message"],
            expected=t["expected_outcome"], got=r["outcome"], outcome_ok=ok_out, contains_ok=ok_in, not_contains_ok=ok_not,
            success=ok_out and ok_in and ok_not, retrieval_hit=hit, grounded=round(groundedness(r["answer"], ctx), 2) if r["outcome"] == "ANSWER" else "",
            latency_ms=tr["latency_ms"], tool_calls=tr["tool_calls"], in_tok=tr["in_tok"], out_tok=tr["out_tok"], cost_inr=tr["cost_inr"],
            events=";".join(tr["events"]), answer=r["answer"], run_id=tr["run_id"]))
    # F8: duplicate tickets after a retry / F13: equal priority for identical claims
    tk = agent.store.tickets(); live = [x for x in tk if x["state"] != "rolled_back"]
    dup = sum(1 for x in live if x["claim_id"] == "CLM-2026-011")
    pr = {x["claim_id"]: x["priority"] for x in live}
    special = dict(F8_live_tickets_for_CLM_011=dup, F8_pass=dup == 1,
                   F13_priority_008_vs_009=f"{pr.get('CLM-2026-008')} vs {pr.get('CLM-2026-009')}",
                   F13_pass=pr.get("CLM-2026-008") == pr.get("CLM-2026-009"),
                   F12_tool_calls_T64=next(r["tool_calls"] for r in rows if r["id"] == "T64"))
    with open(os.path.join(HERE, f"results_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows, special, agent

def summarise(rows, special):
    n = len(rows); lat = sorted(r["latency_ms"] for r in rows)
    ret = [r for r in rows if r["retrieval_hit"] is not None]; ans = [r for r in rows if r["grounded"] != ""]
    esc = [r for r in rows if r["got"] in ("ESCALATE", "HANDOFF", "NOT_FOUND", "PENDING_HUMAN", "FAILED_SAFE")]
    cats = {}
    for r in rows: cats.setdefault(r["category"], []).append(r["success"])
    return dict(cases=n, task_success=round(sum(r["success"] for r in rows) / n, 3),
        retrieval_hit_rate=round(sum(r["retrieval_hit"] for r in ret) / len(ret), 3), retrieval_cases=len(ret),
        faithfulness=round(statistics.mean(r["grounded"] for r in ans), 3) if ans else None, answer_cases=len(ans),
        p95_latency_ms=lat[int(.95 * (n - 1))], avg_tool_calls=round(statistics.mean(r["tool_calls"] for r in rows), 2),
        total_cost_inr=round(sum(r["cost_inr"] for r in rows), 4), human_route_rate=round(len(esc) / n, 3),
        by_category={k: f"{sum(v)}/{len(v)}" for k, v in cats.items()}, special=special)

def load_test(settings, name, rpm_limit=30, n=200, minutes=10):
    """F15: 200 queries in 10 minutes on results day, bursty (60% in the first 2 minutes), drawn from 25 FAQs.
    Virtual clock; provider allows rpm_limit LLM calls per minute. v0: no cache, no queue -> 429 errors.
    v1: answer cache + deterministic routes + queue with backoff."""
    import random; random.seed(7)
    qs = [r["message"] for r in csv.DictReader(open(os.path.join(HERE, "testset.csv"))) if r["category"] in ("Policy", "Retrieval", "Knowledge")][:25]
    arrivals = sorted([random.uniform(0, 120) for _ in range(int(n * .6))] + [random.uniform(120, minutes * 60) for _ in range(n - int(n * .6))])
    agent = ClaimSathi(settings, name + "_load"); calls_per_min, ok, fail, waits, llm_calls = {}, 0, 0, [], 0
    for t in arrivals:
        q = random.choice(qs); before = agent.llm.complete
        used = []
        agent.llm.complete = lambda *a, _b=before, **k: (used.append(1), _b(*a, **k))[1]
        agent.run(q, "F101"); agent.llm.complete = before
        if not used: ok += 1; waits.append(0.3); continue
        llm_calls += 1; m = int(t // 60)
        if calls_per_min.get(m, 0) < rpm_limit: calls_per_min[m] = calls_per_min.get(m, 0) + 1; ok += 1; waits.append(2.5)
        elif settings.rate_limit_cache:   # queue to the next minute with spare capacity
            mm = m
            while calls_per_min.get(mm, 0) >= rpm_limit: mm += 1
            calls_per_min[mm] = calls_per_min.get(mm, 0) + 1; ok += 1; waits.append(2.5 + (mm * 60 - t if mm > m else 0))
        else: fail += 1
    w = sorted(waits)
    return dict(queries=n, succeeded=ok, failed_429=fail, llm_calls=llm_calls,
                p95_latency_s=round(w[int(.95 * (len(w) - 1))], 1) if w else None, cache_or_rule_answers=n - llm_calls)

if __name__ == "__main__":
    out, failure_dir = {}, os.path.join(HERE, "failure_traces"); shutil.rmtree(failure_dir, ignore_errors=True); os.makedirs(failure_dir)
    for name, st in (("baseline", BASELINE), ("remediated", REMEDIATED)):
        rows, special, agent = run_mode(name, st)
        out[name] = summarise(rows, special); out[name]["load_test_F15"] = load_test(st, name)
        traces = {json.loads(l)["run_id"]: json.loads(l) for l in open(os.path.join(ROOT, "logs", f"traces_{name}.jsonl"))}
        for r in rows:
            if r["failure_ref"]:
                json.dump(dict(test=r, trace=traces[r["run_id"]]), open(os.path.join(failure_dir, f"{r['failure_ref']}_{r['id']}_{name}.json"), "w"), indent=1, default=str)
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
