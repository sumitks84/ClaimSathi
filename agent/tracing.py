"""Every run is logged: JSONL trace (full) + CSV row (the 'Google Sheet' log in an n8n build)."""
import csv, json, os, time, uuid

class Trace:
    def __init__(self, log_dir, mode, user_id, role, raw_input):
        self.log_dir, self.mode = log_dir, mode; os.makedirs(log_dir, exist_ok=True)
        self.d = dict(run_id=uuid.uuid4().hex[:8], ts=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, user_id=user_id,
                      role=role, input=raw_input, steps=[], events=[], retrieved=[], in_tok=0, out_tok=0, llm_ms=0.0)
        self.t0 = time.time()
    def step(self, kind, name, args, out): self.d["steps"].append(dict(kind=kind, name=name, args=args, out=out))
    def event(self, kind, detail=""): self.d["events"].append(f"{kind}:{detail}")
    def llm(self, model, usage):
        self.d["in_tok"] += usage["in_tok"]; self.d["out_tok"] += usage["out_tok"]; self.d["llm_ms"] += usage["ms"]
        self.step("llm", model, {}, f"{usage['in_tok']} in / {usage['out_tok']} out tokens")
    def finish(self, intent, outcome, answer, s):
        d = self.d; d.update(intent=intent, outcome=outcome, answer=answer,
            latency_ms=round((time.time() - self.t0) * 1000 + d["llm_ms"], 1),
            tool_calls=sum(1 for x in d["steps"] if x["kind"] == "tool"),
            cost_inr=round(d["in_tok"] / 1e6 * s.price_in_per_m_inr + d["out_tok"] / 1e6 * s.price_out_per_m_inr, 5))
        with open(os.path.join(self.log_dir, f"traces_{self.mode}.jsonl"), "a") as f: f.write(json.dumps(d, default=str) + "\n")
        p = os.path.join(self.log_dir, f"runs_{self.mode}.csv"); new = not os.path.exists(p)
        row = {k: d[k] for k in ("run_id", "ts", "mode", "user_id", "role", "input", "intent", "outcome", "latency_ms",
                                 "tool_calls", "in_tok", "out_tok", "cost_inr")}
        row.update(retrieved=";".join(d["retrieved"]), events=";".join(d["events"]), answer=answer[:400])
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
        return d
