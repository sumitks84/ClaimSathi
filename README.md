# ClaimSathi

A claims assistant for faculty research-grant and TA/DA travel claims. It answers policy questions with citations, checks a claim against the rules, drafts it for submission, and sends anything about money or exceptions to a person. It never approves or pays.

Capstone for AI in Business Solutions, IIM Bodh Gaya. All policies, users, grants, and claims are synthetic.

## Run it

You need Python 3.10 or newer. The core uses only the standard library.

```bash
python eval/make_testset.py      # builds eval/testset.csv (68 cases)
python eval/run_eval.py          # baseline v0 vs remediated v1, plus the 200-query load test
python cli.py --user F101        # chat in the terminal (add --baseline for v0)
pip install streamlit && streamlit run app.py   # optional web UI
```

The eval writes `eval/results_baseline.csv`, `eval/results_remediated.csv`, `eval/summary.json`, and one trace per failure into `eval/failure_traces/`. Every run also writes `logs/traces_*.jsonl` and `logs/runs_*.csv`.

## Live model (free tier)

The default is `offline`: a deterministic simulated model with deliberately planted weaknesses, so the eval is free and repeatable. To use a real model, copy `.env.example`, pick a provider, and export the key:

```bash
export LLM_PROVIDER=groq GROQ_API_KEY=...      # or gemini / openrouter
python eval/run_eval.py
```

Free-tier model names and rate limits change, so check the provider's list first. Live runs are where you get real latency numbers. The offline numbers measure the logic, not the model.

## Layout

| Path | What it is |
|---|---|
| `kb/` | Six policy documents (handbook, 2026 circular, grant guidelines, SOP, plus two staff-only) with version and access front-matter |
| `data/` | Users, grants, claims |
| `agent/retrieval.py` | Hybrid BM25 + trigram search, version filter, access control, conflict detection |
| `agent/guardrails.py` | Input, retrieval, output, and action guards |
| `agent/rules.py` | Deterministic rule engine R1 to R10. Money is never left to the LLM |
| `agent/tools.py` | Allow-listed tools, idempotency, rollback, loop limit, fair priority |
| `agent/agent.py` | Router and state graph (`docs/state_diagram.png`) |
| `agent/config.py` | `BASELINE` (all guards off) and `REMEDIATED` settings; each fix is a toggle |
| `eval/` | Test set generator, runner, results, failure traces |

## n8n wrapper (deployment plan)

For deployment, n8n would call the agent through a single endpoint. The flow is Webhook, then an HTTP Request to `agent.run(question, user_id)`, then a branch on `outcome` (answered / drafted / human), then an email or ticket to Accounts for anything marked human. The agent itself stays in Python, so the eval runs the same code that ships.

## Honest limits

- The 100% remediated score is on our own test set, after we fixed what it found. A hold-out set is the next step.
- The baseline process figures (20 min per claim, 30% returned, 25 days) are assumptions until we interview the Accounts section.
- The offline model is simulated. Re-run on Gemini Flash or Llama 3.3 70B before quoting any latency.
