"""Settings and guard toggles. BASELINE = the deliberately broken v0 agent; REMEDIATED = v1 after fixes.
Every failure in the Failure & Remediation Report maps to one toggle below, so each fix can be switched off to
reproduce the failure and switched on to verify the remedy."""
import os
from dataclasses import dataclass, replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass(frozen=True)
class Settings:
    # ---- LLM (offline = deterministic simulated model; groq / gemini / openrouter = live free tiers)
    provider: str = os.getenv("LLM_PROVIDER", "offline")
    small_model: str = os.getenv("SMALL_MODEL", "llama-3.1-8b-instant")      # classification fallback
    large_model: str = os.getenv("LARGE_MODEL", "llama-3.3-70b-versatile")   # answer drafting
    price_in_per_m_inr: float = 9.0     # assumed blended price (INR per 1M input tokens) for cost projection
    price_out_per_m_inr: float = 35.0   # assumed INR per 1M output tokens
    # ---- Retrieval
    top_k: int = 4
    relevance_threshold: float = 0.30    # F2 fix: below this, say "not found" instead of answering
    version_filter: bool = True          # F1 fix: drop sections superseded by a newer circular
    hinglish_normalise: bool = True      # F4 fix: Hinglish / slang / synonym expansion
    conflict_check: bool = True          # F3 fix: flag two current docs giving different figures
    acl: bool = True                     # F7 fix: role- and owner-based access control
    # ---- Input guardrails
    injection_guard: bool = True         # F5 / F6 fix
    pii_mask: bool = True
    offtopic_guard: bool = True
    distress_handoff: bool = True        # F14 fix
    # ---- Output guardrails
    grounding_check: bool = True
    commitment_filter: bool = True       # F10 fix (output side)
    # ---- Action guardrails
    tool_allowlist: bool = True          # F10 fix (action side): approve_claim is never callable
    idempotency: bool = True             # F8 fix
    saga_rollback: bool = True           # F9 fix
    max_tool_calls: int = 10             # F12 fix (6 was too tight: submit flow needs 7; caught by test set)
    loop_detection: bool = True          # F12 fix
    # ---- Reasoning / fairness
    exception_rules: bool = True         # F11 fix
    bill_match_rule: bool = True         # F16 (bonus) fix
    fair_priority: bool = True           # F13 fix: priority by age and amount, never by designation
    # ---- Load
    rate_limit_cache: bool = True        # F15 fix

REMEDIATED = Settings()
BASELINE = replace(Settings(), relevance_threshold=0.0, version_filter=False, hinglish_normalise=False,
    conflict_check=False, acl=False, injection_guard=False, pii_mask=False, offtopic_guard=False,
    distress_handoff=False, grounding_check=False, commitment_filter=False, tool_allowlist=False,
    idempotency=False, saga_rollback=False, max_tool_calls=50, loop_detection=False, exception_rules=False,
    bill_match_rule=False, fair_priority=False, rate_limit_cache=False)
