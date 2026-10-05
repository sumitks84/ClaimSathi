"""Chat UI: streamlit run app.py   (pick a synthetic user in the sidebar; the role comes from login, never from the chat)."""
import json, streamlit as st
from agent.agent import ClaimSathi
from agent.config import BASELINE, REMEDIATED

st.set_page_config(page_title="ClaimSathi", page_icon="🧾")
mode = st.sidebar.radio("Agent version", ["remediated (v1)", "baseline (v0, guards off)"])
if "agents" not in st.session_state:
    st.session_state.agents = {"remediated (v1)": ClaimSathi(REMEDIATED, "remediated_ui"), "baseline (v0, guards off)": ClaimSathi(BASELINE, "baseline_ui")}
agent = st.session_state.agents[mode]
users = agent.store.users
uid = st.sidebar.selectbox("Logged in as", list(users), format_func=lambda u: f"{users[u]['name']} ({users[u]['designation']})")
fail = st.sidebar.selectbox("Simulate tool outage (F9)", ["none", "notify_claimant", "raise_ticket", "update_status"])
st.title("🧾 ClaimSathi: TA/DA & research-grant claims assistant")
st.caption("Synthetic data only. Answers cite policy sections; the agent recommends, people approve.")
st.session_state.setdefault("chat", [])
for role, text in st.session_state.chat: st.chat_message(role).write(text)
if q := st.chat_input("Ask about TA/DA rules, or 'check my claim CLM-2026-001'"):
    st.chat_message("user").write(q)
    r = agent.run(q, uid, fail_tool=None if fail == "none" else fail)
    st.chat_message("assistant").write(r["answer"])
    with st.expander(f"Trace · {r['outcome']} · {r['trace']['tool_calls']} tool calls · ₹{r['trace']['cost_inr']}"):
        st.json({k: r["trace"][k] for k in ("intent", "outcome", "retrieved", "events", "steps", "in_tok", "out_tok", "latency_ms")})
    st.session_state.chat += [("user", q), ("assistant", r["answer"])]
st.sidebar.markdown("**Approval queue (human)**")
for t in agent.store.tickets():
    if t["state"] == "pending_human_approval":
        st.sidebar.write(f"{t['ticket_id']} · {t['claim_id']} · {t['priority']}")
