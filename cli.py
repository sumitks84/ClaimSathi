"""Terminal chat: python cli.py --user F101 [--baseline]"""
import argparse
from agent.agent import ClaimSathi
from agent.config import BASELINE, REMEDIATED
ap = argparse.ArgumentParser(); ap.add_argument("--user", default="F101"); ap.add_argument("--baseline", action="store_true")
a = ap.parse_args(); agent = ClaimSathi(BASELINE if a.baseline else REMEDIATED, "baseline_cli" if a.baseline else "remediated_cli")
print(f"ClaimSathi ({'v0 baseline' if a.baseline else 'v1'}) as {agent.store.users[a.user]['name']}. Ctrl+C to quit.")
while True:
    try: q = input("\nyou> ")
    except (EOFError, KeyboardInterrupt): break
    r = agent.run(q, a.user)
    print(f"\nClaimSathi [{r['outcome']}]> {r['answer']}\n  trace: {r['trace']['events']}")
