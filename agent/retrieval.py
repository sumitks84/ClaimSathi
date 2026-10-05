"""Section-based chunking + hybrid retrieval (BM25 keywords + character-trigram vectors), with
version filtering, access control and conflict detection. Pure Python, no paid embedding API."""
import glob, math, os, re
from collections import Counter
from .config import ROOT

STOP = set(("a an the of to in for on and or is are be by with as at from this that it its any all may must per can i my me we do does "
  "what how when which who will under up much get had now right give have take long instead please there about also yes is "
  "tak tha hai karna kya karu mein ka ki ke ko se say says tell").split())
SUFFIX = ("ations", "ation", "ments", "ment", "able", "ings", "ing", "ted", "ed", "es", "s", "e")
def stem(w):
    for _ in range(2):
        for suf in SUFFIX:
            if w.endswith(suf) and len(w) - len(suf) >= 4: w = w[:-len(suf)]; break
    if w.endswith("ll") and len(w) > 5: w = w[:-1]
    return {"submission": "submit", "submiss": "submit"}.get(w, w)
HINGLISH = {  # F4 fix: campus Hinglish / slang -> policy vocabulary
 "kitna": "ceiling", "kitne": "amount", "milega": "reimbursed", "milta": "reimbursed", "kab": "deadline when",
 "jama": "submit submission", "karna": "", "hai": "", "ka": "", "ki": "", "ke": "", "mein": "", "kya": "", "karu": "",
 "kho": "lost missing", "gaya": "lost", "paisa": "amount reimbursement", "rehne": "hotel accommodation",
 "khana": "meals daily allowance", "bhatta": "allowance", "safar": "travel", "yatra": "travel", "der": "late delay",
 "wapas": "return", "din": "day", "max": "maximum ceiling", "limit": "ceiling", "tada": "ta da claims",
 "da": "daily allowance da", "conf": "conference", "regn": "registration fee",
 "cab": "taxi local conveyance", "uber": "taxi local conveyance", "ola": "taxi local conveyance",
 "per": "", "diem": "daily allowance da", "lodging": "hotel accommodation", "stay": "hotel accommodation",
 "flight": "air travel ticket", "cap": "ceiling", "reimb": "reimbursement", "free": "free meals",
 # business glossary (staff vocabulary -> document vocabulary)
 "percentage": "exceed sanctioned", "spent": "expenditure", "move": "re-appropriation between heads", "money": "funds",
 "attach": "documents need", "long": "service standard working days", "settle": "settled", "duration": "years",
 "websites": "third-party booking platforms", "lost": "missing", "rate": "allowance", "trip": "travel", "fare": "air ticket", "urgently": "emergency", "maximum": "up to", "fee": "fees",
}

def tok(s):
    return [stem(w) for w in re.findall(r"[a-z0-9₹]+", s.lower().replace(",", "")) if w not in STOP]

from .rules import CITY
def normalise(q):
    out = []
    for w in re.findall(r"[a-zA-Z0-9₹/]+", q.lower().replace("ta/da", "tada").replace("ta da", "tada")):
        out.append(w); out.extend(HINGLISH.get(w, "").split())
        if (w.capitalize() in CITY or w in ("gaya",)) and re.search(r"hotel|ceil|cap|lodg|stay|rehne|accommodation", q.lower()):
            # entity expansion city -> class, only for hotel questions (class only changes the hotel ceiling)
            out.append(CITY.get(w.capitalize(), "Z").lower() + "-class")
    return " ".join(out)

def trigrams(s):
    s = " " + re.sub(r"\s+", " ", s.lower()) + " "
    return Counter(s[i:i+3] for i in range(len(s) - 2))

class KB:
    def __init__(self, kb_dir=None):
        self.chunks, self.docs = [], {}
        for p in sorted(glob.glob(os.path.join(kb_dir or os.path.join(ROOT, "kb"), "*.md"))):
            txt = open(p, encoding="utf-8").read()
            fm, body = txt.split("---", 2)[1], txt.split("---", 2)[2]
            meta = dict(l.split(": ", 1) for l in fm.strip().splitlines() if ": " in l)
            meta["file"] = os.path.basename(p); self.docs[meta["doc_id"]] = meta
            for sec in re.split(r"\n(?=## )", body)[1:]:
                head, _, text = sec.partition("\n")
                m = re.match(r"## (§[\d.]+)\s*(.*)", head)
                topic = re.search(r"<!-- topic: (\w+) -->", text)
                text = re.sub(r"<!--.*?-->", "", text).strip()
                self.chunks.append(dict(doc_id=meta["doc_id"], section=m.group(1), heading=m.group(2), text=text,
                    topic=topic.group(1) if topic else None, access=meta["access"], title=meta["title"],
                    version=meta["version"], effective_from=meta["effective_from"], ref=f"{meta['doc_id']} {m.group(1)}"))
        self.superseded = set()
        for d in self.docs.values():
            for s in d.get("supersedes", "").split(";"):
                if s.strip(): self.superseded.add(s.strip())
        self._index()

    def _index(self):
        self.tf = [Counter(tok(c["title"] + " " + c["heading"] + " " + c["heading"] + " " + c["text"])) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]; self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t); N = len(self.chunks)
        self.idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        self.tri = [trigrams(c["heading"] + " " + c["text"]) for c in self.chunks]

    def _bm25(self, q, i, k1=1.4, b=.75):
        s = 0
        for w in set(q):
            f = self.tf[i].get(w, 0)
            if f: s += self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
        return s

    def search(self, query, settings, role="faculty", k=None):
        """Returns (hits, events). hits carry a 0-1 hybrid score."""
        events, k = [], k or settings.top_k
        q = normalise(query) if settings.hinglish_normalise else query
        qt, qtri = tok(q), trigrams(q)
        content = [w for w in tok(query) if w not in HINGLISH or HINGLISH[w]]   # words the user actually typed
        mx = max(self.idf.values())
        res = []
        for i, c in enumerate(self.chunks):
            if settings.acl and c["access"] == "staff" and role != "accounts_officer":
                continue
            if settings.version_filter and c["ref"] in self.superseded:
                continue
            bm = self._bm25(qt, i); bm = bm / (bm + 6)
            dot = sum(v * self.tri[i].get(g, 0) for g, v in qtri.items())
            cos = dot / (math.sqrt(sum(v*v for v in qtri.values())) * math.sqrt(sum(v*v for v in self.tri[i].values())) or 1)
            terms = set(tok(c["title"] + " " + c["heading"] + " " + c["text"]))
            qw = set(content) | (set(tok(q)) - set(content))
            # coverage: idf-weighted share of the user's content words found in the chunk; unknown words weigh most,
            # so a question about something the corpus never mentions ("visa", "spouse") scores low (F2 fix)
            cov = sum(self.idf.get(w, mx) for w in set(content) if w in terms or any(stem(x) in terms for x in HINGLISH.get(w, "").split())) / (sum(self.idf.get(w, mx) for w in set(content)) or 1)
            res.append(dict(c, score=round(.45 * bm + .35 * cov + .2 * min(1, cos * 2.5), 3), coverage=round(cov, 2)))
        res.sort(key=lambda r: -r["score"])
        hits = res[:k]
        raw = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in STOP and w not in HINGLISH and len(w) >= 3]
        vocab = {w for i, c in enumerate(self.chunks) if not (settings.acl and c["access"] == "staff" and role != "accounts_officer") for w in self.tf[i]}
        oov = [w for w in raw if stem(w) not in vocab]   # role-scoped: staff-only words don't count for faculty
        if oov: events.append(("OOV_TERMS", ",".join(oov)))
        if settings.version_filter:
            events.append(("VERSION_FILTER", f"{len(self.superseded)} superseded sections excluded"))
        if settings.conflict_check:
            amt = lambda t: set(re.findall(r"₹\s?([\d,]+)", t))
            top = [h for h in hits if h["score"] >= max(settings.relevance_threshold, hits[0]["score"] * .6)] if hits else []
            for i, a in enumerate(top):
                for b2 in top[i+1:]:
                    if a["topic"] and a["topic"] == b2["topic"] and a["doc_id"] != b2["doc_id"] and amt(a["text"]) != amt(b2["text"]):
                        events.append(("CONFLICT", f"{a['ref']} vs {b2['ref']} on {a['topic']}"))
        return hits, events
