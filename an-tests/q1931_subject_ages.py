"""q1931 (15 Oct 2026): subject ages of DANDI 001262 from the DANDI asset metadata (one asset per subject), for task P2REPL.
Facts only; writes data_ext/dandi001262/subject_ages.json {subject: age string}."""
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
assets = json.load(open(os.path.join(HERE, "data_ext", "dandi001262", "assets.json")))
first = {}
for a in assets:
    s = a["path"].split("/")[0]
    first.setdefault(s, a["id"])
out = {}
for s, aid in sorted(first.items()):
    url = "https://api.dandiarchive.org/api/assets/%s/" % aid
    try:
        m = json.load(urllib.request.urlopen(url, timeout=30))
        ages = [w.get("age", {}).get("value") for w in m.get("wasAttributedTo", []) if isinstance(w, dict)]
        out[s] = ages[0] if ages else None
    except Exception as e:
        out[s] = "ERROR " + repr(e)[:80]
json.dump(out, open(os.path.join(HERE, "data_ext", "dandi001262", "subject_ages.json"), "w"), indent=1)
print("subjects %d; with age %d" % (len(out), sum(1 for v in out.values() if v and not str(v).startswith("ERROR"))))
print(sorted(out.items())[:5])
