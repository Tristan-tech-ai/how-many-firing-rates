"""flatten_v2.py: inline every \\input{...} of paper2_main.tex into paper2_v2_flat.tex, so that gates that read one file see the
whole draft."""
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(H, "paper2_main.tex"), encoding="utf-8").read()
s = re.sub(r"\\input\{([^}]*)\}", lambda m: open(os.path.join(H, m.group(1) + ".tex"), encoding="utf-8").read(), s)
open(os.path.join(H, "paper2_v2_flat.tex"), "w", encoding="utf-8", newline="\n").write(s)
print("lines", s.count("\n"))
