"""check_plos_body.py (16 Oct 2026, P2TEMPLATE): the numbered text of plos/paper2_plos.tex must equal that of paper2_v2_flat.tex
after the only two changes assemble_plos.py makes (figures lose \\includegraphics; [!ht] becomes [!h]). Last line: VERDICT."""
import io
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
a = io.open(os.path.join(H, "paper2_v2_flat.tex"), encoding="utf-8").read()
b = io.open(os.path.join(H, "plos", "paper2_plos.tex"), encoding="utf-8").read()
A = a[a.index("\\section{"):a.index("\\end{document}")]
A = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}\n?", "", A).replace("\\begin{figure}[!ht]", "\\begin{figure}[!h]")
B = b[b.index("\\linenumbers") + len("\\linenumbers"):b.index("\\nolinenumbers")]
ab_a = a[a.index("\\begin{abstract}") + 16:a.index("\\end{abstract}")].strip()
ab_b = b[b.index("\\section*{Abstract}") + 19:b.index("\\section*{Author summary}")].strip()
def nocom(t):   # 16 Oct 2026: the submission file is written without TeX comments; compare without them on both sides
    keep = []
    for x in t.split("\n"):
        m = re.search(r"(?<!\\)%", x)
        if m is None:
            keep.append(x)
        elif x[:m.start()].strip():
            keep.append(x[:m.start()] + "%")
    return "\n".join(keep).strip()
A, B, ab_a, ab_b = nocom(A), nocom(B), nocom(ab_a), nocom(ab_b)
ok = A.strip() == B.strip() and ab_a == ab_b
print("numbered text %d / %d characters, equal %s; abstract equal %s" % (len(A.strip()), len(B.strip()), A.strip() == B.strip(), ab_a == ab_b))
print("VERDICT check_plos_body: %s" % ("PASS" if ok else "FAIL"))
