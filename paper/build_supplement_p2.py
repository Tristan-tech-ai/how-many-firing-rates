"""build_supplement_p2.py (15 Oct 2026, task P2NOTES): write supplement_p2.tex, the Supplementary Notes (S1 Text) of paper 2, from the
sections of the 4 Oct draft paper2.tex that the rewritten paper no longer carries: the channel model and the count law as stated there
(for the equations the later sections use), neurons more or less regular than Poisson, the baseline rate, short windows (on-off code,
refractoriness, energy-capacity front, Kostal-Shinomoto contrast) and the six-dataset test. supplement_p2.tex is GENERATED: edit this
script or paper2.tex, never the output. Cross-references to sections of the old draft that are not copied are replaced by plain words;
every replacement is listed in REPL and must occur exactly as often as stated, so a change in paper2.tex stops the build.
Usage: py build_supplement_p2.py   (then pdflatex supplement_p2.tex twice)
"""
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "paper2.tex"), encoding="utf-8").read()


def cut(start, stop):
    i = src.index(start); j = src.index(stop, i)
    return src[i:j]


parts = [
    cut("\\section{The neuron as a channel}", "\\subsection{Certified counts}"),
    cut("\\section{Neurons more regular or more variable than Poisson}", "\\section{Discussion}"),
    # 15 Oct 21:1x (P2NOTESAUDIT N1): the baseline section points to the positions of new rates, which only this old appendix holds
    cut("\\section{Where a new rate appears}", "\\FloatBarrier"),
]
# 15 Oct 22:5x (P2CMPFINITE): finiteness for the Conway-Maxwell-Poisson count, after the section on regular counts; source
# supp_cmp_finite.tex (comment lines dropped below with the others)
parts.insert(2, open(os.path.join(H, "supp_cmp_finite.tex"), encoding="utf-8").read())
body = "".join(parts)
body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("%"))

# (old text, new text, expected count)
REPL = [
    ("(Appendix~\\ref{app:birth})", "(Section~\\ref{app:birth})", 1),
    ("degenerate output at silence, at a finite Fisher distance, and the first two", "degenerate output at silence, at a finite Fisher distance. The first two", 1),
    # N3: the Fano range of the old datasets is far from the auditory-nerve fibres of the main text
    ("variability above Poisson never raises it.",
     "variability above Poisson never raises it. The auditory-nerve fibres of the main text, with a median Fano factor of $0.355$, lie\n"
     "far below this range; for them the main text computes the number of rates for a COM-Poisson count matched to each fibre.", 1),
    # N4: link the six-dataset result to the dispersion finding of the main text
    ("these recordings do not show them.",
     "these recordings do not show them. Their median Fano factors, $0.90$ to $1.39$, are close to one, so the regularity that brings\n"
     "most auditory-nerve fibres of the main text within the optimal staircase is absent here. This level-count test also differs in design\n"
     "from the staircase test of the main text. It also lies outside the conditions of the theory in two further ways. The window in which\n"
     "cortical neurons are read is not known, and the number of levels grows with the window. The stimuli were a few conditions chosen\n"
     "by the experimenter, and three datasets come from motor and frontal areas that do not encode a stimulus distribution in Barlow's sense.", 1),
    ("Section~\\ref{sec:long}", "Section~\\ref{sec:countlaw-s}", 1),
    ("The law has not been evaluated at the\ncertified $46$ and $66$ rates of Section~\\ref{sec:cert}.",
     "Its predictions at 25, 47 and 67 rates are reported in the count-law section of the main text.", 1),
    # claim audit item 15: an unpublished working paper is not cited; the next sentence of the text gives the step itself
    ("~\\cite{Firdaus2026a}", "", 1),
    # 15 Oct 23:3x (P2PAPERR): the exact bound of Amendments 2268 and 2296 replaces the wish for one
    ("An exact finite-blocklength bound for a few cases would show whether the on-off\nchoice survives outside the approximation.",
     "An exact finite-blocklength bound answers this for small peak counts. The cases are the Poisson neuron without a spontaneous\n"
     "count at peak counts 4, 5 and 6, blocklengths 5, 10 and 20 and error probabilities $0.1$ and $0.01$. In each, the\n"
     "dependence-testing achievability bound~\\cite{PPV2010} was computed with a certified bracket for 19 on-off laws, 45 graded laws and\n"
     "the capacity-achieving law. The\n"
     "best on-off law reached the larger number of messages in 16 of the 18 cases. A graded law did better only at peak count 6 with\n"
     "error probability $0.01$ and blocklengths 10 and 20, by $0.196$ and $0.130$ nats. The on-off choice therefore survives outside\n"
     "the approximation at these peak counts and gives way first where the peak count is largest and the error smallest.", 1),
]
for old, new, n in REPL:
    got = body.count(old)
    assert got == n, ("replacement count", old, got, n)
    body = body.replace(old, new)
# N2: the main paper writes the COM-Poisson shape parameter as zeta (nu is a signed measure in its appendix)
nu = re.compile(r"\\nu(?![a-zA-Z])")
assert len(nu.findall(body)) == 4, ("nu count", len(nu.findall(body)))
body = nu.sub(r"\\zeta", body)
body = body.replace("\\subsection{The count law}","\\subsection{The count law}\\label{sec:countlaw-s}", 1)
body = body.replace("width=\\columnwidth", "width=0.8\\textwidth")

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", body))
missing = sorted(refs - labels)
assert not missing, ("references without a label in the supplement", missing)

cited = set(k.strip() for m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}", body) for k in m.group(1).split(","))
# 16 Oct 2026 (P2SUPPVANC): Vancouver entries as in the main text (bib_supp_vancouver.tex, data of paper2.tex unchanged)
bib = open(os.path.join(H, "bib_supp_vancouver.tex"), encoding="utf-8").read().strip()
bib = "\n".join(l for l in bib.split("\n") if not l.lstrip().startswith("%"))   # N7: no working notes in the source
bib = "\n".join(re.sub(r"\s*(?<!\\)%.*$", "", l) for l in bib.split("\n"))          # N7: nor trailing comments
items = re.split(r"(?=\\bibitem\{)", bib)
kept = [items[0]] + [it for it in items[1:] if re.match(r"\\bibitem\{([^}]*)\}", it).group(1) in cited]
if not kept[-1].rstrip().endswith("\\end{thebibliography}"):
    kept[-1] = kept[-1].rstrip() + "\n\\end{thebibliography}"
absent = sorted(cited - set(re.findall(r"\\bibitem\{([^}]*)\}", bib)))
assert not absent, ("cited but not in bib_supp_vancouver.tex", absent)
# P2SUPPVANC: numbered in the order of first citation, as in the main text (checked by gate_ref_order_p2.py)
_order = []
for m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}", "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in body.split("\n"))):
    for k in m.group(1).split(","):
        if k.strip() and k.strip() not in _order:
            _order.append(k.strip())
_entries = [it.replace("\\end{thebibliography}", "").rstrip() + "\n" for it in kept[1:]]
_key = lambda it: re.match(r"\\bibitem\{([^}]*)\}", it).group(1)
_entries.sort(key=lambda it: _order.index(_key(it)))
kept = [kept[0]] + _entries + ["\\end{thebibliography}"]

head = r"""% GENERATED by build_supplement_p2.py from paper2.tex; do not edit
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{url}
\usepackage[margin=1in]{geometry}
\newtheorem{theorem}{Theorem}
\newtheorem{conjecture}{Conjecture}
\newtheorem{proposition}{Proposition}[section]
\renewcommand{\thesection}{S1.\arabic{section}}
\begin{document}
\section*{S1 Text. Supplementary Notes}
These notes collect material that supports the main text without being part of its argument. They first give the channel model and the
conjectured count law in the form used below. They then treat counts more or less regular than Poisson counts. For the Conway-Maxwell-Poisson count of the main text they prove
that the optimal input has finitely many rates. They also treat a spontaneous rate,
codes for windows that hold few spikes, a test for discrete levels on six public datasets, and where a new rate appears. Results here carry the same
status words as in the main text.

"""
out = head + body.strip() + "\n\n" + "".join(kept) + "\n\\end{document}\n"
# 16 Oct 2026 (P2SUPPLAYOUT): the figures are taller than a top float allows, so with [t] alone all three went after the references
out = out.replace("\\begin{figure}[t]", "\\begin{figure}[tp]").replace("\\begin{table}[t]", "\\begin{table}[tp]")
open(os.path.join(H, "supplement_p2.tex"), "w", encoding="utf-8", newline="\n").write(out)
print("supplement_p2.tex written: %d lines, cited %d, bibitems %d" % (out.count("\n"), len(cited), len(kept) - 2))
