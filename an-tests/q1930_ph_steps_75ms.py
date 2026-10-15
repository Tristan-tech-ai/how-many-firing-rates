"""q1930 (15 Oct 2026): task P2WINDOW, registered in Amendment 2277 before any 75 ms statistic was computed. q1926 (Amendment 2272)
re-run with one change, the counting window 0 <= t < 75 ms instead of 50 ms; selection, D, K and the verdict rule are those of q1926
(its functions are used unchanged; only its module constant W is set). Why 75 ms: the PH trials of most recordings end near 80 ms
(the largest spike time per recording has median 0.0798 s; 14.2 percent of recordings reach 0.2 s), so a 200 ms window, the queue's
first choice, is not available; 75 ms is the longest window inside every trial.
VERDICT (H_steps75): PASS if D <= K(A_75, lam_75) in at least 80 percent of the selected fibres; FAIL otherwise; VOID below 20 fibres.
Usage: py q1930_ph_steps_75ms.py [--registration-run]"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1926_ph_steps as S

S.W = 0.075
S.OUT = os.path.join(HERE, "paper_neuron", "q1930_rows.json")
if __name__ == "__main__":
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1930: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = S.registration_run()
        print("VERDICT q1930 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        sys.exit(0 if ok else 2)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = S.main()
    out = buf.getvalue().replace("VERDICT q1926 (Amendment 2272, H_steps)", "VERDICT q1930 (Amendment 2277, H_steps75)")
    out = out.replace("PASS", "PASS").replace("INCOMPLETE", "INCOMPLETE")   # verdict words unchanged; lint: PASS FAIL VOID INCOMPLETE
    print(out, end="")
    sys.exit(code)
