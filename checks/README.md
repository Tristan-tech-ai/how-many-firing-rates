# checks: how the numbers and figures of the paper were verified

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. A number passes only if it occurs in such a file to its printed precision, or if a declared formula over such
files reproduces it. This folder holds that code and what it reads.

| folder | content |
|---|---|
| `bindings/` | one entry for each number that no file prints as such: the paper's words around it, and the formula that reproduces it |
| `scripts/` | the number gate (`gate_numbers.py`, with `truth_index.py`, which indexes the numbers of every primary file), the figure gate, and the scripts that build Tables III and IV and the derived numbers (`paper2_derived.py`) |
| `figures/` | the figure scripts; they read primary files and hold no typed-in values |
| `outputs/` | what the build scripts wrote, as used for the paper |
| `inputs/` | primary files that the scripts, bindings and figures read and that the section folders do not already hold |

Table II is built by `sec3-count-law/scripts/build_table2.py`. In a binding, `L(n, regex)` reads a number from entry n of the
registration log (public in the companion repository), `J(path, keys)` a value of a JSON file, and `T(path, regex)` a number of a
text file. The scripts expect the folder layout of the original project, whose paths are listed in [`FILES.md`](FILES.md).

Two kinds of source are not copied. Texts of cited papers are cited in the paper. The simulation data of Barletta and Dytso are
public at https://github.com/ucando83/PoissonCapacity as `capPoisson_darkcurrent*_v1.mat`; `staircases.json`, which the scripts
read, holds the same data read from those files with scipy (registration log, entry 104) and is not copied either.
