Utils
=====

Small utilities for generating noisy Vietnamese-like text variants and
supporting helpers (tone handling, Telex conversion, keyboard errors).

Modules
-------
- abbreviation.py: simple phrase-to-abbreviation substitutions.
- edit_distance.py: random character insert/delete/substitute/transpose.
- fat_finger.py: keyboard adjacency errors (QWERTY and Vietnamese mobile).
- gen_error.py: dispatcher that applies one error type or all types.
- no_diacritics.py: remove Vietnamese diacritics and convert đ/Đ to d/D.
- region.py: regional phonetic substitutions (north/south variants).
- teencode.py: teen code substitutions and occasional insertions.
- telex.py: Telex raw output or tone-dropping errors.
- tone_utils.py: tone tables and helper functions.

Quick usage
-----------
These modules are plain Python files. If you import from the utils folder
directly, run from that directory or add it to PYTHONPATH.

Example:

```python
from gen_error import generate_error, generate_all_errors

text = "Xin chao moi nguoi"
noisy, error_type = generate_error(text)
print(error_type, noisy)

print(generate_all_errors(text))
```

Notes
-----
- Most functions are stochastic; pass parameters like `intensity`,
  `error_rate`, `num_edits`, or `dialect` to control behavior.
- `gen_error.py` lists valid error types in `ALL_ERROR_TYPES`.
