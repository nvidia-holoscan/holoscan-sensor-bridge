---
paths:
  - "tests/**"
  - "**/tests/**"
  - "test_build/**"
  - "pytest.ini"
  - "**/conftest.py"
  - "ci/*test*.sh"
  - "test-*.sh"
  - "docker/*test*"
  - ".gitlab-ci.yml"
---

# HSB testing rules

- Run focused tests appropriate to the change in a supported environment; use the
  existing build and test procedures rather than assuming local GPU or board access.
- Use `--headless` for pytest runs without a window server. Retain visualization testing
  when the change affects displayed output.
- Preserve existing hardware markers and option gates in `pytest.ini` and
  `tests/conftest.py`. Report hardware-dependent skips separately from successful
  hardware validation.
