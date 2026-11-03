# HSB agent guidance

## Project guidance

- HSB means Holoscan Sensor Bridge. Its CMake project is named `hololink`.
- Read [README.md](README.md) for supported build environments and
  [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements.
- Consult relevant task-specific guidance under [skills/](skills/) when working on
  setup, testing, applications, sensors, or FPGA IP.

## Prose

- Follow ASD-STE100 to write plain, concise documentation, code, comments, and
  conversations. Use short sentences, common words, and one instruction per sentence.
- Preserve accuracy and precise semantics.
- Use American English spelling in documentation, comments, and log messages.

## Scoped rules

Read the matching rules below when they apply, including when your agent does not
automatically load `.agents/rules/`.

- When modifying `CMakeLists.txt`, `**/CMakeLists.txt`, `**/*.cmake`, or `cmake/**`,
  follow [CMake rules](.agents/rules/cmake.md).
- When planning or running tests, or modifying `tests/**`, `**/tests/**`,
  `test_build/**`, `pytest.ini`, `**/conftest.py`, `ci/*test*.sh`, `test-*.sh`,
  `docker/*test*`, or `.gitlab-ci.yml`, follow
  [testing rules](.agents/rules/testing.md).

## Contributions and validation

- Follow the formatting rules enforced by `scripts/lint/lint.sh`. Keep formatting
  changes focused; inspect the diff if running the repository-wide formatter.
- Report which checks ran, their results, and any environment limitations. Do not
  present skipped or unavailable checks as passing.
