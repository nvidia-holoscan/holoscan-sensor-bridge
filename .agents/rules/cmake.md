---
paths:
  - "CMakeLists.txt"
  - "**/CMakeLists.txt"
  - "**/*.cmake"
  - "cmake/**"
---

# HSB CMake rules

- Use `${hololink_SOURCE_DIR}` for HSB repository-root source paths,
  `${CMAKE_CURRENT_SOURCE_DIR}` for directory-local sources, and
  `${CMAKE_CURRENT_LIST_DIR}` for module-relative paths.
- Reserve `${CMAKE_SOURCE_DIR}` for paths intentionally tied to the outermost project.
  Preserve standalone-build checks and independently buildable subprojects.
- When changing build behavior, validate affected targets both standalone and within a
  parent project where available. Report configurations that could not be validated.
