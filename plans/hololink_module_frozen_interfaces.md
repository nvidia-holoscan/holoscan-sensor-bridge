# `release_version` markers for the `hololink_module` interface freeze

## Context

The 2.7.0 freeze was recorded as a `Frozen: 2.7.0` line in each released class's header
comment, plus a matching entry in `hololink_module/INTERFACES.md` and a
`"Frozen: 2.7.0"` pybind docstring. Nothing reads any of that. The rule it records — a
released interface's methods are never added, removed, reordered, or re-signatured — is
enforced entirely by reviewer attention.

There are two separate gaps, and they need two separate mechanisms:

1. **A new interface can ship without a freeze marker.** A missing comment is invisible.
2. **A class that carries a marker can still be edited.** The marker says *when* the
   interface froze; it says nothing about *what* froze, so no marker of any kind can
   catch an added method on its own.

Moving the marker from a comment to a `static constexpr const char* release_version`
member alongside `type_id` closes (1) directly — the framework can require the member at
compile time, so omitting it is a build error rather than an omission nobody sees. It
does not close (2) by itself; what it does is make (2) mechanizable, by giving a checker
a machine-readable answer to "which classes are guarded, and against which release." What
reads it — a reviewer, a CI check, a tool — is deliberately not settled here; deliverable 4
states what the release process must do with the marker, and mechanizing that is deferred.

Adding a `static constexpr` data member to an already-frozen class changes neither the
object layout nor the vtable, so tagging the released 2.7.0 set is not itself a change to
those interfaces.

The 25 tagged classes and their `type_id`s are listed in `hololink_module/INTERFACES.md`
§ 2.7.0; this plan does not restate them. Their definitions are unchanged by this work.

## Deliverables

### 1. `release_version` member replaces the `Frozen:` comment

Each released class gains the member next to `type_id`, and the `Frozen: <version>` line
comes out of the comment above it:

```cpp
class HololinkInterfaceV1
    : public ConfigurableService<HololinkInterfaceV1>,
      public std::enable_shared_from_this<HololinkInterfaceV1> {
public:
    static constexpr const char* type_id = "hololink.v1";
    static constexpr const char* release_version = "2.7.0";
```

Classes that are not locator services (`I2cLockV1`, `CsiConverterV1`,
`RoceReceiverFrameInfoV1`, `LinuxReceiverFrameInfoV1`) carry `release_version` with no
`type_id`, exactly as they carry the comment today.

A new header `hololink_module/host/include/hololink/module/release.hpp` declares the two
sentinels:

```cpp
/* Value of an interface's release_version when no release has published it
 * yet. Such an interface is editable in place with its callers updated in the
 * same commit. */
inline constexpr const char* UNRELEASED = "unreleased";

/* Value of release_version on an interface that application code must not
 * use. A TEST_ONLY interface exercises the framework and lives under tests/;
 * no release will ever freeze it. */
inline constexpr const char* TEST_ONLY = "test-only";
```

The two are not interchangeable: `UNRELEASED` says "not published *yet*", so a release
may freeze it, and requirement 3 below asks about every one of them. `TEST_ONLY` says
"never published", so no release ever asks.

`service.hpp` includes it, so every service class picks it up transitively; the four
non-service headers include it directly.

**Implementations inherit the marker; only interfaces declare it.**
`module_core::CoeDataChannelV1` derives from `CoeDataChannelInterfaceV1` and reports
`"2.7.0"` without declaring anything. That fixes the reading of the member: it names the
release that froze the *contract being served*, not a promise about the declaring class.
Implementations remain editable — they are override targets, not the application boundary
(per `plans/hololink_module_interface_versions.md` § "Not frozen, deliberately") — and
what records that is where they live, not a marker each one repeats. The classes that
declare their own are those with no interface base to inherit from: a new interface, and
the `TEST_ONLY` fixtures under `tests/`.

### 2. Compile-time check in the framework

One `static_assert`, placed in **member function bodies** (`Service<Derived>` is a CRTP
base, so `Derived` is incomplete in the base's class body and complete by the time these
bodies are instantiated): a `has_release_version<Derived>` detection idiom, asserted in
`Service<Derived>::get_service` (both forms), `Service<Derived>::for_each_type_id`, and
`ServicePublisher<T>::publish`. Every service published or looked up in-tree hits at
least one of these, so a class that reaches the locator with no marker in reach does not
compile. In practice that means a new interface; an implementation inherits one.

Two gaps stay open at compile time, and both fall to the release process (deliverable 4,
requirement 2):

- An interface neither published nor consumed anywhere in the tree is never instantiated,
  so nothing asserts on it.
- A future `FooInterfaceV2 : FooInterfaceV1` that forgets to redeclare would silently
  report the release that froze V1. No compile-time rule can catch this, because an
  implementation of a frozen interface is structurally identical to such a V2 — both
  derive from it, both carry their own `type_id` and `ServiceAlias` — and the
  implementation case is now the intended one. What separates them is the tree the class
  is declared in, which a header scan can see and the compiler cannot.

### 3. Python: a class attribute, not a docstring

The 17 `py::class_` sites that carry a `"Frozen: 2.7.0"` docstring drop that argument and
set real attributes after the class is defined:

```cpp
c.attr("type_id") = ad::HololinkInterfaceV1::type_id;
c.attr("release_version") = ad::HololinkInterfaceV1::release_version;
```

(`type_id` only where the class has one.) One source of truth instead of a hand-copied
string, and the value is available to `help()`, to generated docs, and to a test.

### 4. Documentation and the release process

**Documentation.**

- `INTERFACES.md` preamble: the marker is the `release_version` member, not a comment.
- `plans/hololink_module.md` § Versioning (~line 969): same correction.
- The module device tutorial (`docs/user_guide/module_device_tutorial.mdx`) walks a
  partner through writing an interface header; its interface example needs
  `release_version = UNRELEASED`, and its implementation example needs a sentence saying
  the marker is inherited rather than repeated.
- `RELEASE_NOTES.md`: one line under the next release noting `release_version` as a new
  public attribute on the versioned classes.
- `hololink_module/README.md`: a "Freezing an interface at a release" section stating the
  requirements below, so the release owner has them in one place.

**What the release process must include.** These are requirements on the release, not a
procedure — how they get carried out (by hand, by a script, by CI) is deferred. A release
that skips any of them ships a `release_version` that lies about what it froze, which is
worse than the comment it replaced, because downstream tooling would trust it.

1. **Audit before stamping.** Confirm no interface already carrying a released version has
   changed since that release. `git diff <last-release-tag> HEAD` over
   `hololink_module/host/include/` and `hololink_module/module/*/include/` is the crude
   form of this check. If a frozen interface did change, the release is blocked until the
   change is either reverted or re-shaped as a new versioned class — freezing a new set on
   top of a broken one compounds the break.

2. **Every service interface declared in the public interface trees declares its own
   marker.** Scan `hololink_module/host/include/hololink/module/` and
   `hololink_module/module/*/include/hololink/module/*/`, and for each class reachable
   through the service locator — one carrying a `type_id`, or one an interface hands to
   application code such as `I2cLockV1` or `RoceReceiverFrameInfoV1` — confirm it
   declares `release_version` (whether `UNRELEASED` or a released version) rather than
   inheriting one. This scan is what makes the marker set complete: the compile-time
   assert in deliverable 2 misses a class that is never instantiated, and it cannot
   distinguish a new `FooInterfaceV2` that forgot to redeclare from an implementation
   that inherits on purpose. Declaration site is the distinction, so the scan is the
   check.

   Two exclusions, so the audit has one reading:

   - **Implementations are out of scope, by rule and by path.** They inherit their
     interface's marker and declare none (requirement 5). The scan paths already exclude
     them — `hololink_module/module/core/` has no `include/` subtree, and the per-board
     implementation headers sit beside their module's sources rather than under
     `include/` — so do not widen the scan to `hololink_module/module/` to "catch" them.
     A class in those files with no marker of its own is correct, not a finding.
   - **Framework and data classes carry no marker at all.** `Adapter`,
     `EnumerationMetadata`, `Module`, `Publisher`, `Watchdog`, `ServiceBase` and their
     kind live in the scanned host tree but are not locator-reachable interfaces, so
     nothing freezes them through this mechanism.

3. **Stamp what the release publishes.** Each class the release publishes moves from
   `release_version = UNRELEASED` to the release's version string, in the same commit
   series as the release, before the tag is cut. The stamped value is a claim about which
   release published the interface; a stamp applied after the fact, or naming a version
   that never shipped, makes the marker unusable.

4. **Never re-stamp.** A class already naming a release keeps that value forever. If it
   needs to change, the answer is a new versioned class, never a new version string on the
   old one.

5. **Never stamp an implementation, and never stamp `TEST_ONLY`.** Classes under
   `hololink_module/module/core/` and the per-board implementation headers inherit their
   interface's marker and declare none; adding one to an implementation asserts a freeze
   over code that is meant to churn. A `TEST_ONLY` fixture is not a candidate at all —
   it is not application surface, so no release publishes it.

6. **Record the newly frozen set in `INTERFACES.md`.** One new `## <version>` section,
   newest first, with an entry per newly frozen class: class name, header path, `type_id`
   (or "not a locator service"), and a one-to-two-sentence description drawn from the
   class's own header comment. Earlier sections are historical record and are not edited.

7. **Account for anything left `UNRELEASED` in a public header.** Such a class still ships
   and is still reachable by applications; leaving it unfrozen is a decision to record in
   the release notes, not an oversight to leave silent.

8. **Note the freeze in `RELEASE_NOTES.md`**, naming the newly frozen interfaces and
   pointing at `INTERFACES.md`.

## Out of scope

- Automated API-doc generation from `INTERFACES.md` — still not wired up.
- **Mechanizing the release process.** Deliverable 4 states what a release must do with
  the marker; who or what carries it out — a checklist, a script, a signature-fingerprint
  tool, a CI gate — is a separate problem, deliberately left for later. Nothing enforces
  the freeze automatically today.
- The C ABI (`abi_check.h`, `HOLOLINK_MODULE_API_VERSION`). Unchanged; the freeze covers
  C++ interface shape, not the loader's ABI handshake.
- Freezing anything currently unfrozen, and any change to a frozen interface's content.
- Cross-`.so` runtime enforcement. A module built against an older frozen header is
  already handled by the ABI check plus `type_id` resolution.

## Verification

1. `cmake` + `make` clean; `ctest` passes. The C++ change is a `constexpr` member plus
   `static_assert`s, so a clean build is most of the bar.
2. Negative checks, run by hand during implementation and recorded with the change:
   a new interface with no marker to declare or inherit → compile error; an
   implementation deriving from a released interface and declaring nothing → compiles,
   reporting its interface's release.
3. `grep -rn "Frozen:" hololink_module/` is empty, and the 25 classes carrying
   `release_version = "2.7.0"` match the `## 2.7.0` section of `INTERFACES.md`.
4. `python3 -c "import hololink_module; print(hololink_module.HololinkInterfaceV1.release_version, hololink_module.HololinkInterfaceV1.type_id)"`
   prints `2.7.0 hololink.v1`.
5. `git diff` over the public include trees shows only added `release_version` lines and
   removed `Frozen:` lines — no signature touched. This is the same check
   `plans/hololink_module_interface_versions.md` used to establish the freeze, applied to
   the change that moves the marker from comment to code.
