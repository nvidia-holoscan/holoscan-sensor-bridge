# Holoscan Sensor Bridge user guide (Fern)

User guide pages are committed as **MDX** (`.mdx`) in this directory, alongside images.
Fern configuration lives under `fern/` (`fern.config.json`, `docs.yml`, `index.yml`,
`assets/`, `dist/`).

## Edit documentation

1. Change `*.mdx` files here (and images alongside them).
2. Update sidebar entries in `fern/index.yml` when adding or renaming pages.
3. Build and validate docs (default: Docker, then `fern check --local`):

   ```bash
   docs/make_docs.sh
   ```

4. Preview locally (uses Docker by default; open http://localhost:3000):

   ```bash
   docs/make_docs.sh --preview
   ```

Use `--no-docker` to run validate, build, or preview on the host instead (requires local `fern`).
Pass `--docker` to force the docs container for preview.

### Fern authentication

Remote previews and production publishing require Fern auth. Log in once from the repo root:

```bash
docs/make_docs.sh --login
```

Inside Docker, Fern uses **device-code** auth (browser OAuth callbacks on `localhost` are not
reachable from the host). Open the URL printed in the terminal, authorize, then paste the code.
On the host (`--no-docker --login`), the normal browser flow works.

Alternatively set `FERN_TOKEN` or run `fern login` on the host; credentials in `~/.fern/` are
mounted into the docs container automatically.

Pass `--skip-fern-check` to run the pipeline only (no Fern validation).

GitLab CI provides the active documentation validation and preview path:

- **Validation:** `hololink-user-guide` runs `fern check --local` on the combined site,
  then runs the documentation fixture suite in the docs image. Host Python does not
  need PyYAML.
- **Preview:** documentation-changing MRs publish a Fern preview after validation and
  fixture tests succeed. The MR shows a **View app** link under **Environments**.
  Validation and preview jobs exclude stress and nightly pipelines.
- **MR comment:** the preview job also posts its link when `GITLAB_HSB_API_TOKEN` or
  `GITLAB_API_TOKEN` is available.

The Jenkins `x86_64` closure contains compatible documentation steps, but the current
merge pipeline dispatches only `igx`. Its documentation steps are inactive. Jenkins
scheduling and hardware jobs remain unchanged.

Set `FERN_TOKEN` in GitLab (**Settings → CI/CD → Variables**, masked) and in Jenkins (credential ID `FERN_TOKEN`) so previews can be published. For MR comments (in addition to the environment link), add `GITLAB_HSB_API_TOKEN` — the personal access token from Jenkins credential `GITLAB_REGISTRY_CREDENTIALS_ID3`.

## Fern CLI

Install the pinned CLI version (matches `fern/fern.config.json`):

```bash
npm install -g fern-api@5.74.1
```

Fern's [organization version policy](https://buildwithfern.com/learn/cli-api-reference/cli-reference/version-policy)
can select a different effective CLI without changing the project pin. Record the
configured version and the runtime version reported by the command. Keep historical
archive provenance unchanged; it describes the original preparation. New snapshots
must record their actual tooling and pass the supported-version checks.

Run Fern commands from **`docs/user_guide/fern`** (where `fern.config.json` lives):

```bash
cd docs/user_guide/fern
fern check --local --warnings
fern docs dev
```

Or use the build script from the repo root (Docker or `--no-docker`).

## C++ API reference

Fern generates the C++ API reference from the headers in the current checkout.
Generate it explicitly with:

```bash
docs/make_docs.sh --with-library-mdx
```

Local generation requires a running Docker-compatible runtime because Fern starts its
C++ parser in a sibling container. Builds with an explicit `FERN_TOKEN` generate the
reference automatically; local checks without the explicit flag use a placeholder page.
Pass `--skip-library-mdx` to suppress automatic generation.

## Publish preview

Remote preview (requires login):

```bash
docs/make_docs.sh --archive-dir /path/to/doc_archives \
  --with-library-mdx --publish-preview --preview-id <owned-preview-id>
```

Or from the repo root:

```bash
docs/make_docs.sh --archive-dir /path/to/doc_archives --publish-preview --preview-id <branch-name> --force
```

## Publish production

Publish to the public site (`docs.nvidia.com/holoscan/sensor-bridge`):

```bash
docs/make_docs.sh \
  --archive-ref "$(cat docs/archive-ref.txt)" \
  --archive-repository https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git \
  --with-library-mdx --publish
```

Run this from the repository root after the reviewed archive commit is available
in the explicit archive repository. A local repository containing that exact commit
can also be supplied. The pin selects the immutable input; the command does not
publish or update the archive branch.

This runs the docs pipeline, then `fern generate --docs`. Fern prompts for confirmation
before publishing to production; choose **Yes** to continue or **No** to cancel. Requires
`fern login` or `FERN_TOKEN` with permissions to publish for the `nvidia` Fern organization.

Committed Fern theme assets live in `fern/dist/` (`output.css`, `output.js`,
`index.html`) and are referenced from `docs.yml`.

## Preserve 2.7 when publishing 3.0

The combined site has **3.0 (Latest)**, marked GA, and a directly selectable **2.7**.
Version 2.7 has no availability badge; the label does not set a support or EOL policy.
Older releases remain on the Archives page. Version 2.7 uses the stable `2-7` slug.
Existing `2-7-latest` links redirect to that version.

For the 3.0 release, reuse the accepted 2.7 snapshot selected by `docs/archive-ref.txt`.
Do not capture 2.7 again, append it or change the pin just to build or preview 3.0.
The pinned-input commands below extract and verify the existing archive.

Before a later default-release change, the designated HSB release engineer freezes
the outgoing guide, APIs and assets with provenance. A documentation maintainer
independently reviews the snapshot and combined preview. The engineer appends the
accepted snapshot without changing existing versions and proposes a reviewed pin
update. The release owner authorizes public archive mirroring and production
publication. Normal builds and CI only read the pinned archive.

The current assembly and validator support `v2.7/` explicitly. Add and review
support for a future archive version before preparing or pinning it; the current
helper is not a generic future-version archiver. The assembler also fixes the
current selector to `3.0 (Latest)`; changing `docs.yml` alone will not update it.
Derive that entry from source configuration in a reviewed follow-up before a later
version release.

A version selector does not freeze content. The accepted 2.7 snapshot is retained
under `v2.7/` on `doc_archives`. A newly supported archive version must include
its authored pages, assets, navigation, and all published generated APIs. Do not generate
historical APIs from the 3.0 checkout. Normal builds check and copy the snapshot;
they do not change it or push an archive branch.

Supply the reviewed snapshot to a local check or preview:

```bash
docs/make_docs.sh --archive-dir /path/to/doc_archives --preview
```

For an immutable Git input, name the full archive commit and repository explicitly:

```bash
docs/make_docs.sh \
  --archive-ref <40-character-archive-commit> \
  --archive-repository <archive-repository-url> \
  --with-library-mdx
```

There is no implicit internal archive URL. A public checkout must use a public
archive repository or a supplied local snapshot. Exporting the source branch does
not export `doc_archives`; release owners must make that input available separately.
A local check without archive options checks only the current source documentation.

`docs/archive-ref.txt` pins the reviewed archive used by CI. Internal jobs fetch
that exact commit through their existing checkout remote. Public release owners
must mirror the same archive commit to the public repository before relying on
a public Git archive input; a supplied local snapshot also works. For example:

```bash
docs/make_docs.sh \
  --archive-ref "$(cat docs/archive-ref.txt)" \
  --archive-repository https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
```

The initial 2.7 snapshot captures the published guide and emulator API. Its
provenance records the served page and asset hashes, repairs, and unknown
original deployment and API-generation commits. It does not claim that the
historical tag reproduces the published site.

The builder generates current APIs before assembly, then prepares a disposable
site with both versions. Use `--output-dir /path/to/new-directory` to retain that
site for review. The output must not already exist. The assembled project uses
its prepared API pages and has no remote library input.

Fern 5.74.1 can skip image rewriting after a Markdown-escaped angle bracket.
Assembly converts those literal brackets to equivalent HTML entities in the staged
archive pages. A raw `<` inside code can also hide later Markdown images from the
scanner. Assembly converts only those affected images to equivalent HTML `<img>`
tags, preserving their alt text and asset targets. Fenced and inline code stay unchanged. The frozen input stays intact.
The staged `v2.7/provenance.json` records the rendering transformation and changed
file hashes. `fern/archive-source-v2.7.json` retains the exact original manifest.

Remote preview and production commands require the archive and real current API
pages. They run Fern validation on the assembled project before any preview deletion
or publication. Placeholders and unresolved Git LFS pointers are rejected. For example:

```bash
docs/make_docs.sh --archive-dir /path/to/doc_archives \
  --with-library-mdx --publish-preview --preview-id <owned-preview-id> --force
```

Production uses `--publish` instead of `--publish-preview`. It is a separate release
operation. Archive preparation and local preview do not publish production documentation.

### Snapshot preparation and provenance

This is maintenance preparation, not the normal 3.0 build path. The `snapshot`
command packages an already prepared tree; it does not discover sources, capture
a website or generate historical APIs. No website-capture client is distributed
in this repository. A fresh capture needs explicit manual page/API/asset acquisition,
normalization, hash inventories and comparison with the published site.

For a source build, generate and post-process the API from the verified historical
source revision using compatible local-input tooling. For a publication capture,
keep the captured API content instead. Record the actual source/tooling revisions,
public refs and comparison evidence; a matching version number alone is insufficient.
Supply that reviewed provenance JSON to package the prepared tree:

```bash
python3 docs/scripts/archive_hololink_docs.py snapshot \
  --source-guide /path/to/verified-2.7/docs/user_guide \
  --output /path/to/new-doc-archives/v2.7 \
  --provenance /path/to/reviewed-provenance.json
python3 docs/scripts/archive_hololink_docs.py check \
  --archive-dir /path/to/new-doc-archives/v2.7
```

Native preparation (`--no-docker`) and the standalone snapshot tool require PyYAML
(`python3 -m pip install PyYAML==6.0.3` in a local virtual environment). The default
Docker workflow uses the container's PyYAML and needs no host installation. Keep
Fern at 5.74.1.

A source-build provenance record requires `kind: "source-build"`, `documentation_commit`, `api_commit`,
`tooling_commit` (full Git commits), `public_source_ref`, `fern_version`, and
`evidence` links to the publication comparison. A publication-capture record requires
`kind: "publication-capture"`, `captured_at`, a complete `source_urls`/hash inventory,
`fern_version`, and `evidence`.
Record unknown original generation revisions as unknown; do not infer them from
an unrelated release tag. The helper adds SHA-256 hashes for every snapshot file.
The record describes the evidence; creating it does not replace independent review.
Generated current APIs also carry a source stamp. It records the full source commit,
configured input-tree hashes, parser configuration and output hashes. Reused output
must match the current input files, including uncommitted header changes.

The historical `2.7.0` tag alone is not the published-site baseline: its Fern label
and API configuration differ from the current 2.7 site. Establish the inputs and
compare the page, asset, navigation, and API inventories before accepting a snapshot.
Check representative hosted pages, API signatures, source links and old URLs in the
combined preview. A successful HTTP response can be the fallback homepage, so also
check the expected page content and hydrated sidebar.
