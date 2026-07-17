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

4. Preview locally (uses host `fern` by default — open http://localhost:3000):

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

CI publishes remote Fern previews and surfaces the link on merge requests:

- **GitLab CI** (MR with `scope::docs` label): publishes a Fern preview; the MR shows a **View app** link under **Environments** (no extra token required)
- **GitLab CI** (optional): posts an MR comment with the same link when `GITLAB_HSB_API_TOKEN` or `GITLAB_API_TOKEN` is set in CI/CD variables (same PAT Jenkins uses)
- **Jenkins merge pipeline** (MR with `scope::docs` label): same preview flow when the Jenkins x86_64 flow runs
- **GitLab CI** `hololink-user-guide` job: runs `fern check --local` on every pipeline for validation

Set `FERN_TOKEN` in GitLab (**Settings → CI/CD → Variables**, masked) and in Jenkins (credential ID `FERN_TOKEN`) so previews can be published. For MR comments (in addition to the environment link), add `GITLAB_HSB_API_TOKEN` — the personal access token from Jenkins credential `GITLAB_REGISTRY_CREDENTIALS_ID3`.

## Fern CLI

Install the pinned CLI version (matches `fern/fern.config.json`):

```bash
npm install -g fern-api@5.74.1
```

Run Fern commands from **`docs/user_guide/fern`** (where `fern.config.json` lives):

```bash
cd docs/user_guide/fern
fern check --local --warnings
fern docs dev
```

Or use the build script from the repo root (Docker or `--no-docker`).

## Publish preview

Remote preview (requires login):

```bash
cd docs/user_guide/fern
fern generate --docs --preview
```

Or from the repo root:

```bash
sh docs/make_docs.sh --publish-preview --preview-id <branch-name> --force
```

## Publish production

Publish to the public site (`docs.nvidia.com/holoscan/sensor-bridge`):

```bash
sh docs/make_docs.sh --publish
```

This runs the docs pipeline, then `fern generate --docs`. Fern prompts for confirmation
before publishing to production; choose **Yes** to continue or **No** to cancel. Requires
`fern login` or `FERN_TOKEN` with permissions to publish for the `nvidia` Fern organization.

Committed Fern theme assets live in `fern/dist/` (`output.css`, `output.js`,
`index.html`) and are referenced from `docs.yml`.
