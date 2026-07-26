#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Build and validate Holoscan Sensor Bridge Fern documentation.

User guide pages are committed as ``*.mdx`` under ``docs/user_guide/``. Fern config
(``fern.config.json``, ``docs.yml``, ``index.yml``, ``assets/``, ``dist/``) lives under
``docs/user_guide/fern/``. The public C++ API for ``src/hololink/emulation`` is
generated under ``docs/user_guide/fern/generated/``.

With ``--with-library-mdx``, the pipeline runs ``fern docs md generate`` and
post-processes the generated MDX before ``fern check --local`` (local validation),
``fern docs dev`` (``--preview``),
``fern generate --docs --preview`` (``--publish-preview`` for CI), or
``fern generate --docs`` (``--publish`` for production).

Usage (from hololink repo root):

  python3 docs/scripts/build_hololink_docs.py
  python3 docs/scripts/build_hololink_docs.py --preview
  python3 docs/scripts/build_hololink_docs.py --no-docker
  python3 docs/scripts/build_hololink_docs.py --with-library-mdx
  python3 docs/scripts/build_hololink_docs.py --skip-fern-check
  python3 docs/scripts/build_hololink_docs.py --publish
  python3 docs/scripts/build_hololink_docs.py --publish-preview --preview-id my-branch --force
  python3 docs/scripts/build_hololink_docs.py --delete-preview-id my-branch --publish-preview --preview-id main --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DOCS_DIR = SCRIPTS.parent
DOCS_ROOT = DOCS_DIR / "user_guide"
REPO = DOCS_DIR.parent
SOURCE = DOCS_ROOT
DEFAULT_FERN_DIR = DOCS_ROOT / "fern"
FIX_LIBRARY_MDX = SCRIPTS / "fix_generated_library_mdx.py"
DOCKERFILE = DOCS_DIR / "Dockerfile.docs"

ASSET_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
SKIP_SOURCE_DIRS = frozenset({"fern", "scripts", "vale", "_static", "_templates"})
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
EXTERNAL_FERN_DIR_MOUNT = Path("/mnt/fern_dir")
PREVIEW_URL_FILE = ".fern-preview-url"
_PREVIEW_URL_RE = re.compile(
    r"https://[^\s\"'<>)\]]+\.docs\.buildwithfern\.com[^\s\"'<>)\]]*",
    re.IGNORECASE,
)
_FERN_PUBLISH_CANCELLED_RE = re.compile(r"Cancelled by user", re.IGNORECASE)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _iter_source_assets() -> list[Path]:
    assets: list[Path] = []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        if rel.parts and rel.parts[0] in SKIP_SOURCE_DIRS:
            continue
        if any(part in SKIP_SOURCE_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in ASSET_EXTS:
            assets.append(path)
    return assets


def _ensure_lfs_assets() -> bool:
    if not shutil.which("git-lfs"):
        return False
    try:
        _run(["git", "lfs", "pull", "--include=docs/user_guide/**"])
        return True
    except subprocess.CalledProcessError:
        print("warning: git lfs pull failed; images may be missing.", file=sys.stderr)
        return False


def _report_lfs_pointers(paths: list[Path]) -> int:
    if not paths:
        return 0
    print(
        f"\nerror: {len(paths)} image(s) under docs/user_guide/ are Git LFS pointers, not "
        "binary files. Fern cannot display them until LFS objects are downloaded.",
        file=sys.stderr,
    )
    print("Install git-lfs, then from the repository root run:", file=sys.stderr)
    print("  git lfs install", file=sys.stderr)
    print("  git lfs pull --include='docs/user_guide/**'", file=sys.stderr)
    print("\nAffected files:", file=sys.stderr)
    for path in paths[:20]:
        print(f"  - {path.relative_to(REPO)}", file=sys.stderr)
    if len(paths) > 20:
        print(f"  ... and {len(paths) - 20} more", file=sys.stderr)
    return 1


def _check_source_assets(*, allow_lfs_pointers: bool) -> int:
    pointers = [p for p in _iter_source_assets() if _is_lfs_pointer(p)]
    if not pointers:
        return 0
    if allow_lfs_pointers:
        print(
            f"warning: {len(pointers)} image(s) are Git LFS pointers; docs preview may show "
            "broken images.",
            file=sys.stderr,
        )
        return 0
    return _report_lfs_pointers(pointers)


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or REPO, check=check)


# Warnings `fern docs dev` always emits for a local, unauthenticated preview: the
# missing-redirects check needs Fern auth, and the accent/background contrast check
# cannot be disabled via config. Filtered from preview output; genuine warnings pass through.
_SUPPRESSED_PREVIEW_WARNINGS = re.compile(
    r"Missing redirects check skipped|contrast ratio between the accent color"
)


def _run_preview(fern_exe: str, fern_dir: Path, *, port: int) -> int:
    """Run ``fern docs dev``, filtering known unauthenticated-local warnings from output."""
    cmd = [fern_exe, "docs", "dev", "--port", str(port)]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=fern_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if _SUPPRESSED_PREVIEW_WARNINGS.search(line):
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
    return proc.wait()


def _fern_exe() -> str | None:
    return shutil.which("fern")


def _require_fern_auth(*, action: str) -> int | None:
    if _fern_auth_token():
        return None
    print(
        f"error: {action} requires Fern auth (run 'fern login' or set FERN_TOKEN).",
        file=sys.stderr,
    )
    return 1


def _extract_preview_url(output: str) -> str | None:
    urls = [match.group(0).rstrip(".,;)") for match in _PREVIEW_URL_RE.finditer(output)]
    if not urls:
        return None
    for url in urls:
        if "-preview-" in url.lower():
            return url
    return urls[0]


def _docs_site_path(fern_dir: Path) -> str:
    docs_yml = fern_dir / "docs.yml"
    if docs_yml.is_file():
        for line in docs_yml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- url:") or stripped.startswith("url:"):
                value = stripped.split(":", 1)[1].strip()
                if value.startswith("nvidia-holoscan.docs.buildwithfern.com"):
                    suffix = value.split(".docs.buildwithfern.com", 1)[-1]
                    return suffix if suffix.startswith("/") else f"/{suffix}"
    return "/holoscan/sensor-bridge"


def _production_docs_urls(fern_dir: Path) -> tuple[str, str | None]:
    """Return (Fern host URL, custom-domain URL or None) from ``docs.yml``."""
    docs_yml = fern_dir / "docs.yml"
    fern_url: str | None = None
    custom_url: str | None = None
    if docs_yml.is_file():
        for line in docs_yml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- url:") or (
                stripped.startswith("url:") and "custom-domain" not in stripped
            ):
                value = stripped.split(":", 1)[1].strip()
                if value and (
                    ".docs.buildwithfern.com" in value or not value.startswith("http")
                ):
                    fern_url = value if value.startswith("http") else f"https://{value}"
            if stripped.startswith("custom-domain:"):
                value = stripped.split(":", 1)[1].strip()
                if value:
                    custom_url = value if value.startswith("http") else f"https://{value}"
    if not fern_url:
        fern_url = "https://nvidia-holoscan.docs.buildwithfern.com/holoscan/sensor-bridge"
    return fern_url, custom_url


def _fallback_preview_url(*, fern_dir: Path, preview_id: str) -> str | None:
    config_path = fern_dir / "fern.config.json"
    if not config_path.is_file():
        return None
    try:
        org = json.loads(config_path.read_text(encoding="utf-8")).get("organization")
    except json.JSONDecodeError:
        return None
    if not org:
        return None
    slug = preview_id.strip().lower().replace("/", "-").replace("_", "-")
    return f"https://{org}-preview-{slug}.docs.buildwithfern.com{_docs_site_path(fern_dir)}"


def _write_preview_url(*, fern_dir: Path, url: str) -> None:
    preview_url_path = fern_dir / PREVIEW_URL_FILE
    preview_url_path.write_text(f"{url}\n", encoding="utf-8")
    print(f"Fern docs preview URL: {url}", flush=True)


def _delete_remote_preview(*, fern_exe: str, fern_dir: Path, preview_id: str) -> None:
    print(f"Deleting Fern preview: {preview_id}", flush=True)
    result = _run(
        [fern_exe, "docs", "preview", "delete", "--id", preview_id],
        cwd=fern_dir,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"warning: fern docs preview delete --id {preview_id!r} exited "
            f"{result.returncode} (preview may already be gone).",
            file=sys.stderr,
        )


def _publish_remote_preview(
    *,
    fern_exe: str,
    fern_dir: Path,
    preview_id: str,
    force: bool,
) -> str | None:
    cmd = [fern_exe, "generate", "--docs", "--preview", "--id", preview_id]
    if force:
        cmd.append("--force")
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=fern_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)
    rc = proc.wait()
    output = "".join(output_chunks)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=output)

    url = _extract_preview_url(output) or _fallback_preview_url(
        fern_dir=fern_dir,
        preview_id=preview_id,
    )
    if url:
        _write_preview_url(fern_dir=fern_dir, url=url)
    else:
        print("warning: could not determine Fern docs preview URL", file=sys.stderr)
    return url


def _publish_production_docs(*, fern_exe: str, fern_dir: Path) -> None:
    cmd = [fern_exe, "generate", "--docs"]
    print("+ " + " ".join(cmd), flush=True)
    interactive = sys.stdin.isatty() and not os.environ.get("CI")
    proc = subprocess.Popen(
        cmd,
        cwd=fern_dir,
        stdin=sys.stdin if interactive else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)
    rc = proc.wait()
    output = "".join(output_chunks)
    if _FERN_PUBLISH_CANCELLED_RE.search(output):
        print("Production docs publish cancelled.", file=sys.stderr)
        raise subprocess.CalledProcessError(1, cmd, output=output)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=output)

    fern_url, custom_url = _production_docs_urls(fern_dir)
    print(f"Fern docs production URL: {fern_url}", flush=True)
    if custom_url:
        print(f"Fern docs custom domain: {custom_url}", flush=True)


def _clean_fern_token(raw: str | None) -> str | None:
    """Return a usable token, or None if empty/obviously invalid.

    A real Fern token (login JWT or org API token) is a single opaque string with no
    whitespace. Reject values like an accidentally-captured error message (e.g.
    ``export FERN_TOKEN=$(fern token)`` run outside a project yields
    ``Directory "fern" not found.``) so they do not force the org-auth path and an
    interactive login prompt.
    """
    if not raw:
        return None
    token = raw.strip()
    if not token or any(ch.isspace() for ch in token):
        return None
    return token


def _env_fern_token() -> str | None:
    return _clean_fern_token(os.environ.get("FERN_TOKEN"))


# Placeholder used wherever a FERN_TOKEN value would otherwise be printed. The token is
# a secret credential and must never be echoed to stdout/stderr or CI logs.
_REDACTED_TOKEN = "***redacted***"  # noqa: S105 - not a credential; masks tokens in logs


def _redacted_cmd(cmd: list[str]) -> str:
    """Render a command for display, masking any ``-eFERN_TOKEN=<value>`` argument."""
    rendered = []
    for part in cmd:
        if part.startswith("-eFERN_TOKEN="):
            rendered.append(f"-eFERN_TOKEN={_REDACTED_TOKEN}")
        else:
            rendered.append(part)
    return " ".join(rendered)


def _warn_if_invalid_env_fern_token() -> None:
    raw = os.environ.get("FERN_TOKEN")
    if raw and _clean_fern_token(raw) is None:
        print(
            f"warning: ignoring invalid FERN_TOKEN ({_REDACTED_TOKEN}); it is not a valid "
            "token (empty or contains whitespace). Run `unset FERN_TOKEN`, or set it to a "
            "non-expiring org token from `fern token`.",
            file=sys.stderr,
            flush=True,
        )


def _fern_auth_token() -> str | None:
    token = _env_fern_token()
    if token:
        return token
    token_path = Path.home() / ".fern" / "token"
    if token_path.is_file():
        return _clean_fern_token(token_path.read_text(encoding="utf-8"))
    return None


def _has_fern_org_token() -> bool:
    """Return whether an explicit org token is available for library generation."""
    return _env_fern_token() is not None


def _clean_generated_api_docs(generated_docs: Path) -> None:
    if generated_docs.is_dir():
        shutil.rmtree(generated_docs)
        try:
            removed = generated_docs.relative_to(REPO)
        except ValueError:
            removed = generated_docs
        print(f"removed: {removed}", flush=True)


def _ensure_emulation_api_placeholder(fern_dir: Path) -> None:
    """Keep unauthenticated local checks usable without generated library MDX.

    Fern's `folder:` entry in ``index.yml`` points inside the ``output.path``
    tree of the ``hololink-emulation`` library (see
    ``docs/user_guide/fern/docs.yml``) -- specifically at the deep subtree
    ``hololink/namespaces/emulation`` where Fern's C++ generator writes the
    per-compound MDX (see the comment on the ``folder:`` entry in
    ``docs/user_guide/fern/index.yml`` for why that path). Fern errors with
    "Folder not found" if that path does not exist on disk, which is exactly
    what happens on an unauthenticated local preview where
    ``fern docs md generate`` never ran (no token) and therefore never wrote
    anything under ``output.path``. This writes one hidden placeholder MDX
    file at that folder so the folder always resolves.
    """
    # Must mirror the `folder:` target in index.yml exactly; otherwise the
    # placeholder would land at a different folder than the one Fern probes.
    output = (
        fern_dir
        / "generated"
        / "api-reference"
        / "cpp"
        / "emulation"
        / "hololink"
        / "namespaces"
        / "emulation"
    )
    if any(output.rglob("*.mdx")) if output.is_dir() else False:
        return
    # Filename prefix `_` and `hidden: true` frontmatter both keep this out of
    # the rendered sidebar; belt-and-suspenders in case either exclusion
    # mechanism changes in a future Fern release.
    output.mkdir(parents=True, exist_ok=True)
    (output / "_placeholder.mdx").write_text(
        "---\ntitle: Emulation API\nhidden: true\n---\n\n"
        "This API reference is generated during authenticated Fern builds.\n",
        encoding="utf-8",
    )


def _has_generated_emulation_api(fern_dir: Path) -> bool:
    output = fern_dir / "generated" / "api-reference" / "cpp" / "emulation"
    if not output.is_dir():
        return False
    return any(
        path.is_file() and path.name not in {"overview.mdx", "_placeholder.mdx"}
        for path in output.rglob("*.mdx")
    )


def _fern_assets_ok(fern_dir: Path) -> bool:
    dist = fern_dir / "dist"
    return (dist / "output.css").is_file() and (dist / "output.js").is_file()


def run_pipeline(args: argparse.Namespace) -> int:
    fern_dir = args.fern_dir.expanduser().resolve()
    generated_docs = fern_dir / "generated"

    if not SOURCE.is_dir():
        print(f"Missing source dir: {SOURCE}", file=sys.stderr)
        return 1

    print("Hololink docs pipeline")
    print(f"  Repo: {REPO}")
    print(f"  Source: {SOURCE}")
    print(f"  Fern: {fern_dir}")

    if not args.skip_lfs_pull:
        if _ensure_lfs_assets():
            print("Git LFS: pulled docs/user_guide image assets.")
        elif shutil.which("git-lfs") is None:
            print("Git LFS: git-lfs not on PATH; skipping automatic pull.")

    rc = _check_source_assets(allow_lfs_pointers=args.allow_lfs_pointers)
    if rc:
        return rc

    if args.with_library_mdx:
        fern_exe = _fern_exe()
        if not fern_exe:
            print(
                "The `fern` CLI was not found on PATH; install it or omit --with-library-mdx.",
                file=sys.stderr,
            )
            return 127
        if not FIX_LIBRARY_MDX.is_file():
            print(f"Missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
            return 2
        _clean_generated_api_docs(generated_docs)
        _run([fern_exe, "docs", "md", "generate"], cwd=fern_dir)

    if generated_docs.is_dir():
        if not FIX_LIBRARY_MDX.is_file():
            print(f"Missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
            return 2
        _run(
            [
                sys.executable,
                str(FIX_LIBRARY_MDX),
                "--root",
                str(generated_docs),
            ]
        )

    if args.publish and not _has_generated_emulation_api(fern_dir):
        print(
            "error: production publication requires generated Emulation C++ API pages; "
            "run with --with-library-mdx.",
            file=sys.stderr,
        )
        return 1

    needs_remote_publish = args.publish_preview or args.delete_preview_id or args.publish
    if needs_remote_publish:
        if (rc := _require_fern_auth(
            action="Remote Fern preview" if not args.publish else "Production docs publish"
        )) is not None:
            return rc
        fern_exe = _fern_exe()
        if not fern_exe:
            print("The `fern` CLI was not found on PATH.", file=sys.stderr)
            return 127
        if args.delete_preview_id:
            _delete_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.delete_preview_id,
            )
        if args.publish_preview:
            if not args.preview_id:
                print("error: --publish-preview requires --preview-id.", file=sys.stderr)
                return 2
            if not _fern_assets_ok(fern_dir):
                print(
                    "error: missing dist theme assets (commit docs/user_guide/fern/dist/)",
                    file=sys.stderr,
                )
                return 1
            _publish_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.preview_id,
                force=args.force,
            )
        if args.publish:
            if not _fern_assets_ok(fern_dir):
                print(
                    "error: missing dist theme assets (commit docs/user_guide/fern/dist/)",
                    file=sys.stderr,
                )
                return 1
            try:
                _publish_production_docs(fern_exe=fern_exe, fern_dir=fern_dir)
            except subprocess.CalledProcessError:
                return 1
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if args.skip_fern_check and not args.preview:
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if not _fern_assets_ok(fern_dir):
        print(
            "error: missing dist theme assets (commit docs/user_guide/fern/dist/)",
            file=sys.stderr,
        )
        return 1

    fern_exe = _fern_exe()
    if not fern_exe:
        print("The `fern` CLI was not found on PATH.", file=sys.stderr)
        return 127

    _ensure_emulation_api_placeholder(fern_dir)

    if args.preview:
        return _run_preview(fern_exe, fern_dir, port=args.port)
    else:
        check_cmd = [fern_exe, "check", "--local"]
        if _fern_auth_token():
            # `--warnings` enables Fern checks that compare against the published site.
            # Without auth, the missing-redirects check only reports that it was skipped.
            check_cmd.append("--warnings")
        _run(check_cmd, cwd=fern_dir)

    return 0


def _docker_image_tag() -> str:
    version_path = REPO / "VERSION"
    version = (
        version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "local"
    )
    return f"hololink-docs:{version}"


def _mount_host_fern_dir(*, docker_home: Path, docker_mounts: list[str]) -> None:
    """Share host ``~/.fern`` with the container so login and tokens persist."""
    host_fern = Path.home() / ".fern"
    host_fern.mkdir(parents=True, exist_ok=True)
    docker_mounts.append(f"{host_fern}:{docker_home / '.fern'}")


def _build_docs_image() -> str:
    image = _docker_image_tag()
    _run(
        [
            "docker",
            "build",
            "--network=host",
            "-t",
            image,
            "-f",
            str(DOCKERFILE),
            str(DOCS_DIR),
        ]
    )
    return image


def _container_fern_dir(fern_dir: Path, repo_root: Path) -> tuple[Path, list[str]]:
    if _is_under(fern_dir, repo_root):
        return fern_dir, [f"{repo_root}:{repo_root}"]
    return EXTERNAL_FERN_DIR_MOUNT, [
        f"{repo_root}:{repo_root}",
        f"{fern_dir}:{EXTERNAL_FERN_DIR_MOUNT}",
    ]


def run_fern_login(args: argparse.Namespace) -> int:
    fern_exe = _fern_exe()
    if not fern_exe:
        print("The `fern` CLI was not found on PATH.", file=sys.stderr)
        return 127
    fern_dir = args.fern_dir.expanduser().resolve()
    _run([fern_exe, "login"], cwd=fern_dir)
    return 0


def _run_fern_login_in_docker(args: argparse.Namespace) -> int:
    if not DOCKERFILE.is_file():
        print(f"Missing Dockerfile: {DOCKERFILE}", file=sys.stderr)
        return 1

    if (rc := _ensure_docker_available()) is not None:
        return rc

    image = _build_docs_image()
    repo_root = REPO.resolve()
    fern_dir = args.fern_dir.expanduser().resolve()
    container_fern_dir, docker_mounts = _container_fern_dir(fern_dir, repo_root)

    docker_home = DOCS_DIR / ".fern-docker-home"
    docker_home.mkdir(parents=True, exist_ok=True)
    _mount_host_fern_dir(docker_home=docker_home, docker_mounts=docker_mounts)

    print(
        "Fern login in Docker uses device-code auth (browser OAuth callbacks on "
        "localhost are not reachable from the host).",
        flush=True,
    )
    print(
        "Open the URL shown below, authorize, then paste the code when prompted.",
        flush=True,
    )

    shell_cmd = f"cd {shlex.quote(str(container_fern_dir))} && fern login --device-code"
    docker_cmd = [
        "docker",
        "run",
        "-ti",
        f"-eHOME={docker_home}",
        "--rm",
        "--name",
        args.container_name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    for mount in docker_mounts:
        docker_cmd.extend(["-v", mount])
    docker_cmd.extend(
        [
            "-w",
            str(repo_root),
            image,
            "sh",
            "-c",
            shell_cmd,
        ]
    )

    print("+ " + " ".join(docker_cmd), flush=True)
    return subprocess.run(docker_cmd, cwd=REPO, check=False).returncode


def _ensure_docker_available() -> int | None:
    if not shutil.which("docker"):
        print(
            "error: docker not found on PATH; install Docker or pass --no-docker.",
            file=sys.stderr,
        )
        return 127
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        print(
            "error: Docker daemon is not running (cannot connect to docker.sock).",
            file=sys.stderr,
        )
        print(
            "Start your Docker runtime (e.g. `colima start`) or run without Docker:",
            file=sys.stderr,
        )
        print("  docs/make_docs.sh --no-docker", file=sys.stderr)
        return 1
    return None


def _run_in_docker(args: argparse.Namespace) -> int:
    if not DOCKERFILE.is_file():
        print(f"Missing Dockerfile: {DOCKERFILE}", file=sys.stderr)
        return 1

    if (rc := _ensure_docker_available()) is not None:
        return rc

    # Pull LFS on the host before starting the container. The bind-mounted workspace is
    # shared with Docker, but git lfs pull inside the container cannot use CI credentials.
    if not args.skip_lfs_pull:
        if _ensure_lfs_assets():
            print("Git LFS: pulled docs/user_guide image assets on host.")
        elif shutil.which("git-lfs") is None:
            print("Git LFS: git-lfs not on PATH; skipping automatic pull.", file=sys.stderr)

    image = _build_docs_image()

    repo_root = REPO.resolve()
    fern_dir = args.fern_dir.expanduser().resolve()
    container_fern_dir, docker_mounts = _container_fern_dir(fern_dir, repo_root)

    pipeline_args = [
        "python3",
        str(SCRIPTS / "build_hololink_docs.py"),
        "--no-docker",
        "--fern-dir",
        str(container_fern_dir),
    ]
    # LFS objects are pulled on the host above; skip in-container pull (no git credentials).
    pipeline_args.append("--skip-lfs-pull")
    if args.allow_lfs_pointers:
        pipeline_args.append("--allow-lfs-pointers")
    if args.with_library_mdx:
        pipeline_args.append("--with-library-mdx")
    if args.skip_library_mdx:
        pipeline_args.append("--skip-library-mdx")
    if args.skip_fern_check:
        pipeline_args.append("--skip-fern-check")
    if args.preview:
        pipeline_args.extend(["--preview", "--port", str(args.port)])
    if args.publish_preview:
        pipeline_args.append("--publish-preview")
    if args.publish:
        pipeline_args.append("--publish")
    if args.preview_id:
        pipeline_args.extend(["--preview-id", args.preview_id])
    if args.delete_preview_id:
        pipeline_args.extend(["--delete-preview-id", args.delete_preview_id])
    if args.force:
        pipeline_args.append("--force")

    pipeline_cmd = " ".join(shlex.quote(part) for part in pipeline_args)
    shell_cmd = (
        f"git config --global --add safe.directory {shlex.quote(str(REPO))} 2>/dev/null || true; "
        f"{pipeline_cmd}"
    )

    docker_home = DOCS_DIR / ".fern-docker-home"
    docker_home.mkdir(parents=True, exist_ok=True)
    _mount_host_fern_dir(docker_home=docker_home, docker_mounts=docker_mounts)

    docker_env = [f"-eHOME={docker_home}"]
    if os.environ.get("CI"):
        docker_env.append("-eCI=1")
    # Forward an explicitly-provided FERN_TOKEN (a non-expiring org token from
    # `fern token`) for org operations (library generation and docs publish). The cached
    # `fern login` credential is shared into the container via the mounted ~/.fern; it is
    # not injected as FERN_TOKEN because Fern reads FERN_TOKEN first and an OAuth login
    # access token is not a valid API token, which would break auth.
    env_token = _env_fern_token()
    if env_token:
        docker_env.append(f"-eFERN_TOKEN={env_token}")
    if args.preview:
        docker_env.append("-eHOST=0.0.0.0")

    docker_cmd = [
        "docker",
        "run",
        *docker_env,
        "--rm",
    ]
    if args.preview:
        # --net host does not publish ports to macOS/Colima; map Fern dev server ports
        # explicitly. Fern serves the frontend on --port and the backend on port + 1.
        frontend_port = args.port
        backend_port = args.port + 1
        docker_cmd.extend(
            [
                "-p",
                f"{frontend_port}:{frontend_port}",
                "-p",
                f"{backend_port}:{backend_port}",
            ]
        )
        print(
            f"Docs preview: open http://localhost:{frontend_port} in your browser "
            "(Docker port mapping; use --no-docker if the page does not load).",
            flush=True,
        )
    else:
        docker_cmd.extend(["--net", "host"])
    docker_cmd.extend(
        [
        "--name",
        args.container_name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        ]
    )
    for mount in docker_mounts:
        docker_cmd.extend(["-v", mount])
    docker_cmd.extend(
        [
            "-w",
            str(repo_root),
            image,
            "sh",
            "-c",
            shell_cmd,
        ]
    )

    if args.preview and sys.stdin.isatty():
        docker_cmd.insert(2, "-ti")
    elif args.publish and sys.stdin.isatty():
        docker_cmd.insert(2, "-ti")

    print("+ " + _redacted_cmd(docker_cmd), flush=True)
    return subprocess.run(docker_cmd, cwd=REPO, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate Holoscan Sensor Bridge Fern documentation."
    )
    parser.add_argument(
        "--fern-dir",
        type=Path,
        default=DEFAULT_FERN_DIR,
        help="Fern project directory (default: docs/user_guide/fern).",
    )
    parser.add_argument(
        "--allow-lfs-pointers",
        action="store_true",
        help="Do not fail when docs/user_guide images are unresolved Git LFS pointers.",
    )
    parser.add_argument(
        "--skip-lfs-pull",
        action="store_true",
        help="Do not run ``git lfs pull`` before validating image assets.",
    )
    parser.add_argument(
        "--with-library-mdx",
        action="store_true",
        help="Generate the Emulation C++ API MDX (requires Fern auth).",
    )
    parser.add_argument(
        "--skip-library-mdx",
        action="store_true",
        help="Skip C++ API generation even when a Fern org token is available.",
    )
    parser.add_argument(
        "--skip-fern-check",
        action="store_true",
        help="Run the pipeline only; skip fern check / preview.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run fern docs dev after the pipeline (live preview).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Preview server port (default: 3000; backend uses PORT+1).",
    )
    parser.add_argument(
        "--publish-preview",
        action="store_true",
        help="Publish a remote Fern docs preview (fern generate --docs --preview).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish docs to production (fern generate --docs).",
    )
    parser.add_argument(
        "--preview-id",
        help="Stable preview id for --publish-preview (typically a branch name).",
    )
    parser.add_argument(
        "--delete-preview-id",
        help="Delete a remote Fern preview with this id before publishing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip overwrite confirmation for remote preview publish (CI only; not used for --publish).",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Log in to Fern (device-code flow in Docker; browser flow on host).",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run on the host instead of the docs container.",
    )
    parser.add_argument(
        "--container-name",
        default="docs",
        help="Docker container name (default: docs).",
    )
    args = parser.parse_args()

    _warn_if_invalid_env_fern_token()

    if args.publish and args.preview:
        print("error: --publish and --preview are mutually exclusive.", file=sys.stderr)
        return 2
    if args.publish and args.publish_preview:
        print(
            "error: --publish and --publish-preview are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    if args.login:
        if args.no_docker:
            return run_fern_login(args)
        return _run_fern_login_in_docker(args)

    if args.skip_library_mdx:
        args.with_library_mdx = False
    elif not args.with_library_mdx:
        # CI preview and production jobs provide an org token. Generate there
        # automatically while keeping unauthenticated local checks non-interactive.
        args.with_library_mdx = _has_fern_org_token()

    if args.with_library_mdx and not _fern_auth_token():
        print(
            "error: C++ API generation requires Fern auth (run 'fern login' or set FERN_TOKEN).",
            file=sys.stderr,
        )
        return 1

    if args.no_docker:
        return run_pipeline(args)
    return _run_in_docker(args)


if __name__ == "__main__":
    raise SystemExit(main())
