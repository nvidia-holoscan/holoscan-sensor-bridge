#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare, check, and assemble immutable Sensor Bridge documentation snapshots.

Snapshots contain prepared MDX, their own navigation, assets, generated APIs and
provenance.json. Assembly never regenerates or changes an input snapshot. Git
archive inputs require an explicit full commit and repository; no private remote
is inferred for public checkouts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import yaml

SITE = "/holoscan/sensor-bridge"
PUBLIC_REPOSITORY = "https://github.com/nvidia-holoscan/holoscan-sensor-bridge"
MANIFEST = "provenance.json"
COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"v([0-9]+)\.([0-9]+)")
LINK = re.compile(
    r"\]\(\s*(?P<markdown><[^>]+>|[^\s)]+)"
    r'|\b(?:href|src)=["\'](?P<attribute>[^"\']+)["\']'
    r"|^\s*\[[^]\n]+\]:\s*(?P<reference><[^>]+>|[^\s]+)",
    re.MULTILINE,
)


CODEBLOCK_LINKS = re.compile(
    r"<CodeBlock\b[^>]*\blinks=\{\{(?P<links>.*?)\}\}", re.DOTALL
)
CODEBLOCK_TARGET = re.compile(r'"[^"\n]+"\s*:\s*"(?P<target>[^"\n]+)"')


def link_targets(text: str) -> list[tuple[int, int, str]]:
    targets = []
    for match in LINK.finditer(text):
        group = next(
            name
            for name in ("markdown", "attribute", "reference")
            if match.group(name) is not None
        )
        targets.append((*match.span(group), match.group(group)))
    for block in CODEBLOCK_LINKS.finditer(text):
        for match in CODEBLOCK_TARGET.finditer(block.group("links")):
            start, end = match.span("target")
            targets.append(
                (
                    block.start("links") + start,
                    block.start("links") + end,
                    match.group("target"),
                )
            )
    return sorted(targets)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def slug(title: str) -> str:
    title = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", title)
    title = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", title)
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def generated_route(path: str) -> str:
    # Fern 5.74.1 folder navigation and library generator normalize every segment
    # with this rule. Keep underscores out of URLs while retaining original files.
    return "/".join(
        re.sub(r"[^a-z0-9-]", "", re.sub(r"\s+", "-", part.lower()))
        for part in path.split("/")
    )


def version_slug(version: str) -> str:
    match = VERSION.fullmatch(version)
    if match is None:
        raise ValueError("Use a minor version such as v2.7.")
    return "-".join(match.groups())


def files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Snapshot contains a symbolic link: {path}")
        if path.is_file():
            if path.read_bytes().startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(f"Unresolved Git LFS pointer: {path}")
            result.append(path)
    return result


def inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files(root)
        if path.relative_to(root).as_posix() != MANIFEST
    }


def mask_code(text: str, *, inline: bool = True) -> str:
    """Mask code examples, retaining inline text only for heading-ID extraction."""
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"\s*(`{3,}|~{3,})", line)
        if match:
            if fence is None:
                fence = match.group(1)
            elif match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = None
            result.append(
                " " * len(line.rstrip("\n")) + ("\n" if line.endswith("\n") else "")
            )
        elif fence:
            result.append(
                " " * len(line.rstrip("\n")) + ("\n" if line.endswith("\n") else "")
            )
        else:
            result.append(
                re.sub(r"(`+).*?\1", lambda m: " " * len(m.group()), line)
                if inline
                else line
            )
    return "".join(result)


def navigation(root: Path, prefix: str) -> tuple[dict[Path, str], dict[str, str]]:
    """Resolve authored pages and folder mounts as Fern 5.74.1 does.

    Generated folder URL segments come from filenames, not C++ display titles.
    Explicit folder slugs must override the deep source folder name.
    """
    root = root.resolve()
    paths: dict[Path, str] = {}
    aliases: dict[str, str] = {}

    def add(path: Path, route: str) -> None:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping navigation page: {path}")
        if route in paths.values():
            raise ValueError(f"Duplicate navigation route: {route}")
        paths[path] = route

    def walk(nodes: list[dict], base: str) -> None:
        for node in nodes:
            if "section" in node:
                route = base + "/" + (node.get("slug") or slug(node["section"]))
                if node.get("path"):
                    add(root / node["path"], route)
                walk(node.get("contents", []), route)
            elif "page" in node:
                add(
                    root / node["path"],
                    base + "/" + (node.get("slug") or slug(node["page"])),
                )
            elif "folder" in node:
                folder = (root / node["folder"]).resolve()
                if not folder.is_relative_to(root.resolve()) or not folder.is_dir():
                    raise ValueError(f"Missing or escaping API folder: {folder}")
                pages = [
                    p
                    for p in folder.rglob("*.mdx")
                    if p.name not in {"_placeholder.mdx", "overview.mdx"}
                ]
                if not pages or any(
                    p.name == "_placeholder.mdx" for p in folder.rglob("*.mdx")
                ):
                    raise ValueError(
                        f"API folder requires real generated pages: {folder}"
                    )
                route = base + "/" + (node.get("slug") or folder.name.lower())
                for page in pages:
                    raw_suffix = (
                        page.relative_to(folder).with_suffix("").as_posix().lower()
                    )
                    # Fern 5.74.1 removes underscores from generated URL segments:
                    # ECB_CMD_CODE.mdx -> ecbcmdcode, ControlPlaneCallback_f -> controlplanecallbackf.
                    suffix = generated_route(raw_suffix)
                    add(page, route + "/" + suffix)
                    aliases[route + "/" + raw_suffix] = paths[page.resolve()]
                    # Library output links can retain C++ namespace paths even
                    # when navigation mounts a deeper directory with a new slug.
                    relative = page.relative_to(root).with_suffix("").as_posix().lower()
                    generated_alias = "/" + relative.removeprefix("generated/")
                    aliases[generated_alias] = paths[page.resolve()]
                    aliases[generated_route(generated_alias)] = paths[page.resolve()]
                    if "/hololink/" in relative:
                        raw = "/hololink/" + relative.split("/hololink/", 1)[1]
                        aliases[raw] = paths[page.resolve()]
                        aliases[generated_route(raw)] = paths[page.resolve()]
                        for parent in page.relative_to(folder).parents:
                            if parent != Path("."):
                                aliases[
                                    "/hololink/"
                                    + folder.relative_to(root)
                                    .as_posix()
                                    .split("/hololink/", 1)[1]
                                    + "/"
                                    + parent.as_posix().lower()
                                ] = (route + "/" + parent.as_posix().lower())
                aliases[route] = route
                relative = folder.relative_to(root).as_posix().lower()
                if "/hololink/" in relative:
                    aliases["/hololink/" + relative.split("/hololink/", 1)[1]] = route

    walk(load_yaml(root / "index.yml").get("navigation", []), prefix)
    if not paths:
        raise ValueError(f"No pages in {root / 'index.yml'}")
    return paths, aliases


def _heading_text(heading: str) -> str:
    """Keep inline-code text while removing markup from the surrounding prose."""

    def prose(text: str) -> str:
        text = re.sub(r"\s*\[#[^]]+\]", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)

    parts = []
    end = 0
    for span in re.finditer(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", heading):
        parts.append(prose(heading[end : span.start()]))
        parts.append(span.group(2))
        end = span.end()
    parts.append(prose(heading[end:]))
    # Fern normalizes heading whitespace before allocating duplicate IDs.
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def anchors(page: Path) -> set[str]:
    source = page.read_text(encoding="utf-8")
    # Explicit anchors ignore code examples; heading IDs include inline-code text.
    text = mask_code(source)
    result = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    result.update(re.findall(r"\[#([^]\n]+)\]", text))
    seen = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", mask_code(source, inline=False)):
        heading = _heading_text(heading)
        key = re.sub(r"[^a-z0-9_\s-]", "", heading.lower()).strip().replace(" ", "-")
        count = seen.get(key, 0)
        result.add(key + (f"-{count}" if count else ""))
        seen[key] = count + 1
    return result


def rewrite_links(root: Path, version: str, *, write: bool) -> list[str]:
    root = root.resolve()
    prefix = SITE + "/" + version_slug(version)
    paths, aliases = navigation(root, prefix)
    known = set(paths.values()) | set(aliases.values()) | {prefix}
    # Fern folder URLs redirect to their first child.
    for route in list(known):
        parent = route.rsplit("/", 1)[0]
        while parent.startswith(prefix + "/"):
            known.add(parent)
            parent = parent.rsplit("/", 1)[0]
    findings = []
    by_route = {route: page for page, route in paths.items()}
    page_anchors = {page: anchors(page) for page in paths}
    for page in root.rglob("*.mdx"):
        text = page.read_text(encoding="utf-8")
        updates = []
        for start, end, original in link_targets(mask_code(text)):
            target = original.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                if url.netloc != "docs.nvidia.com" or not (
                    url.path.rstrip("/") == SITE or url.path.startswith(SITE + "/")
                ):
                    continue
            route = unquote(url.path)
            if not route:
                if url.fragment and unquote(url.fragment) not in page_anchors.get(
                    page.resolve(), anchors(page)
                ):
                    findings.append(f"{page}: missing same-page anchor {target}")
                continue
            resolved = (page.parent / route).resolve()
            replacement = None
            if route.endswith((".mdx", ".md")):
                replacement = paths.get(resolved)
                if replacement is None:
                    findings.append(f"{page}: unresolved page-file link {target}")
                    continue
            elif route.rstrip("/") == SITE:
                replacement = prefix
            elif route.startswith(SITE + "/"):
                tail = route[len(SITE) + 1 :]
                explicit = re.match(
                    r"^([0-9]+)[.-]([0-9]+)(?:[.-][0-9]+)*(?:/|$)", tail
                )
                if explicit and tuple(map(int, explicit.groups())) != tuple(
                    map(int, VERSION.fullmatch(version).groups())
                ):
                    # An explicit link to another release is an intentional
                    # historical reference, not a latest-version page link.
                    continue
                tail = re.sub(r"^(?:2-7-latest|2-7)(?:/|$)", "", tail)
                replacement = aliases.get(
                    "/" + tail.lower(),
                    aliases.get(
                        prefix + "/" + tail.lower(),
                        prefix + ("/" + tail if tail else ""),
                    ),
                )
                if replacement.rstrip("/") not in known:
                    findings.append(f"{page}: unknown archive route {target}")
                    continue
            elif route.startswith(("/hololink/", "/api-reference/")):
                replacement = aliases.get(route.lower())
                if replacement is None:
                    findings.append(
                        f"{page}: generated namespace link has no mounted route: {target}"
                    )
                    continue
            elif route.startswith("/"):
                findings.append(f"{page}: unrecognized absolute local link {target}")
                continue
            elif resolved in paths:
                replacement = paths[resolved]
            elif not resolved.suffix and resolved.with_suffix(".mdx") in paths:
                replacement = paths[resolved.with_suffix(".mdx")]
            elif page.resolve().is_relative_to(root / "generated"):
                current_route = paths.get(page.resolve())
                candidate = generated_route(
                    posixpath.normpath(
                        posixpath.join(
                            (
                                current_route.rsplit("/", 1)[0]
                                if current_route
                                else prefix
                            ),
                            route,
                        )
                    )
                )
                if candidate in known:
                    replacement = candidate
                elif not resolved.is_relative_to(root) or not resolved.is_file():
                    findings.append(
                        f"{page}: missing or unmounted generated link {target}"
                    )
                    continue
            elif not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                findings.append(f"{page}: missing or escaping asset/link {target}")
                continue
            if replacement is not None:
                destination = by_route.get(replacement.rstrip("/"))
                if (
                    destination
                    and url.fragment
                    and unquote(url.fragment) not in page_anchors[destination]
                ):
                    findings.append(f"{page}: missing destination anchor {target}")
                    continue
                rewritten = urlunsplit(("", "", replacement, url.query, url.fragment))
                if rewritten != target:
                    if not write:
                        findings.append(
                            f"{page}: link needs normalization: {target} -> {rewritten}"
                        )
                    updates.append((start, end, rewritten))
        if write:
            for start, end, replacement in reversed(updates):
                text = text[:start] + replacement + text[end:]
            page.write_text(text, encoding="utf-8")
    return findings


def check_rendering(root: Path, data: dict) -> None:
    """Verify a derived rendering against the separately retained source manifest."""
    rendering = data.get("rendering")
    if rendering is None:
        return
    expected = "../archive-source-" + data["version"] + ".json"
    if not isinstance(rendering, dict) or rendering.get("source_manifest") != expected:
        raise ValueError("Derived archive has an invalid source manifest location.")
    raw = (root / expected).read_bytes()
    if hashlib.sha256(raw).hexdigest() != rendering.get("source_manifest_sha256"):
        raise ValueError("Derived archive source manifest hash differs.")
    source = json.loads(raw)
    if rendering.get("transformation") not in {
        "escaped-angle-entities-outside-code",
        "fern-image-scanner-workarounds-outside-code",
    }:
        raise ValueError("Derived archive has an unsupported rendering transformation.")
    source_metadata = {key: value for key, value in source.items() if key != "files"}
    derived_metadata = {
        key: value for key, value in data.items() if key not in {"files", "rendering"}
    }
    if source_metadata != derived_metadata:
        raise ValueError("Derived archive changed its source provenance.")
    original, derived = source.get("files", {}), data["files"]
    if set(original) != set(derived):
        raise ValueError(
            "Derived archive file inventory differs from its source inventory."
        )
    changed = {name for name in original if original[name] != derived[name]}
    records = rendering.get("files", {})
    if set(records) != changed:
        raise ValueError(
            "Derived archive rendering records do not match changed files."
        )
    for name in changed:
        if not name.endswith(".mdx") or records[name] != {
            "before_sha256": original[name],
            "after_sha256": derived[name],
        }:
            raise ValueError(f"Derived archive rendering hashes differ for {name}.")


def scanner_safe_images(text: str) -> str:
    """Use HTML for simple Markdown images skipped after a code '<' by Fern.

    Fern 5.74.1's image scanner does not recognize code boundaries: a heredoc
    operator can start an HTML scan that consumes later Markdown images. Its
    attribute scanner still resolves HTML image sources in that span. Change
    only endangered images outside code, retaining code examples byte for byte.
    """
    visible = mask_code(text)
    spans = [
        match.span()
        for match in re.finditer(r"<[^>]*(?:>|$)", text)
        if visible[match.start()] != "<"
    ]
    updates = []
    for match in re.finditer(r"!\[(?P<alt>[^\]\n]*)\]\((?P<src>[^\s()<>]+)\)", visible):
        if any(start < match.start() < end for start, end in spans):
            alt = html.escape(text[slice(*match.span("alt"))], quote=True)
            src = html.escape(text[slice(*match.span("src"))], quote=True)
            updates.append(
                (match.start(), match.end(), f'<img alt="{alt}" src="{src}" />')
            )
    for start, end, replacement in reversed(updates):
        text = text[:start] + replacement + text[end:]
    return text


def render_archive(root: Path) -> None:
    """Work around Fern 5.74.1's image scanner in a staged copy.

    The scanner can treat a Markdown-escaped less-than sign as an HTML tag and
    skip later image substitutions. Entities have the same visible text. Keep
    code unchanged; use HTML image attributes when an angle inside code hides a
    later Markdown image. Retain the immutable manifest outside the copy.
    """
    root = root.resolve()
    data = check_snapshot(root)
    if data.get("rendering") is not None:
        raise ValueError(
            "Assemble the original archive rather than a derived rendering."
        )
    source_manifest = (root / MANIFEST).read_bytes()
    records = {}
    for page in sorted(root.rglob("*.mdx")):
        before = page.read_bytes()
        text = before.decode("utf-8")
        updates = []
        for match in re.finditer(r"(?P<slashes>\\+)(?P<angle>[<>])", mask_code(text)):
            if len(match.group("slashes")) % 2:
                updates.append(
                    (
                        match.end() - 2,
                        match.end(),
                        "&lt;" if match.group("angle") == "<" else "&gt;",
                    )
                )
        for start, end, replacement in reversed(updates):
            text = text[:start] + replacement + text[end:]
        text = scanner_safe_images(text)
        after = text.encode("utf-8")
        if after == before:
            continue
        page.write_bytes(after)
        records[page.relative_to(root).as_posix()] = {
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest(),
        }
    if not records:
        return
    source_path = root.parent / ("archive-source-" + data["version"] + ".json")
    if source_path.exists():
        raise ValueError(
            f"Derived archive source manifest already exists: {source_path}"
        )
    source_path.write_bytes(source_manifest)
    data["rendering"] = {
        "transformation": "fern-image-scanner-workarounds-outside-code",
        "source_manifest": "../" + source_path.name,
        "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "files": records,
    }
    data["files"] = inventory(root)
    (root / MANIFEST).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    check_snapshot(root)


def check_snapshot(root: Path, version: str = "v2.7") -> dict:
    root = root.resolve()
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("version") != version:
        raise ValueError("Archive manifest schema or version does not match.")
    provenance = data.get("provenance", {})
    if provenance.get("kind") not in {"source-build", "publication-capture"}:
        raise ValueError(
            "Archive requires source-build or publication-capture provenance."
        )
    if not provenance.get("evidence"):
        raise ValueError("Archive requires publication provenance evidence.")
    if provenance.get("fern_version") != "5.74.1":
        raise ValueError("Archive must record the supported Fern version 5.74.1.")
    if provenance["kind"] == "source-build":
        for key in ("documentation_commit", "api_commit", "tooling_commit"):
            if not COMMIT.fullmatch(provenance.get(key, "")):
                raise ValueError(f"Archive requires a resolved {key}.")
        if not provenance.get("public_source_ref"):
            raise ValueError("Source-built archive requires a public source ref.")
    elif not provenance.get("captured_at") or not provenance.get("source_urls"):
        raise ValueError(
            "A publication capture requires its time and source URL inventory."
        )
    actual = inventory(root)
    if actual != data.get("files"):
        raise ValueError("Archive file inventory differs from its immutable manifest.")
    check_rendering(root, data)
    for page in root.rglob("*.mdx"):
        if "gitlab-master.nvidia.com" in page.read_text(encoding="utf-8"):
            raise ValueError(
                f"Archive page contains an internal GitLab reference: {page}"
            )
    if any(p.name == "_placeholder.mdx" for p in root.rglob("*.mdx")):
        raise ValueError("Archive contains placeholder API pages.")
    pages, _ = navigation(root, SITE + "/" + version_slug(version))
    emulator = (
        root / "generated/api-reference/cpp/emulation/hololink/namespaces/emulation"
    )
    if not any(
        page.is_relative_to(emulator) and "/api-reference/emulation-api/" in route
        for page, route in pages.items()
    ):
        raise ValueError(
            "Archive requires the mounted 2.7 emulator API and real generated pages."
        )
    findings = rewrite_links(root, version, write=False)
    if findings:
        raise ValueError("\n".join(findings))
    return data


def prepare_snapshot(
    source: Path, output: Path, provenance: dict, version: str = "v2.7"
) -> None:
    """Freeze a prepared user-guide tree; provenance is supplied and reviewed separately."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a new snapshot destination outside the source tree.")
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in {"fern", "README.md"} or item.name.startswith("."):
            continue
        if item.is_symlink():
            raise ValueError(f"Snapshot source contains a symbolic link: {item}")
        if item.is_dir():
            shutil.copytree(item, output / item.name, symlinks=True)
        else:
            shutil.copy2(item, output / item.name)
    fern = source / "fern"
    shutil.copytree(fern / "generated", output / "generated", symlinks=True)
    files(output)
    nav = load_yaml(fern / "index.yml")

    def flatten(value):
        if isinstance(value, dict):
            return {
                key: (
                    "./" + item[3:]
                    if key in {"path", "folder"}
                    and isinstance(item, str)
                    and item.startswith("../")
                    else flatten(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [flatten(item) for item in value]
        return value

    (output / "index.yml").write_text(yaml.safe_dump(flatten(nav), sort_keys=False))
    public_ref = provenance.get("public_source_ref")
    if public_ref:
        for page in output.rglob("*.mdx"):
            text = page.read_text(encoding="utf-8")
            text = re.sub(
                re.escape(PUBLIC_REPOSITORY)
                + r"/(blob|tree)/(?:main|internal|internal-3x)/",
                lambda match: PUBLIC_REPOSITORY
                + "/"
                + match.group(1)
                + "/"
                + public_ref
                + "/",
                text,
            )
            page.write_text(text, encoding="utf-8")
    if public_ref:
        repository = source.parent.parent.resolve()
        for page in output.rglob("*.mdx"):
            relative = page.relative_to(output)
            source_page = source / (
                Path("fern") / relative
                if relative.parts[0] == "generated"
                else relative
            )
            text = page.read_text(encoding="utf-8")
            updates = []
            for start, end, original in link_targets(mask_code(text)):
                target = original.strip("<>")
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                    continue
                local = (page.parent / unquote(url.path)).resolve()
                original = (source_page.parent / unquote(url.path)).resolve()
                if (
                    (not local.is_relative_to(output.resolve()) or not local.exists())
                    and original.is_relative_to(repository)
                    and original.exists()
                ):
                    kind = "tree" if original.is_dir() else "blob"
                    replacement = (
                        PUBLIC_REPOSITORY
                        + "/"
                        + kind
                        + "/"
                        + public_ref
                        + "/"
                        + original.relative_to(repository).as_posix()
                    )
                    updates.append(
                        (
                            start,
                            end,
                            urlunsplit(
                                (
                                    "https",
                                    "github.com",
                                    urlsplit(replacement).path,
                                    url.query,
                                    url.fragment,
                                )
                            ),
                        )
                    )
            for start, end, replacement in reversed(updates):
                text = text[:start] + replacement + text[end:]
            page.write_text(text, encoding="utf-8")
    findings = rewrite_links(output, version, write=True)
    if findings:
        raise ValueError("\n".join(findings))
    data = {
        "schema_version": 1,
        "version": version,
        "provenance": provenance,
        "files": inventory(output),
    }
    (output / MANIFEST).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    check_snapshot(output, version)


def extract_archive(repository: str, commit: str, output: Path) -> Path:
    if not COMMIT.fullmatch(commit):
        raise ValueError("--archive-ref requires a full 40-character commit SHA.")
    if output.exists():
        raise ValueError("Archive extraction requires a new directory.")
    with tempfile.TemporaryDirectory(prefix="hsb-archive-git-") as temporary:
        checkout = Path(repository).expanduser()
        if not checkout.is_dir():
            checkout = Path(temporary) / "checkout"
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    repository,
                    commit,
                ],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", commit + "^{commit}"], text=True
        ).strip()
        if actual != commit:
            raise ValueError("Archive commit does not match the requested ref.")
        content = subprocess.check_output(
            ["git", "-C", str(checkout), "archive", commit + ":v2.7"]
        )
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            members = archive.getmembers()
            if any(
                not (m.isfile() or m.isdir())
                or Path(m.name).is_absolute()
                or ".." in Path(m.name).parts
                for m in members
            ):
                raise ValueError("Archive contains an unsafe member.")
            output.mkdir(parents=True)
            if hasattr(tarfile, "data_filter"):
                archive.extractall(output, members=members, filter="data")
            else:
                archive.extractall(output, members=members)
    check_snapshot(output)
    return output


def assemble(source_guide: Path, archive: Path, output: Path) -> Path:
    source_guide, archive, output = (
        source_guide.resolve(),
        archive.resolve(),
        output.resolve(),
    )
    if output.exists() or output.resolve().is_relative_to(source_guide.resolve()):
        raise ValueError("Choose a new assembly output outside the source user guide.")
    if output.is_relative_to(archive):
        raise ValueError("Assembly output must be outside the immutable archive.")
    if (archive / "v2.7").is_dir():
        archive = archive / "v2.7"
    before = check_snapshot(archive)
    if before.get("rendering") is not None:
        raise ValueError(
            "Assemble the original archive rather than a derived rendering."
        )
    shutil.copytree(
        source_guide,
        output / "user_guide",
        symlinks=True,
        ignore=shutil.ignore_patterns(".fern*", "node_modules"),
    )
    files(output / "user_guide")
    fern = output / "user_guide" / "fern"
    if (fern / "v2.7").exists():
        raise ValueError("Current user guide already contains a v2.7 archive.")
    shutil.copytree(archive, fern / "v2.7")
    check_snapshot(fern / "v2.7")
    render_archive(fern / "v2.7")
    settings = load_yaml(fern / "docs.yml")
    settings.pop(
        "libraries", None
    )  # APIs have already been generated from their own checkouts.
    versions = settings["products"][0]["versions"]
    versions[:] = [
        {
            "display-name": "3.0 (Latest)",
            "slug": "3-0",
            "path": "./index.yml",
            "availability": "ga",
        },
        {
            "display-name": "2.7",
            "slug": "2-7",
            "path": "./v2.7/index.yml",
        },
        *[v for v in versions if v.get("display-name") == "Archives"],
    ]
    redirects = settings.setdefault("redirects", [])
    redirects.insert(
        0, {"source": SITE + "/2-7-latest/:slug*", "destination": SITE + "/2-7/:slug*"}
    )
    redirects.insert(0, {"source": SITE + "/2-7-latest", "destination": SITE + "/2-7"})
    (fern / "docs.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    legacy_index = fern / "versions" / "archives.mdx"
    if legacy_index.is_file():
        legacy_index.write_text(
            legacy_index.read_text(encoding="utf-8").replace(
                "https://archive.docs.nvidia.com/holoscan/sensor-bridge/2.7.0/",
                SITE + "/2-7/getting-started/introduction",
            ),
            encoding="utf-8",
        )
    if check_snapshot(archive) != before:
        raise ValueError("Input archive changed during assembly.")
    check_snapshot(fern / "v2.7")
    files(output / "user_guide")
    return fern


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--archive-dir", type=Path, required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--source-guide", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            check_snapshot(args.archive_dir.resolve())
        else:
            prepare_snapshot(
                args.source_guide.resolve(),
                args.output.resolve(),
                json.loads(args.provenance.read_text()),
            )
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
