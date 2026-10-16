#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "build_hololink_docs.py"
SPEC = importlib.util.spec_from_file_location("build_hololink_docs", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
BUILD_DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_DOCS)


class BuildHololinkDocsTestCase(TestCase):
    def test_library_inputs_use_the_current_checkout(self) -> None:
        docs_yml = (
            Path(__file__).resolve().parents[2] / "user_guide" / "fern" / "docs.yml"
        )
        content = docs_yml.read_text(encoding="utf-8")

        assert "path: ../../../src/hololink/emulation" in content
        assert "git:" not in content
        assert (docs_yml.parent / "../../../src/hololink/emulation").resolve().is_dir()

    def test_local_library_generation_requires_docker(self) -> None:
        with patch.object(BUILD_DOCS.shutil, "which", return_value=None):
            assert "requires Docker" in BUILD_DOCS._local_library_runtime_error()

    def test_local_library_generation_requires_running_daemon(self) -> None:
        completed_process = subprocess.CompletedProcess(
            ["docker", "info"],
            1,
            stdout="",
            stderr="Cannot connect to the Docker daemon\n",
        )

        with (
            patch.object(BUILD_DOCS.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(BUILD_DOCS.subprocess, "run", return_value=completed_process),
        ):
            assert "daemon is unavailable" in BUILD_DOCS._local_library_runtime_error()

    def test_container_generation_forwards_active_docker_socket(self) -> None:
        completed_process = subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout="991\n",
            stderr="",
        )
        with (
            patch.dict(
                BUILD_DOCS.os.environ,
                {"DOCKER_HOST": "unix:///tmp/colima/docker.sock"},
            ),
            patch.object(BUILD_DOCS.sys, "platform", "linux"),
            patch.object(
                BUILD_DOCS.subprocess,
                "run",
                return_value=completed_process,
            ) as run,
        ):
            options = BUILD_DOCS._container_docker_options(
                image="hololink-docs:test",
                with_library_mdx=True,
            )

        run.assert_called_once()
        assert "-eDOCKER_HOST=unix:///var/run/docker.sock" in options
        assert "991" in options
        assert "/tmp/colima/docker.sock:/var/run/docker.sock" in options

    def test_colima_uses_daemon_visible_docker_socket(self) -> None:
        context_process = subprocess.CompletedProcess(
            ["docker", "context", "show"],
            0,
            stdout="colima\n",
            stderr="",
        )
        with (
            patch.object(BUILD_DOCS.sys, "platform", "darwin"),
            patch.object(
                BUILD_DOCS.subprocess, "run", return_value=context_process
            ) as run,
        ):
            assert BUILD_DOCS._active_docker_socket() == "/var/run/docker.sock"

        run.assert_called_once_with(
            ["docker", "context", "show"],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_container_reuse_does_not_forward_docker_socket(self) -> None:
        with patch.object(BUILD_DOCS.subprocess, "run") as run:
            options = BUILD_DOCS._container_docker_options(
                image="hololink-docs:test",
                with_library_mdx=False,
            )

        run.assert_not_called()
        assert options == []

    def test_generates_library_docs_locally_without_fern_auth(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "docs" / "user_guide"
            fern_dir = source / "fern"
            scripts = root / "docs" / "scripts"
            source.mkdir(parents=True)
            fern_dir.mkdir()
            scripts.mkdir(parents=True)
            fix_library_mdx = scripts / "fix_generated_library_mdx.py"
            fix_library_mdx.touch()
            args = SimpleNamespace(
                fern_dir=fern_dir,
                skip_lfs_pull=True,
                allow_lfs_pointers=False,
                with_library_mdx=True,
                publish=False,
                publish_preview=False,
                delete_preview_id=None,
                skip_fern_check=True,
                preview=False,
            )

            with (
                patch.object(BUILD_DOCS, "SOURCE", source),
                patch.object(BUILD_DOCS, "FIX_LIBRARY_MDX", fix_library_mdx),
                patch.object(BUILD_DOCS, "_fern_exe", return_value="/usr/bin/fern"),
                patch.object(BUILD_DOCS, "_fern_auth_token", return_value=None),
                patch.object(
                    BUILD_DOCS, "_checkout_commit", return_value="abc123def456"
                ),
                patch.object(
                    BUILD_DOCS, "_local_library_runtime_error", return_value=None
                ),
                patch.object(
                    BUILD_DOCS,
                    "_run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as run,
            ):
                assert BUILD_DOCS.run_pipeline(args) == 0

            run.assert_called_once()
            command = run.call_args.args[0]
            keywords = run.call_args.kwargs
            assert command == ["/usr/bin/fern", "docs", "md", "generate", "--local"]
            assert keywords["cwd"] == fern_dir.resolve()
            assert keywords["check"] is False
            assert Path(keywords["env"]["TMPDIR"]).parent == source


if __name__ == "__main__":
    main()
