#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "build_hololink_docs.py"
SPEC = importlib.util.spec_from_file_location("build_hololink_docs", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
BUILD_DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_DOCS)


class BuildHololinkDocsTestCase(TestCase):
    def test_preview_fallback_matches_fern_5741_branch_normalization(self) -> None:
        with TemporaryDirectory() as temporary:
            fern = Path(temporary)
            (fern / "fern.config.json").write_text('{"organization": "nvidia"}')
            for preview_id, expected_slug in (
                ("jcfr/hsb-3.0-docs-archives", "jcfr-hsb-3-0-docs-archives"),
                ("---Mixed...Name___---", "mixed-name"),
                ("", "default"),
            ):
                with self.subTest(preview_id=preview_id):
                    self.assertEqual(
                        BUILD_DOCS._fallback_preview_url(
                            fern_dir=fern, preview_id=preview_id
                        ),
                        "https://nvidia-preview-"
                        + expected_slug
                        + ".docs.buildwithfern.com/holoscan/sensor-bridge",
                    )

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
                patch.object(BUILD_DOCS, "_api_input_state", return_value={}),
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

    def test_remote_preview_requires_archive_before_auth_or_publication(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fern = root / "fern"
            fern.mkdir()
            args = SimpleNamespace(
                fern_dir=fern,
                skip_lfs_pull=True,
                allow_lfs_pointers=False,
                with_library_mdx=False,
                publish=False,
                publish_preview=True,
                delete_preview_id="owned-preview",
                skip_fern_check=True,
                preview=False,
            )
            with (
                patch.object(BUILD_DOCS, "SOURCE", root),
                patch.object(
                    BUILD_DOCS, "_missing_generated_api_titles", return_value=[]
                ),
                patch.object(BUILD_DOCS, "_check_api_source_stamp"),
                patch.object(BUILD_DOCS, "_require_fern_auth") as auth,
                patch.object(BUILD_DOCS, "_publish_remote_preview") as publish,
                patch.object(BUILD_DOCS, "_delete_remote_preview") as delete,
            ):
                self.assertEqual(BUILD_DOCS.run_pipeline(args), 1)
            auth.assert_not_called()
            publish.assert_not_called()
            delete.assert_not_called()

    def test_remote_preview_requires_real_apis(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fern = root / "fern"
            fern.mkdir()
            args = SimpleNamespace(
                fern_dir=fern,
                skip_lfs_pull=True,
                allow_lfs_pointers=False,
                with_library_mdx=False,
                publish=False,
                publish_preview=True,
                delete_preview_id=None,
                skip_fern_check=True,
                preview=False,
                preview_id="owned-preview",
                archive_dir=root / "archive",
            )
            with (
                patch.object(BUILD_DOCS, "SOURCE", root),
                patch.object(BUILD_DOCS, "_require_fern_auth") as auth,
                patch.object(BUILD_DOCS, "_publish_remote_preview") as publish,
            ):
                self.assertEqual(BUILD_DOCS.run_pipeline(args), 1)
            auth.assert_not_called()
            publish.assert_not_called()

    def test_remote_validation_failure_prevents_preview_deletion(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fern = root / "fern"
            fern.mkdir()
            args = SimpleNamespace(
                fern_dir=fern,
                skip_lfs_pull=True,
                allow_lfs_pointers=False,
                with_library_mdx=False,
                publish=False,
                publish_preview=True,
                delete_preview_id="owned-preview",
                skip_fern_check=True,
                preview=False,
                preview_id="owned-preview",
                archive_dir=root / "archive",
            )
            archive_module = SimpleNamespace(assemble=lambda *unused: fern)
            with (
                patch.dict("sys.modules", {"archive_hololink_docs": archive_module}),
                patch.object(BUILD_DOCS, "SOURCE", root),
                patch.object(
                    BUILD_DOCS, "_missing_generated_api_titles", return_value=[]
                ),
                patch.object(BUILD_DOCS, "_check_api_source_stamp"),
                patch.object(BUILD_DOCS, "_require_fern_auth", return_value=None),
                patch.object(BUILD_DOCS, "_fern_exe", return_value="fern"),
                patch.object(BUILD_DOCS, "_fern_assets_ok", return_value=True),
                patch.object(BUILD_DOCS, "_check_api_source_stamp"),
                patch.object(
                    BUILD_DOCS,
                    "_run",
                    side_effect=subprocess.CalledProcessError(1, ["fern", "check"]),
                ) as run,
                patch.object(BUILD_DOCS, "_publish_remote_preview") as publish,
                patch.object(BUILD_DOCS, "_delete_remote_preview") as delete,
            ):
                self.assertEqual(BUILD_DOCS.run_pipeline(args), 1)
            self.assertEqual(
                run.call_args.args[0], ["fern", "check", "--local", "--warnings"]
            )
            publish.assert_not_called()
            delete.assert_not_called()

    def test_source_stamp_detects_dirty_configured_input_and_changed_output(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            repo = Path(temporary)
            fern = repo / "docs/user_guide/fern"
            fern.mkdir(parents=True)
            headers = repo / "src/api"
            headers.mkdir(parents=True)
            header = headers / "api.hpp"
            header.write_text("class ReleaseAPI {};\n")
            (fern / "docs.yml").write_text(
                "libraries:\n  public-api:\n    input:\n      path: ../../../src/api\n    config:\n      doxyfile: ./api.Doxyfile\n"
            )
            (fern / "fern.config.json").write_text('{"version":"5.74.1"}')
            (fern / "api.Doxyfile").write_text("RECURSIVE = YES\n")
            fix = repo / "postprocess.py"
            fix.write_text("# postprocess fixture\n")
            generated = fern / "generated"
            generated.mkdir()
            page = generated / "api.mdx"
            page.write_text("# ReleaseAPI\n")
            with (
                patch.object(BUILD_DOCS, "REPO", repo),
                patch.object(BUILD_DOCS, "FIX_LIBRARY_MDX", fix),
                patch.object(
                    BUILD_DOCS.subprocess, "check_output", return_value="1" * 40
                ),
            ):
                BUILD_DOCS._write_api_source_stamp(fern)
                BUILD_DOCS._check_api_source_stamp(fern)
                header.write_text("class NextReleaseAPI {};\n")
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    BUILD_DOCS._check_api_source_stamp(fern)
                header.write_text("class ReleaseAPI {};\n")
                page.write_text("# Modified generated content\n")
                with self.assertRaisesRegex(ValueError, "pages changed"):
                    BUILD_DOCS._check_api_source_stamp(fern)

    def test_default_docker_dispatch_does_not_import_archive_or_yaml(self) -> None:
        original_import = __import__

        def guarded_import(name, *arguments, **keywords):
            if name in {"yaml", "archive_hololink_docs"}:
                raise AssertionError("Default Docker dispatch imported " + name)
            return original_import(name, *arguments, **keywords)

        with (
            patch(
                "sys.argv",
                [
                    str(SCRIPT),
                    "--archive-ref",
                    "a" * 40,
                    "--archive-repository",
                    "/local/archive-repository",
                    "--skip-library-mdx",
                ],
            ),
            patch("builtins.__import__", side_effect=guarded_import),
            patch.object(BUILD_DOCS, "_run_in_docker", return_value=0) as docker,
            patch.object(BUILD_DOCS, "run_pipeline") as native,
        ):
            self.assertEqual(BUILD_DOCS.main(), 0)
        self.assertEqual(docker.call_args.args[0].archive_ref, "a" * 40)
        native.assert_not_called()

    def test_native_execution_extracts_archive_before_pipeline(self) -> None:
        args = SimpleNamespace(
            archive_ref="a" * 40, archive_repository="/local/archive-repository"
        )
        extracted = Path("/local/prepared-v2.7")
        extract = Mock(return_value=extracted)
        archive_module = SimpleNamespace(extract_archive=extract)
        with (
            patch.dict("sys.modules", {"archive_hololink_docs": archive_module}),
            patch.object(
                BUILD_DOCS, "_run_pipeline_current", return_value=0
            ) as pipeline,
        ):
            self.assertEqual(BUILD_DOCS.run_pipeline(args), 0)
        self.assertEqual(args.archive_dir, extracted)
        self.assertEqual(
            extract.call_args.args[:2], (args.archive_repository, args.archive_ref)
        )
        pipeline.assert_called_once_with(args)

    def test_container_receives_ref_and_readonly_local_repository(self) -> None:
        with TemporaryDirectory(prefix="archive repository ") as temporary:
            repository = Path(temporary).resolve()
            args = SimpleNamespace(
                archive_ref="b" * 40, archive_repository=str(repository)
            )
            flags, mounts = BUILD_DOCS._archive_container_options(
                args, Path("/other/checkout")
            )
        self.assertEqual(
            flags, ["--archive-ref", "b" * 40, "--archive-repository", str(repository)]
        )
        self.assertEqual(mounts, [f"{repository}:{repository}:ro"])

    def test_container_token_stays_in_environment_not_arguments_or_logs(self) -> None:
        fixture_token = "fixture-token-not-a-credential"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                fern_dir=root / "fern",
                skip_lfs_pull=True,
                with_library_mdx=False,
                archive_dir=None,
                archive_ref=None,
                archive_repository=None,
                output_dir=None,
                allow_lfs_pointers=False,
                skip_library_mdx=True,
                skip_fern_check=True,
                preview=False,
                publish_preview=False,
                publish=False,
                preview_id=None,
                delete_preview_id=None,
                force=False,
                container_name="docs-test",
            )
            with (
                patch.dict(
                    BUILD_DOCS.os.environ, {"FERN_TOKEN": "  " + fixture_token + "  "}
                ),
                patch.object(BUILD_DOCS, "REPO", root),
                patch.object(BUILD_DOCS, "DOCS_DIR", root),
                patch.object(BUILD_DOCS, "_ensure_docker_available", return_value=None),
                patch.object(BUILD_DOCS, "_build_docs_image", return_value="docs-test"),
                patch.object(BUILD_DOCS, "_container_docker_options", return_value=[]),
                patch.object(BUILD_DOCS, "_mount_host_fern_dir"),
                patch.object(
                    BUILD_DOCS.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run,
                patch("builtins.print") as output,
            ):
                self.assertEqual(BUILD_DOCS._run_in_docker(args), 0)
            command = run.call_args.args[0]
            self.assertIn("-eFERN_TOKEN", command)
            self.assertNotIn(fixture_token, " ".join(command))
            self.assertEqual(run.call_args.kwargs["env"]["FERN_TOKEN"], fixture_token)
            self.assertNotIn(fixture_token, str(output.call_args_list))


if __name__ == "__main__":
    main()
