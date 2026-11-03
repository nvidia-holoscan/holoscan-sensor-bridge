#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise immutable archive integrity and version isolation with disposable trees."""

import hashlib
import html
import importlib.util
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "archive_hololink_docs.py"
SPEC = importlib.util.spec_from_file_location("archive_hololink_docs", SCRIPT)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class ArchiveDocsTestCase(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guide = self.root / "repo" / "docs" / "user_guide"
        self.fern = self.guide / "fern"
        self.fern.mkdir(parents=True)
        self.generated = (
            self.fern
            / "generated/api-reference/cpp/emulation/hololink/namespaces/emulation"
        )
        (self.generated / "classes").mkdir(parents=True)
        (self.generated / "classes/hsbemulator.mdx").write_text(
            "# HSBEmulator\n\n[Memory](/hololink/namespaces/emulation/classes/addressmemory)\n"
        )
        (self.generated / "classes/addressmemory.mdx").write_text("# AddressMemory\n")
        (self.guide / "intro.mdx").write_text(
            "# Introduction\n\n[Setup](setup.mdx#install)\n"
            "[Versioned](/holoscan/sensor-bridge/2-7-latest/getting-started/setup)\n"
            '<Card href="/holoscan/sensor-bridge/getting-started/setup">Setup</Card>\n'
            "[setup]: /holoscan/sensor-bridge/getting-started/setup\n"
            "![Image](image.png)\n"
            "```python\nexample = '[Do not change](setup.mdx)'\n```\n"
        )
        (self.guide / "setup.mdx").write_text("# Setup\n\n## Install\n")
        (self.guide / "image.png").write_bytes(b"PNG fixture")
        (self.fern / "index.yml").write_text(
            "navigation:\n"
            "  - section: Getting Started\n"
            "    contents:\n"
            "      - page: Introduction\n"
            "        path: ../intro.mdx\n"
            "      - page: Setup\n"
            "        path: ../setup.mdx\n"
            "  - section: API reference\n"
            "    contents:\n"
            "      - folder: ./generated/api-reference/cpp/emulation/hololink/namespaces/emulation\n"
            "        slug: emulation-api\n"
            "        title: Emulation API\n"
        )
        (self.fern / "docs.yml").write_text(
            "libraries:\n  emulator:\n    input:\n      path: ../../../src\n"
            "products:\n  - display-name: Holoscan Sensor Bridge\n    slug: /\n"
            "    path: index.yml\n    versions:\n"
            "      - display-name: 3.0 (Latest)\n        path: ./index.yml\n"
            "      - display-name: Archives\n        path: ./versions/archives.yml\n"
            "redirects: []\n"
        )
        self.provenance = {
            "kind": "source-build",
            "documentation_commit": "1" * 40,
            "api_commit": "1" * 40,
            "tooling_commit": "2" * 40,
            "public_source_ref": "2.7.0",
            "fern_version": "5.74.1",
            "evidence": ["https://example.test/reviewed-publication-comparison"],
        }
        self.snapshot = self.root / "archives" / "v2.7"

    def freeze(self):
        ARCHIVE.prepare_snapshot(self.guide, self.snapshot, self.provenance)

    def test_snapshot_scopes_pages_namespace_links_and_attributes(self):
        self.freeze()
        content = (self.snapshot / "intro.mdx").read_text()
        self.assertIn("/2-7/getting-started/setup#install", content)
        self.assertNotIn("/2-7-latest/", content)
        self.assertIn(
            'href="/holoscan/sensor-bridge/2-7/getting-started/setup"', content
        )
        self.assertIn(
            "[setup]: /holoscan/sensor-bridge/2-7/getting-started/setup", content
        )
        self.assertIn("'[Do not change](setup.mdx)'", content)
        generated = (
            self.snapshot
            / "generated/api-reference/cpp/emulation/hololink/namespaces/emulation/classes/hsbemulator.mdx"
        )
        self.assertIn(
            "/2-7/api-reference/emulation-api/classes/addressmemory",
            generated.read_text(),
        )
        ARCHIVE.check_snapshot(self.snapshot)

    def test_assembly_retains_archive_hashes_and_own_navigation(self):
        self.freeze()
        before = ARCHIVE.inventory(self.snapshot)
        fern = ARCHIVE.assemble(self.guide, self.snapshot.parent, self.root / "output")
        versions = ARCHIVE.load_yaml(fern / "docs.yml")["products"][0]["versions"]
        self.assertEqual([v.get("slug") for v in versions[:2]], ["3-0", "2-7"])
        self.assertEqual(versions[1]["path"], "./v2.7/index.yml")
        self.assertEqual(versions[0]["display-name"], "3.0 (Latest)")
        self.assertEqual(versions[0]["availability"], "ga")
        self.assertEqual(versions[1]["display-name"], "2.7")
        self.assertNotIn("availability", versions[1])
        self.assertNotIn("libraries", ARCHIVE.load_yaml(fern / "docs.yml"))
        self.assertEqual(before, ARCHIVE.inventory(self.snapshot))
        self.assertEqual(before, ARCHIVE.inventory(fern / "v2.7"))
        self.assertIn(
            "[Setup](setup.mdx#install)", (fern.parent / "intro.mdx").read_text()
        )

    def test_tampered_archive_is_rejected_before_assembly(self):
        self.freeze()
        (self.snapshot / "setup.mdx").write_text("Changed")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            ARCHIVE.assemble(self.guide, self.snapshot, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_placeholder_api_is_rejected(self):
        (self.generated / "_placeholder.mdx").write_text("# Hidden placeholder\n")
        with self.assertRaisesRegex(ValueError, "requires real generated pages"):
            self.freeze()

    def test_missing_image_is_rejected(self):
        (self.guide / "image.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing or escaping"):
            self.freeze()

    def test_namespace_link_to_missing_symbol_is_rejected(self):
        (self.generated / "classes/hsbemulator.mdx").write_text(
            "[Missing](/hololink/namespaces/emulation/classes/missing)"
        )
        with self.assertRaisesRegex(ValueError, "no mounted route"):
            self.freeze()

    def test_no_unknown_absolute_page_is_silently_versioned(self):
        (self.guide / "intro.mdx").write_text(
            "[Missing](/holoscan/sensor-bridge/not-a-page)"
        )
        with self.assertRaisesRegex(ValueError, "unknown archive route"):
            self.freeze()

    def test_lfs_pointers_and_symlinks_are_rejected(self):
        image = self.guide / "image.png"
        image.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            self.freeze()
        image.unlink()
        image.symlink_to(self.guide / "setup.mdx")
        self.snapshot = self.root / "second-archives" / "v2.7"
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.freeze()

    def test_original_generation_unknown_requires_explicit_capture_provenance(self):
        self.provenance.pop("api_commit")
        with self.assertRaisesRegex(ValueError, "api_commit"):
            self.freeze()

    def test_capture_preserves_unknown_original_generation(self):
        self.provenance = {
            "kind": "publication-capture",
            "fern_version": "5.74.1",
            "captured_at": "2026-11-03T07:00:00Z",
            "source_urls": {
                "intro.mdx": {
                    "url": "https://example.test/intro.md",
                    "sha256": "a" * 64,
                }
            },
            "original_generator_revision": None,
            "evidence": ["https://example.test/approved-route-inventory"],
        }
        self.freeze()
        self.assertIsNone(
            ARCHIVE.check_snapshot(self.snapshot)["provenance"][
                "original_generator_revision"
            ]
        )

    def test_source_links_are_pinned_to_public_release(self):
        (self.guide / "intro.mdx").write_text(
            "[Header](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/src/a.hpp)\n"
            "[Repository](../../README.md)\n"
        )
        (self.guide.parent.parent / "README.md").write_text("Public README")
        self.freeze()
        text = (self.snapshot / "intro.mdx").read_text()
        self.assertIn("/blob/2.7.0/src/a.hpp", text)
        self.assertIn("/blob/2.7.0/README.md", text)

    def test_public_local_ref_extracts_without_network_or_private_default(self):
        self.freeze()
        subprocess.run(
            ["git", "init", "--quiet", str(self.snapshot.parent)], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.snapshot.parent), "add", "v2.7"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.snapshot.parent),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "Archive fixture",
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(self.snapshot.parent), "rev-parse", "HEAD"], text=True
        ).strip()
        extracted = ARCHIVE.extract_archive(
            str(self.snapshot.parent), commit, self.root / "public-snapshot"
        )
        self.assertEqual(ARCHIVE.inventory(self.snapshot), ARCHIVE.inventory(extracted))
        with patch.object(ARCHIVE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "40-character"):
                ARCHIVE.extract_archive(
                    "https://example.test/public.git",
                    "doc_archives",
                    self.root / "invalid",
                )
        run.assert_not_called()

    def test_authored_only_archive_cannot_omit_emulator_api(self):
        nav = (self.fern / "index.yml").read_text()
        (self.fern / "index.yml").write_text(nav.split("  - section: API reference")[0])
        import shutil

        shutil.rmtree(self.fern / "generated")
        (self.fern / "generated").mkdir()
        with self.assertRaisesRegex(ValueError, "mounted 2.7 emulator API"):
            self.freeze()

    def test_assembly_rejects_current_source_symlinks(self):
        self.freeze()
        outside = self.root / "outside.txt"
        outside.write_text("Outside content")
        (self.guide / "leaked.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            ARCHIVE.assemble(self.guide, self.snapshot, self.root / "output")
        self.assertTrue((self.root / "output/user_guide/leaked.txt").is_symlink())

    def test_assembly_cannot_write_inside_its_input_archive(self):
        self.freeze()
        before = ARCHIVE.inventory(self.snapshot)
        with self.assertRaisesRegex(ValueError, "outside the immutable archive"):
            ARCHIVE.assemble(self.guide, self.snapshot, self.snapshot / "output")
        self.assertEqual(before, ARCHIVE.inventory(self.snapshot))

    def test_generated_extensionless_and_codeblock_links_use_mounted_routes(self):
        (self.generated / "classes/hsbemulator.mdx").write_text(
            "# HSBEmulator\n\n[Memory](addressmemory)\n"
            '<CodeBlock links={{"Memory": "addressmemory"}}>\n'
            "```cpp\nAddressMemory *memory;\n```\n</CodeBlock>\n"
        )
        self.freeze()
        page = (
            self.snapshot
            / "generated/api-reference/cpp/emulation/hololink/namespaces/emulation/classes/hsbemulator.mdx"
        )
        self.assertEqual(
            page.read_text().count(
                "/2-7/api-reference/emulation-api/classes/addressmemory"
            ),
            2,
        )

    def test_missing_anchor_is_rejected(self):
        (self.guide / "intro.mdx").write_text("[Missing](setup.mdx#does-not-exist)")
        with self.assertRaisesRegex(ValueError, "missing destination anchor"):
            self.freeze()

    def test_generated_route_slugs_remove_underscores_as_fern_5741_does(self):
        for directory, name in (
            ("enums", "ECB_CMD_CODE"),
            ("typedefs", "ControlPlaneCallback_f"),
            ("functions", "DLTensorNBytes"),
        ):
            (self.generated / directory).mkdir(exist_ok=True)
            (self.generated / directory / (name + ".mdx")).write_text(
                "# " + name + "\n"
            )
        self.freeze()
        paths, aliases = ARCHIVE.navigation(self.snapshot, ARCHIVE.SITE + "/2-7")
        routes = set(paths.values())
        for suffix in (
            "enums/ecbcmdcode",
            "typedefs/controlplanecallbackf",
            "functions/dltensornbytes",
        ):
            self.assertIn(
                ARCHIVE.SITE + "/2-7/api-reference/emulation-api/" + suffix, routes
            )

    def test_generated_relative_urls_resolve_case_independently_of_filesystem(self):
        (self.generated / "classes/DataPlane.mdx").write_text("# DataPlane\n")
        (self.generated / "classes/hsbemulator.mdx").write_text(
            "# HSBEmulator\n[Data](dataplane)\n"
        )
        self.freeze()
        page = (
            self.snapshot
            / "generated/api-reference/cpp/emulation/hololink/namespaces/emulation/classes/hsbemulator.mdx"
        )
        self.assertIn(
            "/2-7/api-reference/emulation-api/classes/dataplane", page.read_text()
        )

    def test_archive_home_links_do_not_escape_to_latest(self):
        (self.guide / "intro.mdx").write_text(
            "[Home](https://docs.nvidia.com/holoscan/sensor-bridge)\n"
            "[Old](/holoscan/sensor-bridge/2-7-latest/)\n"
        )
        self.freeze()
        text = (self.snapshot / "intro.mdx").read_text()
        self.assertIn("[Home](/holoscan/sensor-bridge/2-7)", text)
        self.assertNotIn("2-7-latest", text)

    def test_explicit_historical_release_links_are_preserved(self):
        content = (
            "[Old](https://docs.nvidia.com/holoscan/sensor-bridge/1.1.0/setup.html)\n"
            "[Firmware](https://docs.nvidia.com/holoscan/sensor-bridge/2.0.0/sensor_bridge_firmware_setup.html)\n"
        )
        (self.guide / "intro.mdx").write_text(content)
        self.freeze()
        self.assertEqual((self.snapshot / "intro.mdx").read_text(), content)

    def render_fixture(self):
        body = (
            "Threshold \\<10 ns and \\>1 ns.\n\n"
            "Inline `\\<literal\\>` stays code.\n"
            "Double ``\\<second\\>`` stays code.\n"
            "```cpp\nvalue \\< limit;\n```\n"
            "~~~text\n\\<fenced\\>\n~~~\n"
            "Even backslashes \\\\<unescaped stay unchanged.\n"
        )
        (self.guide / "intro.mdx").write_text(body)
        self.freeze()
        before = ARCHIVE.inventory(self.snapshot)
        raw_manifest = (self.snapshot / ARCHIVE.MANIFEST).read_bytes()
        fern = ARCHIVE.assemble(self.guide, self.snapshot, self.root / "output")
        return body, before, raw_manifest, fern

    def test_rendering_entities_preserve_code_and_original_archive(self):
        body, before, raw_manifest, fern = self.render_fixture()
        rendered = (fern / "v2.7/intro.mdx").read_text()
        self.assertIn("Threshold &lt;10 ns and &gt;1 ns.", rendered)
        self.assertIn("Inline `\\<literal\\>` stays code.", rendered)
        self.assertIn("Double ``\\<second\\>`` stays code.", rendered)
        self.assertIn("```cpp\nvalue \\< limit;\n```", rendered)
        self.assertIn("~~~text\n\\<fenced\\>\n~~~", rendered)
        self.assertIn("Even backslashes \\\\<unescaped", rendered)
        self.assertEqual(
            html.unescape(rendered.splitlines()[0]), "Threshold <10 ns and >1 ns."
        )
        self.assertEqual((self.snapshot / "intro.mdx").read_text(), body)
        self.assertEqual((fern.parent / "intro.mdx").read_text(), body)
        self.assertEqual(ARCHIVE.inventory(self.snapshot), before)
        self.assertEqual((self.snapshot / ARCHIVE.MANIFEST).read_bytes(), raw_manifest)
        ARCHIVE.check_snapshot(self.snapshot)
        ARCHIVE.check_snapshot(fern / "v2.7")

    def test_derived_rendering_manifest_records_truthful_source_and_content_hashes(
        self,
    ):
        body, before, raw_manifest, fern = self.render_fixture()
        raw_path = fern / "archive-source-v2.7.json"
        self.assertEqual(raw_path.read_bytes(), raw_manifest)
        manifest = ARCHIVE.check_snapshot(fern / "v2.7")
        rendering = manifest["rendering"]
        self.assertEqual(
            rendering["source_manifest_sha256"],
            hashlib.sha256(raw_manifest).hexdigest(),
        )
        self.assertEqual(
            rendering["transformation"], "fern-image-scanner-workarounds-outside-code"
        )
        self.assertEqual(set(rendering["files"]), {"intro.mdx"})
        self.assertEqual(
            rendering["files"]["intro.mdx"]["before_sha256"], before["intro.mdx"]
        )
        self.assertEqual(
            rendering["files"]["intro.mdx"]["after_sha256"],
            manifest["files"]["intro.mdx"],
        )
        self.assertEqual(manifest["files"], ARCHIVE.inventory(fern / "v2.7"))

    def test_code_angles_only_rewrite_endangered_images_in_derived_archive(self):
        heredoc = (
            "```none\nsudo tee /tmp/pin >/dev/null <<'EOF'\nPackage: nvidia\nEOF\n```\n"
        )
        examples = "`![Inline](image.png)`\n```md\n![Example](image.png)\n```\n"
        ending = "```none\nAPPEND ...<other-settings>...\n```\n"
        body = (
            "# Setup\n\n## Install\n![Before](image.png)\n"
            + heredoc
            + "![IGX QSFP0](image.png)\n"
            + examples
            + "![IGX QSFP1](image.png)\n"
            + ending
            + "![After](image.png)\n"
        )
        (self.guide / "setup.mdx").write_text(body)
        self.freeze()
        original = ARCHIVE.inventory(self.snapshot)
        raw_manifest = (self.snapshot / ARCHIVE.MANIFEST).read_bytes()
        fern = ARCHIVE.assemble(self.guide, self.snapshot, self.root / "output")
        rendered = (fern / "v2.7/setup.mdx").read_text()
        expected = body
        for alt in ("IGX QSFP0", "IGX QSFP1"):
            expected = expected.replace(
                f"![{alt}](image.png)", f'<img alt="{alt}" src="image.png" />'
            )
        self.assertEqual(rendered, expected)
        self.assertEqual((self.snapshot / "setup.mdx").read_text(), body)
        self.assertEqual(ARCHIVE.inventory(self.snapshot), original)
        self.assertEqual((self.snapshot / ARCHIVE.MANIFEST).read_bytes(), raw_manifest)
        self.assertEqual((fern / "archive-source-v2.7.json").read_bytes(), raw_manifest)
        manifest = ARCHIVE.check_snapshot(fern / "v2.7")
        self.assertEqual(set(manifest["rendering"]["files"]), {"setup.mdx"})
        self.assertEqual(manifest["files"], ARCHIVE.inventory(fern / "v2.7"))
        ARCHIVE.check_snapshot(self.snapshot)

    def test_scanner_safe_images_preserve_alt_and_asset_attribute_values(self):
        body = "`<unclosed`\n![Port `A` & cable](assets/qsfp.webp?x=1&y=2)\n"
        self.assertEqual(
            ARCHIVE.scanner_safe_images(body),
            '`<unclosed`\n<img alt="Port `A` &amp; cable" src="assets/qsfp.webp?x=1&amp;y=2" />\n',
        )
        self.assertEqual(
            ARCHIVE.scanner_safe_images("`<closed>`\n![Safe](image.png)\n"),
            "`<closed>`\n![Safe](image.png)\n",
        )

    def test_derived_source_manifest_tampering_is_rejected(self):
        body, before, raw_manifest, fern = self.render_fixture()
        (fern / "archive-source-v2.7.json").write_bytes(raw_manifest + b"\n")
        with self.assertRaisesRegex(ValueError, "source manifest hash differs"):
            ARCHIVE.check_snapshot(fern / "v2.7")

    def test_derived_rendering_hash_record_tampering_is_rejected(self):
        body, before, raw_manifest, fern = self.render_fixture()
        path = fern / "v2.7" / ARCHIVE.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["rendering"]["files"]["intro.mdx"]["before_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "rendering hashes differ"):
            ARCHIVE.check_snapshot(fern / "v2.7")

    def test_inline_code_heading_links_and_duplicate_ids_match_fern(self):
        (self.guide / "setup.mdx").write_text(
            "# Setup\n\n## Install `SDK`\n## Install ``SDK``\n"
            "## Use `<SDK>`\n## Use ``SDK `tool` ``\n"
        )
        (self.guide / "intro.mdx").write_text(
            "[Install](setup.mdx#install-sdk)\n"
            "[Again](setup.mdx#install-sdk-1)\n"
            "[Type](setup.mdx#use-sdk)\n"
            "[Tool](setup.mdx#use-sdk-tool)\n"
        )
        self.freeze()
        ARCHIVE.check_snapshot(self.snapshot)
        self.assertTrue(
            {"install-sdk", "install-sdk-1", "use-sdk", "use-sdk-tool"}
            <= ARCHIVE.anchors(self.snapshot / "setup.mdx")
        )

    def test_heading_fix_keeps_code_examples_out_of_explicit_anchors_and_links(self):
        (self.guide / "setup.mdx").write_text(
            "# Setup\n## Install `SDK`\n"
            '<Anchor id="real-explicit" />\n'
            '`<Anchor id="inline-only" /> [#inline-marker]`\n'
            '``<a name="multi-only"></a>``\n'
            '```md\n## Install `SDK`\n<Anchor id="fenced-only" />\n[#fenced-marker]\n```\n'
        )
        examples = (
            "`[Example](setup.mdx#missing-inline)`\n"
            "``[Example](setup.mdx#missing-multi)``\n"
            "```md\n[Example](setup.mdx#missing-fenced)\n```\n"
        )
        (self.guide / "intro.mdx").write_text(
            "[Install](setup.mdx#install-sdk)\n[Explicit](setup.mdx#real-explicit)\n"
            + examples
        )
        self.freeze()
        actual = ARCHIVE.anchors(self.snapshot / "setup.mdx")
        self.assertIn("real-explicit", actual)
        self.assertTrue(
            {
                "inline-only",
                "inline-marker",
                "multi-only",
                "fenced-only",
                "fenced-marker",
                "install-sdk-1",
            }.isdisjoint(actual)
        )
        self.assertIn(examples, (self.snapshot / "intro.mdx").read_text())

    def test_missing_inline_code_heading_anchor_still_fails(self):
        (self.guide / "setup.mdx").write_text("# Setup\n## Install `SDK`\n")
        (self.guide / "intro.mdx").write_text(
            "[Missing](setup.mdx#install-sdk-missing)\n"
        )
        with self.assertRaisesRegex(ValueError, "missing destination anchor"):
            self.freeze()


if __name__ == "__main__":
    main()
