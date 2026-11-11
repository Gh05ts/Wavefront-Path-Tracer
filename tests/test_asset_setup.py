"""Run with: python -m unittest discover -s tests -p test_asset_setup.py"""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import setup_assets as setup
from convert_dds_gltf import convert
from prepare_gltf_lighting import prepare_lighting


class AssetSetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_offline_lighting_rebases_assets_and_preserves_emission_and_source(self):
        source_dir = self.root / "source"
        source_dir.mkdir()
        (source_dir / "mesh.bin").write_bytes(b"mesh")
        (source_dir / "color.png").write_bytes(b"texture")
        doc = {"asset": {"version": "2.0"}, "buffers": [{"uri": "mesh.bin"}],
               "images": [{"uri": "color.png"}],
               "extensions": {"KHR_lights_punctual": {"lights": [
                   {"type": "directional", "intensity": 6830},
                   {"type": "point", "intensity": 300}]}},
               "materials": [{"emissiveFactor": [0, 11.4, .52]}, {}]}
        source = source_dir / "source.gltf"
        source.write_text(json.dumps(doc))
        original = source.read_bytes()
        settings = self.root / "settings.json"
        settings.write_text(json.dumps({"punctual_intensity_scale": 2.5/6830}))
        output = self.root / "derived/scene.gltf"
        prepare_lighting(source, output, settings)
        actual = json.loads(output.read_text())
        lights = actual["extensions"]["KHR_lights_punctual"]["lights"]
        self.assertAlmostEqual(lights[0]["intensity"], 2.5)
        self.assertAlmostEqual(lights[1]["intensity"], 300*2.5/6830)
        self.assertEqual(actual["materials"], doc["materials"])
        self.assertEqual((output.parent / actual["images"][0]["uri"]).resolve(), source_dir / "color.png")
        self.assertEqual(source.read_bytes(), original)
        stamp = output.stat().st_mtime_ns
        prepare_lighting(source, output, settings, verify_only=True)
        self.assertEqual(output.stat().st_mtime_ns, stamp)
        settings.write_text(json.dumps({"punctual_intensity_scale": 1}))
        with self.assertRaisesRegex(ValueError, "preserved"):
            prepare_lighting(source, output, settings)
        settings.write_text(json.dumps({"punctual_intensity_scale": -1}))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            prepare_lighting(source, self.root / "invalid.gltf", settings)

    def record(self, payload=b"runtime asset"):
        return {"path": "scene/mesh.bin", "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    def test_fetch_verifies_then_reuses_without_writing_or_network(self):
        payload = b"runtime asset"
        record = self.record(payload)
        with patch.object(setup, "urlopen", return_value=io.BytesIO(payload)):
            self.assertTrue(setup.fetch_lfs(self.root, record, "https://example.test/asset"))
        target = self.root / record["path"]
        before = target.stat().st_mtime_ns
        with patch.object(setup, "urlopen", side_effect=AssertionError("unexpected network")):
            self.assertFalse(setup.fetch_lfs(self.root, record, "unused"))
        self.assertEqual(target.stat().st_mtime_ns, before)
        target.write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "preserved"):
            setup.fetch_lfs(self.root, record, "unused")
        self.assertEqual(target.read_bytes(), b"edited")

    def test_bad_download_preserves_pointer_and_removes_temporary_file(self):
        record = self.record(b"good")
        target = self.root / record["path"]
        target.parent.mkdir()
        pointer = setup.LFS_HEADER + b"\noid sha256:example\n"
        target.write_bytes(pointer)
        for response in (b"evil", b"oversized", b"x"):
            with self.subTest(response=response):
                with patch.object(setup, "urlopen", return_value=io.BytesIO(response)):
                    with self.assertRaises(ValueError):
                        setup.fetch_lfs(self.root, record, "https://example.test/asset")
                self.assertEqual(target.read_bytes(), pointer)
                self.assertEqual(list(target.parent.glob("*.download")), [])

    def test_verify_only_and_path_escape(self):
        with patch.object(setup, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaises(FileNotFoundError):
                setup.fetch_lfs(self.root, self.record(), "unused", verify_only=True)
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaises(ValueError):
            setup.safe_path(self.root, "../outside")

    def test_document_dependencies_reject_lfs_pointers(self):
        target = self.root / "scene.gltf"
        target.write_text(json.dumps({"buffers": [{"uri": "mesh.bin", "byteLength": 4}]}))
        (self.root / "mesh.bin").write_bytes(setup.LFS_HEADER)
        with self.assertRaises(FileNotFoundError):
            setup.verify_document(self.root, "scene.gltf")
        (self.root / "mesh.bin").write_bytes(b"data")
        setup.verify_document(self.root, "scene.gltf")

    def test_converter_reuses_recovers_and_verifies_without_rewriting(self):
        Image.new("RGBA", (4, 4), (128, 64, 32, 255)).save(self.root / "texture.dds")
        source = self.root / "scene.gltf"
        source.write_text(json.dumps({"asset": {"version": "2.0"},
            "images": [{"uri": "texture.dds"}, {"uri": "texture.png"}],
            "textures": [{"extensions": {"MSFT_texture_dds": {"source": 0}}}],
            "extensionsUsed": ["MSFT_texture_dds"]}))
        derived = convert(source)
        png = self.root / "texture.png"
        before = (derived.stat().st_mtime_ns, png.stat().st_mtime_ns)
        convert(source, reuse=True)
        self.assertEqual((derived.stat().st_mtime_ns, png.stat().st_mtime_ns), before)
        self.assertEqual(len(json.loads(derived.read_text())["images"]), 1)
        png.unlink()
        with self.assertRaises(FileNotFoundError):
            convert(source, verify_only=True)
        self.assertFalse(png.exists())
        convert(source, reuse=True)
        self.assertTrue(png.is_file())
        self.assertEqual(derived.stat().st_mtime_ns, before[0])
        derived.write_text("{}")
        with self.assertRaisesRegex(ValueError, "differs"):
            convert(source, reuse=True)

    def test_fresh_submodule_initialization_is_pinned_and_sparse(self):
        origin, parent = self.root / "upstream", self.root / "parent"
        for repo in (origin, parent):
            repo.mkdir()
            setup.git(repo, "init", "--quiet")
            setup.git(repo, "config", "user.name", "Asset test")
            setup.git(repo, "config", "user.email", "assets@example.test")
        (origin / "selected").mkdir()
        (origin / "unused").mkdir()
        (origin / "selected/scene.gltf").write_text("{}")
        (origin / "unused/source.txt").write_text("unused source")
        (origin / "LICENSE").write_text("test license")
        setup.git(origin, "add", ".")
        setup.git(origin, "commit", "--quiet", "-m", "fixture")
        revision = setup.git(origin, "rev-parse", "HEAD")
        source = {"path": "assets/source", "revision": revision, "sparse_paths": ["selected"]}
        setup.git(parent, "config", "-f", ".gitmodules", "submodule.assets/source.path", source["path"])
        setup.git(parent, "config", "-f", ".gitmodules", "submodule.assets/source.url", str(origin))
        setup.git(parent, "add", ".gitmodules")
        setup.git(parent, "update-index", "--add", "--cacheinfo", "160000", revision, source["path"])
        setup.git(parent, "commit", "--quiet", "-m", "fixture parent")
        with self.assertRaises(FileNotFoundError):
            setup.initialize(parent, source, verify_only=True)
        with patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            checkout = setup.initialize(parent, source)
        self.assertEqual(setup.git(checkout, "rev-parse", "HEAD"), revision)
        self.assertTrue((checkout / "LICENSE").is_file())
        self.assertTrue((checkout / "selected/scene.gltf").is_file())
        self.assertFalse((checkout / "unused/source.txt").exists())
        (checkout / "selected/scene.gltf").write_text("edited")
        setup.initialize(parent, source)
        self.assertEqual((checkout / "selected/scene.gltf").read_text(), "edited")
        with self.assertRaisesRegex(ValueError, "gitlink"):
            setup.initialize(parent, dict(source, revision="0" * 40))


if __name__ == "__main__":
    unittest.main()
