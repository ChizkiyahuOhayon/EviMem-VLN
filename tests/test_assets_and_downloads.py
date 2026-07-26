import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from evimem.assets import verify_assets
from evimem.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _complete_asset_tree(root):
    checkpoint = root / "checkpoints" / "gavln_official"
    _write(checkpoint / "config.json")
    _write(checkpoint / "model.safetensors.index.json")
    _write(checkpoint / "model-00001-of-00001.safetensors")

    siglip = root / "model" / "siglip-so400m-patch14-384"
    _write(siglip / "config.json")
    _write(siglip / "model.safetensors")

    vggt = root / "model" / "VGGT-1B"
    _write(vggt / "config.json")
    _write(vggt / "model.safetensors")

    for split in ("train", "val_seen", "val_unseen"):
        _write(root / "vln_data" / "datasets" / "r2r" / split / f"{split}.json.gz")
    for scene_id in range(90):
        _write(
            root
            / "vln_data"
            / "scene_datasets"
            / "mp3d"
            / f"scene-{scene_id:03d}"
            / f"scene-{scene_id:03d}.glb"
        )


class AssetVerificationTests(unittest.TestCase):
    def test_complete_baseline_assets_are_ready_without_optional_rxr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _complete_asset_tree(root)
            report = verify_assets(root)
            self.assertTrue(report["baseline_ready"])
            statuses = {item["name"]: item["status"] for item in report["checks"]}
            self.assertEqual(statuses["rxr_vlnce"], "missing")
            self.assertTrue(all(statuses[name] == "ok" for name in statuses if name != "rxr_vlnce"))
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["verify-assets", "--project-root", str(root)])
            self.assertEqual(result, 0)

    def test_empty_or_missing_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _complete_asset_tree(root)
            (root / "model" / "VGGT-1B" / "model.safetensors").write_bytes(b"")
            report = verify_assets(root)
            self.assertFalse(report["baseline_ready"])
            status = {item["name"]: item["status"] for item in report["checks"]}
            self.assertEqual(status["vggt"], "invalid")
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["verify-assets", "--project-root", str(root), "--json"])
            self.assertEqual(result, 2)


class DownloadHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = _load_script("download_models.py")
        cls.vlnce = _load_script("download_vlnce.py")

    def test_model_revisions_are_immutable_hashes(self):
        for model in self.models.MODELS.values():
            self.assertRegex(model.revision, r"^[0-9a-f]{40}$")

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escaped.txt", "no")
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                self.vlnce._safe_extract(archive, root / "extract")
            self.assertFalse((root / "escaped.txt").exists())

    def test_r2r_normalization_uses_expected_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "archive" / "R2R_VLNCE_v1-3"
            for split in ("train", "val_seen", "val_unseen"):
                _write(extracted / f"{split}.json.gz", split.encode())
            copied = self.vlnce._normalize_r2r(extracted, root / "datasets")
            self.assertEqual(set(copied), {"train", "val_seen", "val_unseen"})
            for split in copied:
                self.assertEqual(
                    Path(copied[split]).read_bytes(),
                    split.encode(),
                )


if __name__ == "__main__":
    unittest.main()
