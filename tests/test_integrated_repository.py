import subprocess
import sys
import unittest
from pathlib import Path

from gavln.memory.base import MemoryBackendConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrationTests(unittest.TestCase):
    def test_internal_packages_import_without_external_checkout(self):
        import evimem
        import gavln
        import llava
        import vggt

        self.assertTrue(Path(evimem.__file__).resolve().is_relative_to(REPO_ROOT))
        self.assertTrue(Path(gavln.__file__).resolve().is_relative_to(REPO_ROOT))
        self.assertTrue(Path(llava.__file__).resolve().is_relative_to(REPO_ROOT))
        self.assertTrue(Path(vggt.__file__).resolve().is_relative_to(REPO_ROOT))

    def test_eval_help_is_lightweight(self):
        result = subprocess.run(
            [sys.executable, "-m", "gavln.gavln_eval", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--memory_backend", result.stdout)

    def test_missing_assets_are_reported_before_heavy_imports(self):
        result = subprocess.run(
            [sys.executable, "-m", "gavln.gavln_eval"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Required runtime assets are missing", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_backend_configuration_rejects_invalid_limits(self):
        MemoryBackendConfig(name="gavln")
        MemoryBackendConfig(name="evimem", horizon="route")
        with self.assertRaises(ValueError):
            MemoryBackendConfig(name="evimem", resident_slots=0)
        with self.assertRaises(ValueError):
            MemoryBackendConfig(name="evimem", token_budget=0)

    def test_no_external_ga_vln_runtime_dependency(self):
        self.assertFalse((REPO_ROOT / ".gitmodules").exists())
        forbidden = ("GA_VLN_ROOT", "sys.path.insert(", "bootstrap_ga_vln")
        offenders = []
        for suffix in ("*.py", "*.sh"):
            for path in REPO_ROOT.rglob(suffix):
                if ".git" in path.parts or path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                for marker in forbidden:
                    if marker in text:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
                if "git clone" in text and "GA-VLN" in text:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}: runtime GA-VLN clone"
                    )
        self.assertEqual(offenders, [])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is installed by the full CUDA environment")
class TorchMemoryBackendTests(unittest.TestCase):
    def _batch(self, observation_id, action_step, offset=0.0):
        from gavln.memory.base import MemoryBatch

        return MemoryBatch(
            siglip_features=torch.tensor(
                [[[1.0 + offset, 2.0], [3.0, 4.0]]]
            ),
            siglip_world_xy=torch.tensor(
                [[[0.1 + offset, 0.1], [1.1 + offset, 0.1]]]
            ),
            vggt_features=torch.tensor([[[5.0, 6.0]]]),
            vggt_world_xy=torch.tensor([[[0.2 + offset, 0.1]]]),
            observation_ids=torch.tensor([observation_id]),
            action_steps=torch.tensor([action_step]),
            agent_position=torch.tensor([0.0, 0.0]),
            agent_rotation=torch.eye(2),
        )

    def test_gavln_and_evimem_initialize(self):
        from gavln.memory import create_memory_backend

        baseline = create_memory_backend(
            MemoryBackendConfig(
                name="gavln", cell_size=1.0, grid_size=4, feature_dim=2
            )
        )
        baseline.reset("g")
        self.assertGreater(baseline.build_tokens(self._batch(0, 0)).features.shape[0], 0)

        evimem = create_memory_backend(
            MemoryBackendConfig(
                name="evimem",
                horizon="route",
                resident_slots=4,
                token_budget=1,
                cell_size=1.0,
                grid_size=4,
                feature_dim=2,
            )
        )
        evimem.reset("e")
        batch = self._batch(0, 0)
        evimem.update(batch)
        tokens = evimem.build_tokens(batch)
        self.assertLessEqual(tokens.features.shape[0], 1)
        self.assertLessEqual(evimem.active_count, 4)
        for observation_id in range(1, 40):
            evimem.update(self._batch(observation_id, observation_id))
        self.assertEqual(len(evimem.observation_buffer), 33)
        self.assertEqual(evimem.observation_buffer[0]["observation_id"], 7)

    def test_gavln_backend_matches_upstream_scatter_mean(self):
        from gavln.memory import create_memory_backend

        config = MemoryBackendConfig(
            name="gavln", cell_size=1.0, grid_size=4, feature_dim=2
        )
        batch = self._batch(0, 0)
        actual = create_memory_backend(config).build_tokens(batch)

        features = torch.cat(
            (
                batch.siglip_features.reshape(-1, 2),
                batch.vggt_features.reshape(-1, 2),
            )
        )
        points = torch.cat(
            (
                batch.siglip_world_xy.reshape(-1, 2),
                batch.vggt_world_xy.reshape(-1, 2),
            )
        )
        local = (points - batch.agent_position) @ (-batch.agent_rotation)
        indices = torch.floor(local / config.cell_size).long() + config.grid_size // 2
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < config.grid_size)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < config.grid_size)
        )
        flat = indices[valid, 1] * config.grid_size + indices[valid, 0]
        expected = torch.zeros(config.grid_size**2, config.feature_dim)
        counts = torch.zeros(config.grid_size**2)
        expected.scatter_add_(0, flat[:, None].expand(-1, 2), features[valid])
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        occupied = counts > 0
        expected[occupied] /= counts[occupied, None]

        self.assertTrue(torch.equal(actual.flat_indices, torch.nonzero(occupied).flatten()))
        self.assertTrue(torch.equal(actual.features, expected[occupied]))

    def test_reset_isolation_horizon_and_determinism(self):
        from gavln.memory import create_memory_backend

        config = MemoryBackendConfig(
            name="evimem",
            horizon=8,
            resident_slots=2,
            token_budget=2,
            seed=7,
            cell_size=1.0,
            grid_size=8,
            feature_dim=2,
        )
        first = create_memory_backend(config)
        second = create_memory_backend(config)
        first.reset("same")
        second.reset("same")
        batch = self._batch(0, 0)
        first.update(batch)
        second.update(batch)
        first_tokens = first.build_tokens(batch)
        second_tokens = second.build_tokens(batch)
        self.assertTrue(torch.equal(first_tokens.flat_indices, second_tokens.flat_indices))
        self.assertTrue(torch.equal(first_tokens.features, second_tokens.features))

        second.reset("other")
        self.assertEqual(second.active_count, 0)
        self.assertEqual(len(second.observation_buffer), 0)
        self.assertGreater(first.active_count, 0)

        first.update(self._batch(1, 9, offset=3.0))
        self.assertLessEqual(first.active_count, 2)
        self.assertTrue(all(key[0] >= 3 for key in first._key_to_slot))


if __name__ == "__main__":
    unittest.main()
