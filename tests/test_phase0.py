import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evimem.phase0.config import Phase0Config
from evimem.phase0.integration import Phase0MemoryState
from evimem.phase0.manifest import (
    ResumeMismatchError,
    assert_resume_compatible,
    build_manifest,
    partition_episode_ids,
    write_manifest,
)
from evimem.phase0.memory import (
    FallbackRateError,
    QuotaMonitor,
    RasterizedGrid,
    ResidualUnderfillError,
    SelectionResult,
    SparseLedger,
    agent_to_world,
    bf16_to_float32,
    blake2b63,
    enforce_exact_quota,
    incidence_grid_from_raw,
    quantize_world,
    rasterize_ledger,
    scatter_mean,
    stable_select,
    world_to_agent,
)
from evimem.phase0.noise import (
    apply_depth_offset,
    apply_local_se2,
    assert_production_payload,
    generate_matched_noise_pair,
    lag1_autocorrelation,
    noise_pair_manifest,
    time_permute_trace,
    translation_rms,
)
from evimem.phase0.ring import MissingObservationError, ObservationRing
from evimem.phase0.trace import LegacyTraceRecorder, compare_a0a_fixture


def _ring_payload(value):
    array = np.asarray([value], dtype=np.float32)
    return {
        "siglip_image": array,
        "vggt_image": array + 1,
        "world_xy_siglip": np.asarray([[value, 0]], dtype=np.float32),
        "world_xy_vggt": np.asarray([[value, 1]], dtype=np.float32),
        "position": np.asarray([value, 0], dtype=np.float32),
        "rotation": np.eye(2, dtype=np.float32),
    }


class ConfigAndRingTests(unittest.TestCase):
    def test_composable_config_modes(self):
        gavln = Phase0Config(carrier="gavln", token_cap="none", horizon=None)
        capped = Phase0Config(carrier="gavln", token_cap="B_t", horizon=None)
        sparse = Phase0Config(carrier="sparse", token_cap="K_eq", horizon="route")
        self.assertEqual(gavln.experiment_mode, "gavln")
        self.assertEqual(capped.experiment_mode, "gavln_cap")
        self.assertEqual(sparse.experiment_mode, "sparse_hroute")
        self.assertIsNone(sparse.horizon_actions)
        with self.assertRaises(ValueError):
            Phase0Config(carrier="sparse", horizon=16)
        with self.assertRaises(ValueError):
            Phase0Config(carrier="gavln", token_cap="none", horizon=32)

    def test_ring_event_chronology_legacy_and_resets(self):
        ring = ObservationRing()
        self.assertIsNone(
            ring.push_generation_event(2, 0, **_ring_payload(0))
        )
        for step in range(33):
            record = ring.push_generation_event(0, step, **_ring_payload(step))
            self.assertEqual(record.observation_id, step)
            self.assertEqual(record.generation_id, step)
        self.assertEqual(ring.storage_slots, 33)
        self.assertEqual(len(ring), 33)
        self.assertEqual(
            [record.observation_id for record in ring.gather(32, 32)], list(range(33))
        )
        self.assertEqual(
            [record.observation_id for record in ring.gather_legacy([2, 0, 2])],
            [2, 0, 2],
        )

        ring.dialogue_reset()
        self.assertEqual(len(ring), 33)
        ring.push_generation_event(0, 33, **_ring_payload(33))
        self.assertEqual(
            [record.observation_id for record in ring.gather(32, 33)], list(range(1, 34))
        )
        with self.assertRaises(MissingObservationError):
            ring.gather_legacy([0])

        storage_id = id(ring._records)
        for step in range(34, 5000):
            ring.push_generation_event(0, step, **_ring_payload(step))
        self.assertEqual(id(ring._records), storage_id)
        self.assertEqual(ring.storage_slots, 33)
        self.assertEqual(len(ring), 33)

        ring.episode_reset()
        self.assertEqual(len(ring), 0)
        self.assertEqual(ring.next_generation_id, 0)
        self.assertEqual(ring.next_observation_id, 0)


class LedgerTests(unittest.TestCase):
    def test_coordinates_negative_floor_and_inclusive_ttl(self):
        keys = quantize_world(
            np.asarray([[-0.01, 0.0], [-0.25, 0.0], [-0.251, 0.0]])
        )
        np.testing.assert_array_equal(keys[:, 0], np.asarray([-1, -1, -2]))

        world = np.asarray([[1.0, 2.0], [-0.5, 0.25]])
        position = np.asarray([0.2, -0.3])
        local = world_to_agent(world, position, 0.7)
        np.testing.assert_allclose(agent_to_world(local, position, 0.7), world, atol=1e-12)

        ledger = SparseLedger("ttl", resident_slots=4, feature_dim=2)
        ledger.update_observation(
            0,
            0,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            horizon=8,
        )
        self.assertEqual(ledger.expire(8, 8), 0)
        self.assertEqual(ledger.active_count, 1)
        self.assertEqual(ledger.expire(9, 8), 1)
        self.assertEqual(ledger.active_count, 0)

        route = SparseLedger("route", resident_slots=4, feature_dim=2)
        route.update_observation(
            0,
            0,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            horizon="route",
        )
        self.assertEqual(route.expire(100000, "route"), 0)
        self.assertEqual(route.active_count, 1)
        route.reset("new-episode")
        self.assertEqual(route.active_count, 0)

    def test_fixed_schema_bytes_and_10000_write_bottom_k(self):
        fixed = SparseLedger("schema")
        schema = fixed.tensor_schema()
        self.assertEqual(schema["keys"], ("int32", (2048, 2)))
        self.assertEqual(schema["features"], ("bfloat16", (2048, 1152)))
        self.assertEqual(schema["support_count"], ("uint16", (2048,)))
        self.assertEqual(schema["active"], ("bool", (2048,)))
        self.assertEqual(fixed.logical_bytes, 4_798_464)

        ledger = SparseLedger("many", seed=17, resident_slots=2048, feature_dim=8)
        array_ids = {
            name: id(getattr(ledger, name))
            for name in (
                "keys",
                "features",
                "weight",
                "confidence",
                "priority",
                "reservoir_key",
                "first_step",
                "last_step",
                "support_count",
                "active",
            )
        }
        logical_bytes = ledger.logical_bytes
        for observation_id in range(100):
            start = observation_id * 100
            cell_ids = np.arange(start, start + 100, dtype=np.float64)
            points = np.column_stack(((cell_ids + 0.5) * 0.25, np.full(100, 0.125)))
            features = np.repeat(
                (cell_ids % 13).astype(np.float32)[:, None], 8, axis=1
            )
            ledger.update_observation(
                observation_id,
                observation_id,
                points,
                features,
                horizon="route",
            )
        self.assertEqual(ledger.active_count, 2048)
        self.assertEqual(ledger.resident_overflow, 0)
        self.assertEqual(ledger.logical_bytes, logical_bytes)
        for name, original_id in array_ids.items():
            self.assertEqual(id(getattr(ledger, name)), original_id)

        expected = sorted(
            (blake2b63(17, "many", x, 0), x, 0) for x in range(10_000)
        )[:2048]
        self.assertEqual(ledger.resident_tuples(), expected)
        self.assertEqual(
            ledger.stats.inserted
            + ledger.stats.replaced
            + ledger.stats.rejected,
            10_000,
        )

    def test_existing_update_dedup_expiry_and_order_independence(self):
        self.assertEqual(blake2b63(17, "ep-α", -2, 3), 1_998_489_119_542_377_873)
        ledger = SparseLedger("updates", seed=3, resident_slots=4, feature_dim=1)
        ledger.update_observation(
            0,
            0,
            np.asarray([[0.1, 0.1], [0.2, 0.2]]),
            np.asarray([[1.0], [3.0]], dtype=np.float32),
            horizon=2,
        )
        slot = int(np.flatnonzero(ledger.active)[0])
        self.assertEqual(int(ledger.support_count[slot]), 1)
        self.assertAlmostEqual(float(bf16_to_float32(ledger.features[slot])[0]), 2.0)

        ledger.update_observation(
            1,
            1,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[4.0]], dtype=np.float32),
            horizon=2,
        )
        self.assertEqual(int(ledger.support_count[slot]), 2)
        self.assertEqual(float(ledger.weight[slot]), 2.0)
        self.assertAlmostEqual(float(bf16_to_float32(ledger.features[slot])[0]), 3.0)

        ledger.update_observation(
            2,
            4,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[8.0]], dtype=np.float32),
            horizon=2,
        )
        slot = int(np.flatnonzero(ledger.active)[0])
        self.assertEqual(ledger.stats.expired, 1)
        self.assertEqual(int(ledger.first_step[slot]), 4)
        self.assertEqual(int(ledger.support_count[slot]), 1)
        self.assertAlmostEqual(float(bf16_to_float32(ledger.features[slot])[0]), 8.0)

        keys = [(x, 0) for x in range(10)]
        first = SparseLedger("order", seed=9, resident_slots=3, feature_dim=1)
        second = SparseLedger("order", seed=9, resident_slots=3, feature_dim=1)
        for observation_id, key in enumerate(keys):
            first.update_observation(
                observation_id,
                observation_id,
                np.asarray([[(key[0] + 0.5) * 0.25, 0.125]]),
                np.asarray([[key[0]]], dtype=np.float32),
                horizon="route",
            )
        for observation_id, key in enumerate(reversed(keys)):
            second.update_observation(
                observation_id,
                observation_id,
                np.asarray([[(key[0] + 0.5) * 0.25, 0.125]]),
                np.asarray([[key[0]]], dtype=np.float32),
                horizon="route",
            )
        self.assertEqual(first.resident_tuples(), second.resident_tuples())

    def test_incidence_rasterization_and_raw_dedup_reference(self):
        ledger = SparseLedger("raster", resident_slots=8, feature_dim=1)
        ledger.update_observation(
            0,
            0,
            np.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.1]]),
            np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
            horizon="route",
        )
        ledger.update_observation(
            1,
            1,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[6.0]], dtype=np.float32),
            horizon="route",
        )
        grid = rasterize_ledger(ledger, [0.0, 0.0], 0.0)
        self.assertEqual(int(grid.incidence[40, 40]), 2)
        self.assertEqual(int(grid.incidence[40, 41]), 1)
        self.assertAlmostEqual(float(grid.features[40, 40, 0]), 4.0)
        self.assertAlmostEqual(float(grid.features[40, 41, 0]), 5.0)

        raw = incidence_grid_from_raw(
            observation_ids=np.asarray([0, 0, 0, 1]),
            world_xy=np.asarray(
                [[0.1, 0.1], [0.2, 0.2], [0.3, 0.1], [0.1, 0.1]]
            ),
            agent_position=[0.0, 0.0],
            agent_yaw=0.0,
        )
        np.testing.assert_array_equal(raw, grid.incidence)


class ReadoutTests(unittest.TestCase):
    def test_selector_order_cap_underfill_and_quota_failure(self):
        grid = RasterizedGrid(
            features=np.arange(4, dtype=np.float32).reshape(2, 2, 1),
            incidence=np.asarray([[2, 2], [3, 0]], dtype=np.int64),
            nonzero=np.asarray([[True, True], [True, False]]),
        )
        selected = stable_select(grid, 2)
        np.testing.assert_array_equal(selected.flat_indices, np.asarray([0, 2]))
        self.assertFalse(selected.underfilled)
        capped = stable_select(grid, 5)
        self.assertTrue(capped.underfilled)
        self.assertEqual(capped.flat_indices.size, 3)

        exact = enforce_exact_quota(
            primary=selected,
            fallback_flat_indices=np.asarray([2, 3, 1]),
            fallback_features=np.asarray([[20.0], [30.0], [10.0]], dtype=np.float32),
            fallback_last_step=np.asarray([100, 90, 80]),
            quota=3,
        )
        np.testing.assert_array_equal(exact.flat_indices, np.asarray([0, 2, 3]))
        self.assertEqual(exact.fallback_count, 1)
        self.assertFalse(exact.underfilled)
        with self.assertRaises(ResidualUnderfillError):
            enforce_exact_quota(
                primary=selected,
                fallback_flat_indices=np.asarray([2]),
                fallback_features=np.asarray([[20.0]], dtype=np.float32),
                fallback_last_step=np.asarray([100]),
                quota=3,
            )

        no_fallback = SelectionResult(
            flat_indices=np.asarray([0]),
            features=np.asarray([[0.0]], dtype=np.float32),
            incidence=np.asarray([1]),
            requested=1,
            underfilled=False,
            fallback_count=0,
        )
        monitor = QuotaMonitor()
        for _ in range(19):
            monitor.observe(no_fallback)
        monitor.observe(
            SelectionResult(
                flat_indices=np.asarray([0]),
                features=np.asarray([[0.0]], dtype=np.float32),
                incidence=np.asarray([1]),
                requested=1,
                underfilled=False,
                fallback_count=1,
            )
        )
        monitor.validate(locked_test=True)
        monitor.observe(no_fallback)
        monitor.observe(
            SelectionResult(
                flat_indices=np.asarray([0]),
                features=np.asarray([[0.0]], dtype=np.float32),
                incidence=np.asarray([1]),
                requested=1,
                underfilled=False,
                fallback_count=1,
            )
        )
        with self.assertRaises(FallbackRateError):
            monitor.validate(locked_test=True)

    def test_sparse_scatter_mean_matches_loop_reference(self):
        indices = np.asarray([0, 1, 0, 2])
        features = np.asarray([[1, 2], [4, 6], [3, 4], [8, 10]], dtype=np.float32)
        weights = np.asarray([1, 2, 3, 1], dtype=np.int64)
        actual, totals = scatter_mean(indices, features, weights, num_cells=4)
        expected = np.zeros((4, 2), dtype=np.float64)
        expected_weight = np.zeros(4, dtype=np.float64)
        for index, feature, weight in zip(indices, features, weights):
            expected[index] += feature * weight
            expected_weight[index] += weight
        for index in range(4):
            if expected_weight[index]:
                expected[index] /= expected_weight[index]
        np.testing.assert_allclose(actual, expected.astype(np.float32))
        np.testing.assert_array_equal(totals, expected_weight)


class NoiseTests(unittest.TestCase):
    def test_matched_noise_autocorrelation_se2_yaw_and_depth(self):
        pair = generate_matched_noise_pair(seed=123, length=512)
        self.assertAlmostEqual(translation_rms(pair.iid.translation), 0.05, places=12)
        self.assertAlmostEqual(translation_rms(pair.correlated.translation), 0.05, places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(pair.iid.yaw ** 2))), math.radians(5), places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(pair.correlated.yaw ** 2))), math.radians(5), places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(pair.iid.depth_offset ** 2))), 0.05, places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(pair.correlated.depth_offset ** 2))), 0.05, places=12)
        self.assertLess(abs(lag1_autocorrelation(pair.iid.translation)), 0.15)
        self.assertGreater(lag1_autocorrelation(pair.correlated.translation), 0.8)
        permuted = time_permute_trace(pair.correlated, seed=456)
        self.assertLess(abs(lag1_autocorrelation(permuted.translation)), 0.15)

        perturbed = apply_local_se2(
            [1.0, 2.0, math.pi / 2], [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(perturbed, [1.0, 3.0, math.pi / 2], atol=1e-12)
        wrapped = apply_local_se2([0.0, 0.0, math.pi - 0.01], [0.0, 0.0, 0.02])
        self.assertGreaterEqual(wrapped[2], -math.pi)
        self.assertLess(wrapped[2], math.pi)
        depth = apply_depth_offset(
            np.asarray([0.01, 1.0, 9.9], dtype=np.float32), -0.5, 0.1, 10.0
        )
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all(depth >= 0.1))
        self.assertTrue(np.all(depth <= 10.0))
        manifest = noise_pair_manifest(pair)
        self.assertEqual(manifest["shape"], [512, 4])
        self.assertEqual(len(manifest["iid"]["sha256"]), 64)

    def test_production_payload_rejects_evaluator_only_fields(self):
        assert_production_payload(
            {
                "rgbd": np.zeros((2, 2)),
                "pose": [0.0, 0.0, 0.0],
                "metadata": {"dialogue_group": 1, "acquisition_step": 4},
            }
        )
        for forbidden in ("gt_pose", "gt_depth", "ancestry", "draw_id"):
            with self.assertRaises(ValueError):
                assert_production_payload({"nested": {forbidden: 1}})


class ReproducibilityAndTraceTests(unittest.TestCase):
    def _make_git_repo(self, root):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Phase0 Test"], check=True)
        (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)

    def test_manifest_worktree_hash_resume_rejection_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            self._make_git_repo(root)
            asset = Path(directory) / "asset.bin"
            asset.write_bytes(b"asset")
            config = Phase0Config(carrier="sparse", token_cap="B_t", horizon=32)
            first = build_manifest(
                experiment_id="synthetic",
                config=config,
                source_root=root,
                asset_paths={"present": asset, "missing": Path(directory) / "missing"},
                checkpoint_hash="checkpoint",
                dataset_hash="dataset",
                episode_list_hash="episodes",
                evaluator_hash="evaluator",
                noise_trace_hash="noise",
            )
            second = build_manifest(
                experiment_id="synthetic",
                config=config,
                source_root=root,
                asset_paths={"present": asset, "missing": Path(directory) / "missing"},
                checkpoint_hash="checkpoint",
                dataset_hash="dataset",
                episode_list_hash="episodes",
                evaluator_hash="evaluator",
                noise_trace_hash="noise",
            )
            assert_resume_compatible(first, second)
            self.assertFalse(first["source"]["dirty"])
            self.assertEqual(first["asset_hashes"]["missing"]["status"], "missing")

            (root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
            changed = build_manifest(
                experiment_id="synthetic",
                config=config,
                source_root=root,
                asset_paths={"present": asset, "missing": Path(directory) / "missing"},
                checkpoint_hash="checkpoint",
                dataset_hash="dataset",
                episode_list_hash="episodes",
                evaluator_hash="evaluator",
                noise_trace_hash="noise",
            )
            self.assertNotEqual(
                first["source"]["worktree_hash"], changed["source"]["worktree_hash"]
            )
            with self.assertRaises(ResumeMismatchError):
                assert_resume_compatible(first, changed)

            changed_config = dict(second)
            changed_config["config_hash"] = "different-config"
            with self.assertRaises(ResumeMismatchError):
                assert_resume_compatible(first, changed_config)

            manifest_path = Path(directory) / "manifest.json"
            write_manifest(manifest_path, changed)
            self.assertEqual(json.loads(manifest_path.read_text())["experiment_id"], "synthetic")

        episode_ids = ["episode-{}".format(index) for index in range(11)]
        ranks = [partition_episode_ids(episode_ids, rank, 2) for rank in range(2)]
        self.assertFalse(set(ranks[0]) & set(ranks[1]))
        self.assertEqual(set(ranks[0]) | set(ranks[1]), set(episode_ids))
        with self.assertRaises(ValueError):
            partition_episode_ids(["duplicate", "duplicate"], 0, 2)

    def test_asset_free_trace_recorder_and_state_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = LegacyTraceRecorder(Path(directory))
            common = dict(
                ids=np.asarray([1, 2], dtype=np.int64),
                frame_ids=np.asarray([0, 1], dtype=np.int64),
                gather_order=np.asarray([0, 1], dtype=np.int64),
                nonzero_mask=np.asarray([True, False]),
                nonzero_indices=np.asarray([0], dtype=np.int64),
                raw_float_inputs={"image": np.asarray([1.0], dtype=np.float32)},
                computed_tensors={"grid": np.asarray([2.0], dtype=np.float32)},
                action_logits=np.asarray([[0.1, 0.9]], dtype=np.float32),
            )
            legacy = recorder.record_refresh(refresh_id="legacy", **common)
            ring = recorder.record_refresh(refresh_id="ring", **common)
            compare_a0a_fixture(legacy, ring)

            changed = dict(common)
            changed["raw_float_inputs"] = {"image": np.asarray([1.1], dtype=np.float32)}
            mismatch = recorder.record_refresh(refresh_id="mismatch", **changed)
            with self.assertRaises(AssertionError):
                compare_a0a_fixture(legacy, mismatch)

        config = Phase0Config(
            carrier="sparse",
            token_cap="B_t",
            horizon=32,
            resident_slots=8,
            feature_dim=2,
        )
        state = Phase0MemoryState(config, episode_id="one")
        record = state.on_generation_event(0, 0, **_ring_payload(0))
        state.ingest_observation_packets(
            record.observation_id,
            0,
            np.asarray([[0.1, 0.1]]),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )
        state.dialogue_reset()
        self.assertEqual(len(state.ring), 1)
        self.assertEqual(state.ledger.active_count, 1)
        state.episode_reset("two")
        self.assertEqual(len(state.ring), 0)
        self.assertEqual(state.ledger.active_count, 0)
        self.assertEqual(state.ledger.episode_id, "two")


if __name__ == "__main__":
    unittest.main()
