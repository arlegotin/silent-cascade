import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

from silent_cascade.train.pilot_config import resolve_pilot_config


def test_debug_audit_is_complete_structural_evidence_but_never_acceptance(tmp_path):
    from silent_cascade.eval.pilot_audit import audit_pilot_manifest
    from silent_cascade.train.pilot_data import freeze_pilot_manifest

    resolved = resolve_pilot_config("phase4_smoke")
    manifest = freeze_pilot_manifest(
        resolved, stage="robustness", output_path=tmp_path / "manifest.json", source_commit="a" * 40
    )
    report = audit_pilot_manifest(manifest, config=resolved.config, output_dir=tmp_path / "audit")
    assert report.schema_version == "phase4-pilot-audit-v1"
    assert report.profile == "debug-non-acceptance"
    assert report.acceptance is False
    assert (
        report.oracle_verified_count
        == report.parent_verified_count
        == report.feasible_trace_count
        == 16
    )
    assert report.variant_counts == {"positive": 8, "safe_negative": 4, "disconnected_negative": 4}
    assert report.hazard_record_count == 32
    assert report.safe_record_count == 16
    assert len(report.projected_episode_sha256s) == 16
    assert report.shortcut_probes == ()
    assert report.statistical_status == "insufficient-debug-corpus"
    assert json.loads((tmp_path / "audit/report.json").read_bytes())["acceptance"] is False


def test_neutral_wrapper_refuses_incomplete_quartets_and_full_minimums():
    from silent_cascade.env.leakage import AuditExample, audit_public_shortcuts
    from silent_cascade.env.pilot import curriculum_to_bundle
    from silent_cascade.eval.pilot_audit import PILOT_AUDIT_PROFILE
    from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example

    config = resolve_pilot_config("phase4_smoke").config
    examples = []
    for index in range(20):
        example = make_curriculum_example(
            config, CurriculumKey("ofd-one-hop-v1", "debug", 449, 457, index, "one_hop")
        )
        examples.append(
            AuditExample(
                curriculum_to_bundle(example, config=config),
                index,
                "independent",
                index // 4,
                index,
                quartet_member_index=index % 4,
            )
        )
    with pytest.raises(ValueError, match=r"count|quartet|held.out"):
        audit_public_shortcuts(
            examples,
            config=config.data.leakage_audit,
            profile=PILOT_AUDIT_PROFILE,
            corpus_hash="a" * 64,
        )
    with pytest.raises(ValueError, match=r"count|quartet"):
        audit_public_shortcuts(
            examples[:-1],
            config=config.data.leakage_audit,
            profile=PILOT_AUDIT_PROFILE,
            corpus_hash="a" * 64,
        )


def test_neutral_kernel_preserves_quartets_and_detects_diagnostic_leak(tmp_path):
    # Native model tests accumulate allocator RSS in the main pytest process.
    # Exercise the real fixed-ceiling audit in a fresh process, as production
    # audits do, without weakening the resource guard or dropping assertions.
    program = """
import runpy
import sys
from pathlib import Path
module = runpy.run_path(sys.argv[1])
module['_assert_neutral_kernel_and_report_integrity'](Path(sys.argv[2]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve()), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_neutral_kernel_and_report_integrity(tmp_path):
    import numpy as np

    from silent_cascade.env import leakage
    from silent_cascade.env.config import LeakageAuditProfileConfig
    from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256, episode_sha256
    from silent_cascade.env.leakage import (
        AuditExample,
        _audit_public_shortcut_controls,
        _prepare_public_shortcuts,
        _run_probes,
        _split_memberships,
        audit_public_shortcuts,
    )
    from silent_cascade.env.pilot import curriculum_to_bundle
    from silent_cascade.hashing import sha256_bytes, sha256_file
    from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example

    config = resolve_pilot_config("phase4_smoke").config
    examples = []
    for index in range(100):
        example = make_curriculum_example(
            config, CurriculumKey("ofd-one-hop-v1", "debug", 449, 457, index, "one_hop")
        )
        examples.append(
            AuditExample(
                curriculum_to_bundle(example, config=config),
                index,
                "independent",
                index // 4,
                index,
                quartet_member_index=index % 4,
            )
        )
    digests = tuple(episode_sha256(e.bundle) for e in examples)
    corpus_hash = corpus_sha256(
        (
            CorpusDigestEntry(e.bundle.public.init.episode_public_id, h)
            for e, h in zip(examples, digests, strict=True)
        ),
        expected_count=100,
    )
    profile = LeakageAuditProfileConfig(
        episode_count=100,
        permutation_replicates=9,
        positive_control_episode_count=100,
        positive_control_permutation_replicates=199,
        minimum_test_examples_per_class=1,
        enforce_clean_statistical_gate=False,
    )
    kwargs = dict(config=config.data.leakage_audit, profile=profile, corpus_hash=corpus_hash)
    values, rows, train, test, membership = _prepare_public_shortcuts(examples, **kwargs)
    assert len(train) == 80 and len(test) == 20
    assert {rows[i].group_id for i in train}.isdisjoint({rows[i].group_id for i in test})
    strict_train, strict_test, strict_membership = _split_memberships(
        rows,
        config.data.leakage_audit.audit_seed,
        corpus_hash,
        "independent",
        strict_divisible=True,
    )
    assert list(strict_train) == list(train)
    assert list(strict_test) == list(test)
    assert strict_membership == membership
    # The dense compatibility path and bounded store have identical raw bytes,
    # row metadata and split coordinates, including a partial final batch.
    stored_path = tmp_path / "caller-owned-features.bin"
    stored = _prepare_public_shortcuts(
        examples,
        feature_path=stored_path,
        **{
            **kwargs,
            "config": config.data.leakage_audit.model_copy(update={"feature_batch_size": 33}),
        },
    )
    try:
        assert sha256_file(stored_path) == sha256_bytes(values.tobytes())
        assert stored[1] == rows and stored[4] == membership
        np.testing.assert_array_equal(stored[2], train)
        np.testing.assert_array_equal(stored[3], test)
    finally:
        stored[0]._mmap.close()
    clean = audit_public_shortcuts(examples, **kwargs)
    assert clean == tuple(
        _run_probes(values, rows, train, test, config.data.leakage_audit, profile, corpus_hash)
    )
    assert len(clean) == 27
    original_prepare = leakage._prepare_public_shortcuts
    original_control = leakage._run_positive_control_probe
    observed = {}

    def observe_owned_preparation(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        # Controls own a fresh diagnostic copy, separate from clean features.
        assert isinstance(prepared[0], np.memmap)
        assert not np.shares_memory(prepared[0], values)
        observed["buffer"] = weakref.ref(prepared[0])
        observed["path"] = Path(prepared[0].filename)
        return prepared

    def observe_injected_kernel(injected, *args, **kwargs):
        # Reopening the same owned store must not retain the old mapping.
        assert isinstance(injected, np.memmap)
        assert Path(injected.filename) == observed["path"]
        old = observed["buffer"]()
        assert old is None or old._mmap.closed
        return original_control(injected, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leakage, "_prepare_public_shortcuts", observe_owned_preparation)
        patch.setattr(leakage, "_run_positive_control_probe", observe_injected_kernel)
        controls = _audit_public_shortcut_controls(examples, **kwargs)
    assert not observed["path"].exists()
    expected_injected = values.copy()
    start, _ = leakage._feature_bounds(leakage.ShortcutFeatureGroup.COUNTS)
    expected_injected[:, start] = leakage._task_labels(rows, leakage.ShortcutTask.POSITIVE_BINARY)
    assert controls.clean_feature_sha256 == sha256_bytes(values.tobytes())
    assert controls.injected_feature_sha256 == sha256_bytes(expected_injected.tobytes())
    assert len(controls.shuffled_probes) == 27
    assert controls.control_id == "PILOT_PC_BINARY_COUNT_FEATURE"
    assert controls.injected_probe.balanced_accuracy >= 0.95
    assert controls.injected_probe.holm_adjusted_p < 0.01
    assert controls.positive_control_passed
    assert controls.split_membership_sha256 == membership
    assert controls.shuffled_probes == tuple(
        _run_probes(
            values,
            rows,
            train,
            test,
            config.data.leakage_audit,
            profile,
            corpus_hash,
            label_shuffled=True,
        )
    )
    injector = leakage.NamedLeakInjector(
        "PILOT_PC_BINARY_COUNT_FEATURE",
        leakage.ShortcutTask.POSITIVE_BINARY,
        "counts/positive_binary",
        lambda source: source,
    )
    dense_probe = original_control(
        expected_injected,
        rows,
        train,
        test,
        injector,
        leakage.ShortcutFeatureGroup.COUNTS,
        config.data.leakage_audit,
        profile,
        controls.injected_feature_sha256,
    )
    assert (
        controls.injected_probe == leakage._holm([dense_probe], config.data.leakage_audit.alpha)[0]
    )

    # Deliberate downstream failures must close maps and remove only the
    # wrapper's store, including after control injection reopens its map.
    original_probes = leakage._run_probes
    for operation, target in (
        (audit_public_shortcuts, "_run_probes"),
        (_audit_public_shortcut_controls, "_run_probes"),
        (_audit_public_shortcut_controls, "_run_positive_control_probe"),
    ):

        def fail_after_real_kernel(array, *args, target=target, **kwargs):
            observed["failed_map"] = weakref.ref(array)
            kernel = original_probes if target == "_run_probes" else original_control
            kernel(array, *args, **kwargs)
            raise RuntimeError("intentional downstream failure")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(leakage, "_prepare_public_shortcuts", observe_owned_preparation)
            patch.setattr(leakage, target, fail_after_real_kernel)
            with pytest.raises(RuntimeError, match="intentional downstream failure"):
                operation(examples, **kwargs)
        assert not observed["path"].exists()
        failed_map = observed["failed_map"]()
        assert failed_map is None or failed_map._mmap.closed
        assert stored_path.exists()

    # A failure during extraction still cleans the partially written store.
    original_write = leakage._write_feature_batch

    def fail_after_write(path, *args, **kwargs):
        observed["partial_path"] = path
        original_write(path, *args, **kwargs)
        raise RuntimeError("intentional extraction failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leakage, "_write_feature_batch", fail_after_write)
        with pytest.raises(RuntimeError, match="intentional extraction failure"):
            audit_public_shortcuts(examples, **kwargs)
    assert not observed["partial_path"].exists()
    assert stored_path.exists()
    assert tuple(episode_sha256(e.bundle) for e in examples) == digests

    # Reduced diagnostic statistics must not acquire full-profile authority by
    # editing report denominators, even when their internal confusion is valid.
    from silent_cascade.eval.pilot_audit import PilotAuditReport, audit_pilot_manifest
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_data import freeze_pilot_manifest

    resolved = resolve_pilot_config("phase4_smoke")
    manifest = freeze_pilot_manifest(
        resolved, stage="one_hop", output_path=tmp_path / "m.json", source_commit="a" * 40
    )
    report = audit_pilot_manifest(manifest, config=resolved.config, output_dir=tmp_path / "audit")
    forged = report.model_copy(
        update={
            "profile": "phase4-pilot-10000",
            "count": 10_000,
            "oracle_verified_count": 10_000,
            "parent_verified_count": 10_000,
            "feasible_trace_count": 10_000,
            "unique_public_id_count": 10_000,
            "projected_episode_sha256s": tuple(f"{i:064x}" for i in range(10_000)),
            "oracle_trace_sha256s": tuple(f"{i:064x}" for i in range(10_000)),
            "variant_counts": {
                "positive": 5000,
                "safe_negative": 2500,
                "disconnected_negative": 2500,
            },
            "path_counts": {"1": 10_000},
            "record_count_histogram": {"4": 10_000},
            "hazard_record_count": 20_000,
            "safe_record_count": 10_000,
            "statistical_status": "passed",
            "acceptance": True,
            "shortcut_probes": clean,
            "controls": controls,
        }
    )
    with pytest.raises(ValueError, match=r"profile|statistical|probe"):
        PilotAuditReport.model_validate_json(canonical_json_bytes(forged))


def test_audit_authenticates_hashes_before_statistical_work(tmp_path):
    from silent_cascade.eval.pilot_audit import audit_pilot_manifest
    from silent_cascade.train.pilot_data import freeze_pilot_manifest

    resolved = resolve_pilot_config("phase4_smoke")
    manifest = freeze_pilot_manifest(
        resolved, stage="one_hop", output_path=tmp_path / "m.json", source_commit="a" * 40
    )
    forged = manifest.model_copy(
        update={
            "entries": (
                manifest.entries[0].model_copy(update={"parent_hash": "f" * 64}),
                *manifest.entries[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="hash"):
        audit_pilot_manifest(forged, config=resolved.config, output_dir=tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
