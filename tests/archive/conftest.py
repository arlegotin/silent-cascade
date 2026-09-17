import pytest


@pytest.fixture
def tiny_archive_policy():
    from silent_cascade.archive.types import ArchivePolicy

    return ArchivePolicy(
        workspace_bytes=40_000,
        spool_bytes=4_096,
        cache_bytes=4_096,
        pinned_bytes=1_024,
        metadata_bytes=16_384,
        scratch_bytes=1_024,
        logs_bytes=512,
        emergency_bytes=2_048,
        reserve_bytes=4_096,
        episode_bytes=2_048,
        pack_target_bytes=512,
        pack_episodes=4,
        journal_bytes=256,
        journal_records=4,
        chunk_bytes=64,
        page_bytes=1_024,
        page_entries=3,
        remote_bytes=100_000,
    )


@pytest.fixture
def archive_identity() -> dict[str, object]:
    return {
        "run_id": "debug-fixture",
        "source_commit": "0" * 40,
        "config_sha256": "1" * 64,
        "evidence_identity_sha256": "2" * 64,
        "checkpoint_sha256": None,
        "writer_stopped": True,
        "checkpoint_committed": False,
    }
