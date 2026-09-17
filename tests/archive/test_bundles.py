from pathlib import Path

import pytest


def _preserve_chunks(chunks, destination: Path) -> tuple[Path, ...]:
    destination.mkdir()
    result = []
    for index, source in enumerate(chunks):
        target = destination / f"chunk-{index:05d}"
        target.write_bytes(source.read_bytes())
        result.append(target)
    return tuple(result)


def _seal(
    tmp_path,
    policy,
    identity,
    files,
    *,
    kind="partial",
    logical_root="unit",
    episode_groups=(),
    borrowed=(),
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    for name, payload in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    ref = seal_unit(
        run_dir=run,
        control_dir=tmp_path / "control",
        logical_root=logical_root,
        paths=tuple(sorted(files)),
        kind=kind,
        identity=identity,
        policy=policy,
        episode_groups=episode_groups,
        borrowed=borrowed,
    )
    return run, ref


def _chunks(tmp_path, run, ref, policy):
    from silent_cascade.archive.bundles import iter_unit_chunks

    return _preserve_chunks(
        iter_unit_chunks(
            run_dir=run,
            control_dir=tmp_path / "control",
            ref=ref,
            scratch_dir=tmp_path / "scratch",
            policy=policy,
        ),
        tmp_path / "saved-chunks",
    )


def test_streaming_chunks_round_trip_split_files_and_empty_member(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.bundles import restore_unit

    policy = tiny_archive_policy.model_copy(update={"chunk_bytes": 4})
    files = {"unit/a": b"abcdef", "unit/empty": b"", "unit/z": b"12345"}
    run, ref = _seal(tmp_path, policy, archive_identity, files)
    chunks = _chunks(tmp_path, run, ref, policy)
    assert [path.read_bytes() for path in chunks] == [b"abcd", b"ef12", b"345"]
    assert all(path.stat().st_size <= policy.chunk_bytes for path in chunks)

    destination = tmp_path / "restored"
    assert (
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=iter(chunks),
            destination=destination,
            policy=policy,
        )
        == destination
    )
    assert {name: (destination / name).read_bytes() for name in files} == files
    assert (destination / "unit/empty").is_file()


def test_chunk_rebuild_rejects_source_change_and_symlink_parent_swap(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.bundles import iter_unit_chunks

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"original"})
    (run / "unit/a").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="changed"):
        tuple(
            iter_unit_chunks(
                run_dir=run,
                control_dir=tmp_path / "control",
                ref=ref,
                scratch_dir=tmp_path / "scratch-a",
                policy=tiny_archive_policy,
            )
        )

    other = tmp_path / "other"
    other.mkdir()
    (other / "a").write_bytes(b"original")
    moved = run / "real-unit"
    (run / "unit").rename(moved)
    (run / "unit").symlink_to(other, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        tuple(
            iter_unit_chunks(
                run_dir=run,
                control_dir=tmp_path / "control",
                ref=ref,
                scratch_dir=tmp_path / "scratch-b",
                policy=tiny_archive_policy,
            )
        )


@pytest.mark.parametrize("fault", ["missing", "corrupt", "extra"])
def test_restore_authenticates_complete_chunk_stream_before_publication(
    tmp_path, tiny_archive_policy, archive_identity, fault
):
    from silent_cascade.archive.bundles import restore_unit

    policy = tiny_archive_policy.model_copy(update={"chunk_bytes": 4})
    run, ref = _seal(tmp_path, policy, archive_identity, {"unit/a": b"abcdefgh"})
    chunks = list(_chunks(tmp_path, run, ref, policy))
    if fault == "missing":
        chunks.pop()
    elif fault == "corrupt":
        chunks[-1].write_bytes(b"xxxx")
    else:
        extra = tmp_path / "extra"
        extra.write_bytes(b"x")
        chunks.append(extra)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="chunk"):
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=destination,
            policy=policy,
        )
    assert not destination.exists()


def test_restore_crash_before_atomic_publication_leaves_destination_absent(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    from silent_cascade.archive import bundles

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"abc"})
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)

    def fail_publish(*_args, **_kwargs):
        raise OSError("injected publication crash")

    monkeypatch.setattr(bundles, "_rename_directory_noreplace", fail_publish)
    destination = tmp_path / "restored"
    with pytest.raises(OSError, match="publication crash"):
        bundles.restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=destination,
            policy=tiny_archive_policy,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".restored.*.restore"))


def test_sparse_episode_restore_requires_one_complete_authenticated_owned_set(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.bundles import restore_unit

    policy = tiny_archive_policy.model_copy(update={"chunk_bytes": 4})
    files = {
        "pack/episodes/00000.neural.json": b"a",
        "pack/episodes/00000.trajectory.json.gz": b"1234",
        "pack/episodes/00001.neural.json": b"b",
        "pack/episodes/00001.trajectory.json.gz": b"5678",
        "pack/crashes/crash-00000.json": b"crash",
    }
    episode_a = (
        "pack/crashes/crash-00000.json",
        "pack/episodes/00000.neural.json",
        "pack/episodes/00000.trajectory.json.gz",
    )
    episode_b = (
        "pack/episodes/00001.neural.json",
        "pack/episodes/00001.trajectory.json.gz",
    )
    run, ref = _seal(
        tmp_path,
        policy,
        archive_identity,
        files,
        kind="episode_pack",
        logical_root="pack",
        episode_groups=(episode_a, episode_b),
    )
    chunks = _chunks(tmp_path, run, ref, policy)
    destination = tmp_path / "episode-a"
    restore_unit(
        control_dir=tmp_path / "control",
        ref=ref,
        chunks=chunks[:3],
        destination=destination,
        policy=policy,
        selected_paths=episode_a,
    )
    assert (destination / "pack/episodes/00000.neural.json").read_bytes() == b"a"
    assert (destination / "pack/crashes/crash-00000.json").read_bytes() == b"crash"
    assert not (destination / "pack/episodes/00001.neural.json").exists()

    with pytest.raises(ValueError, match="episode"):
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=tmp_path / "partial",
            policy=policy,
            selected_paths=("pack/episodes/00000.neural.json",),
        )

    corrupt = list(chunks[:3])
    corrupt[-1].write_bytes(b"xxxx")
    with pytest.raises(ValueError, match="chunk"):
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=corrupt,
            destination=tmp_path / "corrupt",
            policy=policy,
            selected_paths=episode_a,
        )

    from silent_cascade.archive.types import UnitManifest

    manifest = UnitManifest.model_validate_json(
        (tmp_path / "control" / ref.manifest_path).read_bytes()
    )
    unit_dir = (tmp_path / "control" / ref.manifest_path).parent
    (unit_dir / manifest.inventory_shards[-1].path).unlink()
    with pytest.raises(ValueError, match="inventory"):
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks[:3],
            destination=tmp_path / "missing-inventory-tail",
            policy=policy,
            selected_paths=episode_a,
        )


def test_sparse_diagnostic_restore_is_explicit_and_bounded(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.bundles import restore_unit

    files = {"diagnostic/a": b"a", "diagnostic/b": b"b"}
    run, ref = _seal(
        tmp_path,
        tiny_archive_policy,
        archive_identity,
        files,
        kind="diagnostic",
        logical_root="diagnostic",
    )
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    destination = tmp_path / "selected"
    restore_unit(
        control_dir=tmp_path / "control",
        ref=ref,
        chunks=chunks,
        destination=destination,
        policy=tiny_archive_policy,
        selected_paths=("diagnostic/b",),
    )
    assert (destination / "diagnostic/b").read_bytes() == b"b"
    assert not (destination / "diagnostic/a").exists()


def test_restore_refuses_existing_destination(tmp_path, tiny_archive_policy, archive_identity):
    from silent_cascade.archive.bundles import restore_unit

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"a"})
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    destination = tmp_path / "restored"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=destination,
            policy=tiny_archive_policy,
        )


def test_restore_never_materializes_borrowed_shared_inputs(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.bundles import restore_unit
    from silent_cascade.archive.types import FileEntry
    from silent_cascade.hashing import sha256_bytes

    files = {"pack/episode.json": b"episode", "weights/shared": b"weights"}
    shared = files.pop("weights/shared")
    run = tmp_path / "run"
    shared_path = run / "weights/shared"
    shared_path.parent.mkdir(parents=True)
    shared_path.write_bytes(shared)
    run, ref = _seal(
        tmp_path,
        tiny_archive_policy,
        archive_identity,
        files,
        kind="episode_pack",
        logical_root="pack",
        episode_groups=(("pack/episode.json",),),
        borrowed=(
            FileEntry(path="weights/shared", sha256=sha256_bytes(shared), bytes=len(shared)),
        ),
    )
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    destination = tmp_path / "restored"
    restore_unit(
        control_dir=tmp_path / "control",
        ref=ref,
        chunks=chunks,
        destination=destination,
        policy=tiny_archive_policy,
    )
    assert (destination / "pack/episode.json").read_bytes() == b"episode"
    assert not (destination / "weights").exists()


def test_restore_rejects_destination_parent_swap_before_publication(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    from silent_cascade.archive import bundles

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"a"})
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    parent = tmp_path / "destination-parent"
    parent.mkdir()
    real_parent = tmp_path / "real-destination-parent"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    original_restore = bundles._restore_records

    def swap_after_restore(**arguments):
        original_restore(**arguments)
        parent.rename(real_parent)
        parent.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(bundles, "_restore_records", swap_after_restore)
    with pytest.raises((OSError, ValueError)):
        bundles.restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=parent / "restored",
            policy=tiny_archive_policy,
        )
    assert not (attacker / "restored").exists()
    assert not (real_parent / "restored").exists()


def test_restore_does_not_replace_destination_created_during_staging(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    from silent_cascade.archive import bundles

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"a"})
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    destination = tmp_path / "restored"
    original_restore = bundles._restore_records

    def create_destination_after_restore(**arguments):
        original_restore(**arguments)
        destination.mkdir()

    monkeypatch.setattr(bundles, "_restore_records", create_destination_after_restore)
    with pytest.raises(FileExistsError):
        bundles.restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=destination,
            policy=tiny_archive_policy,
        )
    assert destination.is_dir()
    assert not (destination / "unit").exists()


def test_restore_atomically_refuses_destination_created_after_final_check(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    from silent_cascade.archive import bundles

    run, ref = _seal(tmp_path, tiny_archive_policy, archive_identity, {"unit/a": b"a"})
    chunks = _chunks(tmp_path, run, ref, tiny_archive_policy)
    destination = tmp_path / "restored"
    original_stat = bundles.os.stat
    missing_checks = 0

    def create_destination_after_final_check(path, *args, **kwargs):
        nonlocal missing_checks
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if path == destination.name:
                missing_checks += 1
                if missing_checks == 2:
                    bundles.os.mkdir(path, dir_fd=kwargs["dir_fd"])
            raise

    monkeypatch.setattr(bundles.os, "stat", create_destination_after_final_check)
    with pytest.raises(FileExistsError):
        bundles.restore_unit(
            control_dir=tmp_path / "control",
            ref=ref,
            chunks=chunks,
            destination=destination,
            policy=tiny_archive_policy,
        )
    assert destination.is_dir()
    assert not (destination / "unit").exists()
