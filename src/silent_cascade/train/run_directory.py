"""Create run directories through pinned no-follow descriptors."""

import os
import secrets
from contextlib import contextmanager, suppress
from pathlib import Path

from silent_cascade.train.state import TrainingError


@contextmanager
def _directory(path: Path, *, create: bool):
    absolute = path.absolute()
    if ".." in absolute.parts or not absolute.name:
        raise TrainingError("Invalid run directory path")
    descriptor = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except OSError as error:
        raise TrainingError("Run path must contain only real directories") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def prepare_run_directory(path: Path, *, resume: bool) -> Path:
    """Validate/create a real run directory without following symlinks."""
    with _directory(path, create=not resume) as descriptor:
        if not resume and os.listdir(descriptor):
            raise TrainingError("Existing run directory requires explicit resume")
    return path.absolute()


def create_attempt_directory(path: Path) -> Path:
    """Create one unique attempt through a validated directory descriptor."""
    with _directory(path, create=False) as descriptor:
        while True:
            name = "attempt-" + secrets.token_hex(16)
            try:
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
                return path.absolute() / name
            except FileExistsError:
                continue
