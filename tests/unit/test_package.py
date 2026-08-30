from importlib.metadata import version

from silent_cascade import __version__


def test_public_version_matches_distribution_metadata() -> None:
    assert __version__ == version("silent-cascade") == "0.1.0"
