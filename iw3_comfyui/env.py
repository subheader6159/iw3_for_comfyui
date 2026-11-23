import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUNIF_ROOT = os.path.join(PACKAGE_ROOT, "nunif")


def setup_nunif(models_dir: str | None = None) -> None:
    """Make the bundled nunif/iw3 importable. Must run before importing `iw3` or `nunif`."""
    if not os.path.isfile(os.path.join(NUNIF_ROOT, "iw3", "utils.py")):
        raise RuntimeError(
            f"nunif submodule not found at {NUNIF_ROOT}. "
            f"Run `git submodule update --init` inside {PACKAGE_ROOT}, "
            "or reinstall the node with `git clone --recursive`."
        )
    # iw3 stores weights under $NUNIF_HOME/iw3/pretrained_models. Respect a user-defined NUNIF_HOME.
    if models_dir and not os.environ.get("NUNIF_HOME"):
        os.environ["NUNIF_HOME"] = models_dir
    if NUNIF_ROOT not in sys.path:
        # Appended so the nunif repository root never shadows other top-level modules.
        sys.path.append(NUNIF_ROOT)
