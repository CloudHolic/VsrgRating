"""Composes the Hydra configs under the repository's `configs/` directory."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def compose_config(name: str, overrides: list[str]) -> DictConfig:
	"""The config `name` with Hydra-syntax overrides applied, interpolations left unresolved."""
	# hydra.main would also parse the command line, but it crashes on Python 3.14's argparse.
	with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
		return compose(config_name=name, overrides=overrides)
