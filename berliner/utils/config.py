# berliner/utils/config.py
from pathlib import Path
import yaml

# this file lives at .../the_berliner_search/berliner/utils/config.py
# parents[0] = utils, [1] = berliner, [2] = project root
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

def load_config(path: str | Path | None = None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
