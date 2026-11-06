import hydra
from omegaconf import DictConfig

from engine.evaluate import bench, infer
from engine.fit import fit

MODES = {"fit": fit, "bench": bench, "infer": infer}


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {cfg.mode!r}")
    MODES[cfg.mode](cfg)


if __name__ == "__main__":
    main()
