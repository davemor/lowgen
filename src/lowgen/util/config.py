from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SeedConfig:
    base_seed: int = 0
    num_seeds: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedConfig":
        return cls(**data)


@dataclass
class Config:
    datasets: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    num_folds: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data)

        if "seeds" in data:
            data["seeds"] = SeedConfig.from_dict(data["seeds"])

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
