"""PLM Registry: maps PLM names to extractors, compressors, and checkpoints."""

import dataclasses
import importlib
from functools import partial
from pathlib import Path

import torch

from src.compressors.channel_compressor import ChannelCompressor


@dataclasses.dataclass
class PLMConfig:
    """Configuration for a registered PLM."""

    name: str
    input_dim: int
    extractor_module: str
    extractor_fn: str
    extractor_kwargs: dict
    default_latent_dim: int
    checkpoint_pattern: str  # e.g. "channel_prot_t5_contrastive_d{dim}_s42"


class PLMRegistry:
    """Registry that maps PLM names to extraction functions and compressor checkpoints."""

    def __init__(
        self,
        checkpoint_base: Path | str = Path("data/checkpoints/channel"),
    ):
        self._configs: dict[str, PLMConfig] = {}
        self._checkpoint_base = Path(checkpoint_base)
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            PLMConfig(
                name="prot_t5_xl",
                input_dim=1024,
                extractor_module="src.extraction.prot_t5_extractor",
                extractor_fn="extract_prot_t5_embeddings",
                extractor_kwargs={},
                default_latent_dim=256,
                checkpoint_pattern="channel_prot_t5_contrastive_d{dim}_s42",
            )
        )
        self.register(
            PLMConfig(
                name="esm2_650m",
                input_dim=1280,
                extractor_module="src.extraction.esm_extractor",
                extractor_fn="extract_residue_embeddings",
                extractor_kwargs={"model_name": "esm2_t33_650M_UR50D"},
                default_latent_dim=256,
                checkpoint_pattern="channel_contrastive_d{dim}_s42",
            )
        )

    def register(self, config: PLMConfig) -> None:
        self._configs[config.name] = config

    def list_plms(self) -> list[str]:
        return list(self._configs.keys())

    def get_config(self, name: str) -> PLMConfig:
        if name not in self._configs:
            raise KeyError(
                f"Unknown PLM '{name}'. Available: {self.list_plms()}"
            )
        return self._configs[name]

    def get_compressor(
        self,
        name: str,
        latent_dim: int | None = None,
        device: torch.device | None = None,
    ) -> ChannelCompressor:
        """Load a trained ChannelCompressor for the given PLM."""
        config = self.get_config(name)
        dim = latent_dim or config.default_latent_dim

        model = ChannelCompressor(
            input_dim=config.input_dim,
            latent_dim=dim,
            dropout=0.1,
            use_residual=True,
        )

        ckpt_name = config.checkpoint_pattern.format(dim=dim)
        ckpt_path = self._checkpoint_base / ckpt_name / "best_model.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Train a compressor first or check checkpoint_base."
            )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        if device is not None:
            model = model.to(device)

        return model

    def get_extractor(self, name: str):
        """Import and return the extraction function for the given PLM."""
        config = self.get_config(name)
        mod = importlib.import_module(config.extractor_module)
        fn = getattr(mod, config.extractor_fn)
        if config.extractor_kwargs:
            return partial(fn, **config.extractor_kwargs)
        return fn
