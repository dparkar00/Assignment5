"""Vision Transformer on CIFAR-100: Swin (primary) vs. plain ViT (baseline).

Re-exports the package's public API so callers can do e.g.
`from src import SwinTransformer, VisionTransformer` instead of having to
know (or remember) which submodule each name lives in. Submodule imports
(`from src.models import SwinTransformer`) still work exactly as before --
this is purely additive.
"""

from src.augment import (
    MixupCutmixConfig,
    apply_mixup_cutmix,
    one_hot_with_smoothing,
    soft_cross_entropy,
)
from src.data import (
    Cifar100Datasets,
    DataConfig,
    build_data_config,
    build_dataloaders,
    build_datasets,
    build_eval_transform,
    build_train_transform,
)
from src.models import (
    SwinConfig,
    SwinTransformer,
    ViTConfig,
    VisionTransformer,
    count_parameters,
)
from src.train import EpochContext, build_model, train
from src.utils import get_device, load_yaml_config, set_seed

__all__ = [
    # augment
    "MixupCutmixConfig",
    "apply_mixup_cutmix",
    "one_hot_with_smoothing",
    "soft_cross_entropy",
    # data
    "Cifar100Datasets",
    "DataConfig",
    "build_dataloaders",
    "build_datasets",
    "build_eval_transform",
    "build_train_transform",
    # models
    "SwinConfig",
    "SwinTransformer",
    "ViTConfig",
    "VisionTransformer",
    "count_parameters",
    # train
    "EpochContext",
    "build_model",
    "train",
    # utils
    "get_device",
    "load_yaml_config",
    "set_seed",
]
