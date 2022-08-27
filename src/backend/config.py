# -*- coding: utf-8 -*-

##################################################
## Configurations for FastAPI and transfomers Bloom model
##################################################
__author__ = "Raymond Ng"
__license__ = "MIT"
__maintainer__ = "Raymond Ng"
__version__ = "0.1.0"
__status__ = "Dev"
##################################################


from typing import Dict, Union

import pydantic
import torch


class Settings(pydantic.BaseSettings):
    """Settings class for API and Bloom model"""

    # API Config
    API_NAME: str = "bloom_api"
    API_STR: str = "/api/v1"
    LOGGER_CONFIG_PATH: str = "conf/logging.yml"

    # Bloom Model Config
    CHECKPOINT_NAME: str = "bigscience/bloom"
    CACHE_DIR: Union[str, None] = None
    DEVICE_MAP: Union[str, None] = "auto"
    LOAD_IN_8BIT: bool = True
    INT8_THRESHOLD: float = 6.0
    DTYPE: Union[torch.dtype, None] = torch.bfloat16
    MAX_MEMORY: Union[Dict[int, str], None] = {
        0: "0GIB",
        1: "51GIB",
        2: "51GIB",
        3: "51GIB",
        4: "51GIB",
        5: "51GIB",
        6: "51GIB",
        7: "51GIB",
    }

    @classmethod
    def for_cpu(cls, checkpoint_name: str = "bigscience/bloom", cache_dir: Union[str, None] = None):
        settings = Settings()
        settings.CHECKPOINT_NAME = checkpoint_name
        settings.CACHE_DIR = cache_dir
        settings.DEVICE_MAP = None
        settings.LOAD_IN_8BIT = False
        return settings

    @classmethod
    def for_16x_gpus(
        cls,
        checkpoint_name: str = "bigscience/bloom",
        cache_dir: Union[str, None] = None,
        max_memory: Dict[int, str] = {
            0: "0GIB",
            1: "25GIB",
            2: "25GIB",
            3: "25GIB",
            4: "25GIB",
            5: "25GIB",
            6: "25GIB",
            7: "25GIB",
            8: "25GIB",
            9: "25GIB",
            10: "25GIB",
            11: "25GIB",
            12: "25GIB",
            13: "25GIB",
            14: "25GIB",
            15: "25GIB",
        },
    ):
        settings = Settings()
        settings.CHECKPOINT_NAME = checkpoint_name
        settings.CACHE_DIR = cache_dir
        settings.MAX_MEMORY = max_memory
        return settings
