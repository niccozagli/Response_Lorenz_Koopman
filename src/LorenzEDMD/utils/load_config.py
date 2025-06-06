"""
load_config.py

Holds the functions to load the settings specified in `config.py` and `.env`.
"""

from functools import lru_cache

from LorenzEDMD.config import EDMDSettings, ModelSettings


@lru_cache
def get_model_settings() -> ModelSettings:
    """
    Loads the settings for the Dynamical Systems.

    :return: Dynamical System settings object.
    """
    return ModelSettings()


@lru_cache
def get_edmd_settings() -> EDMDSettings:
    """
    Loads the settings for the Dynamical Systems.

    :return: Dynamical System settings object.
    """
    return EDMDSettings()
