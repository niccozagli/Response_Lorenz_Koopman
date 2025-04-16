"""
config.py

This file contains the settings class for the dynamical system
"""

from typing import Dict, Literal, Optional, Union
#from pydantic_settings import BaseSettings


class ModelSettings:#class DynamicalSettings(BaseSettings):
    rho : float = 28.0
    sigma : float = 10.0 
    beta : float = 8/3
    noise : float = 2

    tmin : float = 0.0
    tmax : float = 6000.0
    dt   : float = 0.01
    tau  : float = 10  # this saves the output every Delta_t = tau*dt
    transient : float = 100

    