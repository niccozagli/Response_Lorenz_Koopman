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
    tmax : float = 10**5 #6000.0
    dt   : float = 0.005 # John uses 0.001
    tau  : float = 20  # this saves the output every Delta_t = tau*dt
    transient : float = 500

class EDMDSettings:
    flight_time : int = 1 # John uses 100, 10^6 datapoints 
    degree : int = 7