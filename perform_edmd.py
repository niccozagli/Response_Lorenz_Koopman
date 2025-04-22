from LorenzEDMD.dynamical_system.Lorenz import lorenz63
from LorenzEDMD.EDMD.EDMD import EDMD_CHEB
from LorenzEDMD.utils.data_processing import normalise_data_chebyshev 
from LorenzEDMD.utils.load_config import get_edmd_settings
import pickle

######### CHOOSE THE PARAMETERS FOR EDMD ##########
degrees = [11,12,13,14,15]
flight_times = [1]

# Integrate the Lorenz system
lorenz = lorenz63()
t , X = lorenz.integrate_EM()
# Scale the data
scaled_data, data_min, data_max = normalise_data_chebyshev(X)

list_degree = []
for degree in degrees:
    list_ftime = []
    for f_time in flight_times:
        EDMD_SETTINGS = get_edmd_settings()
        EDMD_SETTINGS.degree = degree
        EDMD_SETTINGS.flight_time = f_time
        edmd = EDMD_CHEB(EDMD_SETTINGS)
        K = edmd.perform_edmd_chebyshev(scaled_data)
        list_ftime.append(edmd)
    list_degree.append(list_ftime)


with open("./data/edmd_objects.pkl", "wb") as f:
    pickle.dump(list_degree, f)

lorenz.trajectory = None
with open("./data/lorenz_low.pkl","wb") as f:
    pickle.dump(lorenz,f)