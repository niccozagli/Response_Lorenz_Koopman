from LorenzEDMD.dynamical_system.Lorenz import lorenz63
import pickle
import numpy as np
from tqdm import tqdm

def get_observables(trajectory: np.ndarray):
    x, y , z = trajectory[:,0] , trajectory[:,1] , trajectory[:,2]
    observables = (x,y,z,x**2,y**2,z**2)
    return np.column_stack(observables) 

def rho_pert(point : np.ndarray):
    x , y , z = point
    return np.array( [0,x,0] )


# Unperturbed system
lorenz = lorenz63()
lorenz.noise = 2
lorenz.t_span = (0,10**5/2)
lorenz.dt = 0.005
lorenz.tau = 100
t , X = lorenz.integrate_EM()
avg_obs = get_observables(X).mean(axis=0)

# Perturbation experiments settings
lorenzResponse = lorenz63()
lorenzResponse.noise = lorenz.noise
lorenzResponse.t_span = (0,50)
lorenzResponse.dt = 0.005
lorenzResponse.tau = 1
lorenzResponse.transient = 0

# Perturbation amplitudes
amplitudes = [0.2,0.4,0.8]

# Performing the experiments
RESP_P = []
RESP_M = []

for eps in amplitudes:
    resp_p , resp_m = 0 , 0
    for i in tqdm(range( X.shape[0])) :
        # Positive response experiment
        lorenz63.y0 = X[i,:] + eps * rho_pert(X[i,:])
        t , resp = lorenzResponse.integrate_EM(show_progress=False)
        obs = get_observables(resp)
        resp_p += (obs - avg_obs) / X.shape[0]

        # Negative response experiment
        lorenz63.y0 = X[i,:] - eps * rho_pert(X[i,:])
        t , resp = lorenzResponse.integrate_EM(show_progress=False)
        obs = get_observables(resp)
        resp_m += (obs - avg_obs) / X.shape[0]

    RESP_P.append(resp_p)
    RESP_M.append(resp_m)

# Saving the data
lorenz.trajectory = None
lorenzResponse.trajectory = None
dictionary = {
    "Positive Response": RESP_P,
    "Negative Response": RESP_M,
    "Amplitudes" : amplitudes,
    "Response Settings": lorenzResponse,
    "Unperturbed Settings": lorenz
}
with open("./data/response.pkl", "wb") as f:
    pickle.dump(dictionary, f)