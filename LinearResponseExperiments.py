from LorenzEDMD.dynamical_system.Lorenz import lorenz63
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def get_observables(trajectory: np.ndarray):
    x, y , z = trajectory[:,0] , trajectory[:,1] , trajectory[:,2]
    observables = (x,y,z,x**2,y**2,z**2)
    return np.column_stack(observables) 

def rho_pert(point : np.ndarray):
    x , y , z = point
    return np.array( [0,x,0] )

def single_response(xi, eps, avg_obs, base_lorenz, seed=None):
    try:
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]
        ss = np.random.SeedSequence(seed)
        rng_p, rng_m = [np.random.default_rng(s) for s in ss.spawn(2)]

        lorenzResponse = lorenz63()
        lorenzResponse.noise = base_lorenz.noise
        lorenzResponse.t_span = base_lorenz.t_span
        lorenzResponse.dt = base_lorenz.dt
        lorenzResponse.tau = base_lorenz.tau
        lorenzResponse.transient = base_lorenz.transient

        lorenzResponse.y0 = xi + eps * rho_pert(xi)
        _, resp_p = lorenzResponse.integrate_EM(rng=rng_p, show_progress=False)

        lorenzResponse.y0 = xi - eps * rho_pert(xi)
        _, resp_m = lorenzResponse.integrate_EM(rng=rng_m, show_progress=False)

        obs_p = get_observables(resp_p)
        obs_m = get_observables(resp_m)

        return obs_p - avg_obs, obs_m - avg_obs
    except Exception as e:
        print(f"Error in single_response: {e}")
        return None



def main():
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
    lorenzResponse.t_span = (0,30)
    lorenzResponse.dt = 0.005
    lorenzResponse.tau = 1
    lorenzResponse.transient = 0

    # Perturbation amplitudes
    amplitudes = [0.2,0.4,0.8]

    # Performing the experiments
    RESP_P = []
    RESP_M = []

    for eps in amplitudes:
        resp_p , resp_m = 0, 0

        results = Parallel(n_jobs=4,batch_size=50)(
            delayed(single_response)(X[i,:], eps, avg_obs, lorenzResponse) 
            for i in tqdm(range(X.shape[0]))
        )
        
        resp_p_all = np.stack([r[0] for r in results], axis=0)
        resp_m_all = np.stack([r[1] for r in results], axis=0)

        resp_p = np.mean(resp_p_all, axis=0)  
        resp_m = np.mean(resp_m_all, axis=0)
        
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
        "Unperturbed Settings": lorenz,
    }
    with open("./data/response.pkl", "wb") as f:
        pickle.dump(dictionary, f)

if __name__ == "__main__":
    main()