import numpy as np

def add_gausian_nosie(tdoa_values, sigma_ns, rng="None"):
    """
    Add gausian noise to time difference of arrival delay values.
    signa_ns: noise standard deviation in ns
    rng: optional rng for repoducibility
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma_sec = sigma_ns * 1e-9
    noise = rng.normal(0, sigma_sec, size=tdoa_values.shape)
    return tdoa_values + noise

    
    