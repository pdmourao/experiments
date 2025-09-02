import numpy as np
from time import time as ttime
from tqdm import tqdm
from laboratory.systems import TAM as tam
# from systemsCopy import TAM as tam

def sanity_check(*args, checker = None, idx = None):
    if checker is not None:
        for idx_result, result in enumerate(checker):
            assert np.array_equal(args[idx_result][idx], result[idx_result][idx]), f'Check {idx_result} failed.'


def splitting_optimal(entropy, neurons, layers, k, m, r_values, beta_values, lmb_values, max_it, error, av_counter, h_norm, dynamic, checker = None, disable = True):

    t = ttime()
    len_r = len(r_values)

    rng_seeds = np.random.SeedSequence(entropy).spawn(len_r)

    mattis = np.zeros((len_r, 2, 3, 3))
    mattis_ex = np.zeros((len_r, 2, 3, 3))
    max_its = np.zeros((len_r, 2), dtype=int)

    with tqdm(total=len_r, disable=disable) as pbar:
        for idx_r, r_v in enumerate(r_values):

            system = tam(rng_ss = rng_seeds[idx_r], neurons = neurons, layers = layers, lmb = lmb_values[0, idx_r], r = r_v, m = m, supervised = True,
                         split = False)

            system.add_patterns(k)
            system.initial_state = system.mix()
            system.external_field = system.mix(0)

            mattis[idx_r, 0], mattis_ex[idx_r, 0], max_its[idx_r, 0] = system.simulate(beta=beta_values[0, idx_r], dynamic=dynamic, av_counter = av_counter, h_norm = h_norm, max_it = max_it, error = error)
            system.set_interaction(lmb = lmb_values[1, idx_r], split = True)
            mattis[idx_r, 1], mattis_ex[idx_r, 1], max_its[idx_r, 1] = system.simulate(beta=beta_values[1, idx_r], dynamic=dynamic, av_counter = av_counter, h_norm = h_norm, max_it = max_it, error = error)

            sanity_check(mattis, mattis_ex, max_its, checker = checker, idx = idx_r)

            if not disable:
                pbar.update(1)

    t = ttime() - t
    print(f'System ran in {round(t / 60)} minutes.')

    return mattis, mattis_ex, max_its


def splitting_beta(entropy, beta_values, neurons, k, layers, supervised, r, lmb, m, max_it, error, av_counter, h_norm, dynamic, checker = None, disable = True):

    # this function runs the splitting experiment with only varying temperature
    # see the HopfieldMC class for more details

    len_beta = len(beta_values)

    mattis = np.zeros((len_beta, 2, 3, 3))
    mattis_ex = np.zeros((len_beta, 2, 3, 3))
    max_its = np.zeros((len_beta, 2), dtype=int)

    t = ttime()


    system = tam(rng_ss = np.random.SeedSequence(entropy = entropy), neurons = neurons, r = r, lmb = lmb, m = m, supervised = supervised, split = False, layers = layers)

    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)

    # run across the betas
    for idx_beta, beta in enumerate(tqdm(beta_values, disable = disable)):
        if idx_beta > 0:
            system.set_interaction(split = False)

        mattis[idx_beta, 0], mattis_ex[idx_beta, 0], max_its[idx_beta, 0] = system.simulate(beta=beta, av_counter = av_counter, max_it = max_it, error = error, h_norm = h_norm, dynamic = dynamic)
        system.set_interaction(split = True)
        mattis[idx_beta, 1], mattis_ex[idx_beta, 1], max_its[idx_beta, 1] = system.simulate(beta=beta, av_counter = av_counter, max_it = max_it, error = error, h_norm = h_norm, dynamic = dynamic)

        sanity_check(mattis, mattis_ex, max_its, checker = checker, idx = idx_beta)

    if not disable:
        print(f'Sample ran in {round(ttime() - t / 60)} minutes.')

    return mattis, mattis_ex, max_its


def disentanglement(neurons, layers, k, r, m, lmb, split, supervised, beta, h_norm, max_it, error, av_counter, dynamic, rng_ss, av = True):


    system = tam(rng_ss = rng_ss, neurons=neurons, layers = layers, r=r, m=m, lmb=lmb, split = split, supervised = supervised)

    t = ttime()
    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)

    return system.simulate(beta=beta, h_norm = h_norm, max_it=max_it, error=error, av_counter=av_counter, dynamic=dynamic, av = av)


def disentanglement_2d(entropy, y_values, y_arg, x_values, x_arg, m, supervised, disable=True, checker = None, **kwargs):

    len_y = len(y_values)
    len_x = len(x_values)

    mattis = np.zeros((len_x, len_y, 3, 3))
    if supervised:
        mattis_ex = np.zeros((len_x, len_y, 3, 3))
    else:
        mattis_ex = np.zeros((len_x, len_y, m, 3, 3))
    max_its = np.zeros((len_x, len_y), dtype=int)

    t0 = ttime()

    rng_seeds = np.random.SeedSequence(entropy).spawn(len_x * len_y)

    with tqdm(total=len_x * len_y, disable=disable) as pbar:
        for idx_x, x_v in enumerate(x_values):
            kwargs[x_arg] = x_v
            for idx_y, y_v in enumerate(y_values):
                kwargs[y_arg] = y_v

                mattis[idx_x, idx_y], mattis_ex[idx_x, idx_y], max_its[idx_x, idx_y] = disentanglement(rng_ss=rng_seeds[idx_x * len_y + idx_y], m = m, supervised = supervised, **kwargs)

                sanity_check(mattis, mattis_ex, max_its, checker = checker, idx = (idx_x, idx_y))
                if not disable:
                    pbar.update(1)

    print(f'System ran in {round((ttime()-t0 )/ 60)} minutes.')

    return mattis, mattis_ex, max_its


# Disentanglement experiments in terms of lambda and beta
def disentanglement_lmb_beta(entropy, neurons, layers, k, r, m, lmb, beta, dynamic, split, supervised, max_it, error, av_counter, h_norm,
                             disable=True, checker = None):

    # Get length of input arrays
    len_lmb = len(lmb)
    len_beta = len(beta)

    mattis = np.zeros((len_lmb, len_beta, 3, 3))

    if supervised:
        mattis_ex = np.zeros((len_lmb, len_beta, 3, 3))
    else: # no average across examples in unsupervised experiments
        mattis_ex = np.zeros((len_lmb, len_beta, m, 3, 3))
    max_its = np.zeros((len_lmb, len_beta))

    t = ttime()

    system = tam(rng_ss = np.random.SeedSequence(entropy = entropy), neurons=neurons, layers = layers, r=r, m=m, split = split, supervised = supervised)

    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)

    with tqdm(total=len_lmb * len_beta, disable=disable) as pbar:
        for idx_lmb, lmb_v in enumerate(lmb):
            matrix_J = system.insert_g(lmb_v)
            for idx_beta, beta_v in enumerate(beta):

                mattis[idx_lmb, idx_beta], mattis_ex[idx_lmb, idx_beta], max_its[idx_lmb, idx_beta] = system.simulate(beta = beta_v, max_it = max_it, dynamic = dynamic, error = error, av_counter = av_counter, h_norm = h_norm, sim_J = matrix_J)
                sanity_check(mattis, mattis_ex, max_its, checker = checker, idx = (idx_lmb, idx_beta))

                if not disable:
                    pbar.update(1)

    t = ttime() - t
    print(f'System ran in {round(t / 60)} minutes.')

    return mattis, mattis_ex, max_its