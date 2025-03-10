import time
import numpy as np


def fair_wg(
            list_items,
            rel_vec,
            cost_vec,
            sim_matrix,
            type_matrix,
            disc_matrix,
            bundle_size,
            budget,
            disc_budgets,
            gamma,
            lambda_pfair=0,
            lims_type=None,
        ):

    start = time.time()

    pivot = rel_vec.argmax()

    active = np.arange(len(list_items))
    active = active[active != pivot]
    cost = cost_vec[pivot]
    if type_matrix is not None:
        covered = np.zeros(type_matrix.shape[1]) + type_matrix[pivot]
    finished = False

    bundle = [pivot]
    current_rel = (1 - gamma) * rel_vec[pivot]
    current_sim = 0
    N = 1

    while not finished:

        # Reduce the search space by removing all covered items
        if type_matrix is not None:
            covered_items = active[np.argwhere((type_matrix[active] + covered > lims_type).any(axis=1)).squeeze()]
            active = active[~np.isin(active, covered_items)]

        if len(active) == 0 or len(bundle) == bundle_size:
            break

        rels = (N / (N + 1)) * current_rel + (1 / N) * (1 - gamma) * rel_vec[active]
        if N == 1:
            sims = (1 / N) * current_sim + (1 / N) * gamma * sim_matrix[active][:, bundle].sum(axis=1)
        else:
            sims = (1 / N) * current_sim + (1 / (N * (N - 1))) * gamma * sim_matrix[active][:, bundle].sum(axis=1)
        scores = rels + sims - lambda_pfair * disc_budgets[disc_matrix[active].argmax(axis=1)]
        best_idx = scores.argmax()
        best_id = active[best_idx]

        # Test feasibility
        if budget > 0:
            feasibility_clause = (cost + cost_vec[best_id] <= budget)
        else:
            feasibility_clause = True
        if feasibility_clause and len(bundle) + 1 <= bundle_size:
            bundle.append(best_id)
            cost += cost_vec[best_id]
            # I realize it is not important to maintain the quantities below
            current_rel = (N / (N + 1)) * current_rel + (1 / N) * (1 - gamma) * rel_vec[best_id]
            if N == 1:
                current_sim = (1 / N) * current_sim + (1 / N) * gamma * sim_matrix[best_id][bundle].sum()
            else:
                current_sim = (1 / N) * current_sim + (1 / (N * (N - 1))) * gamma * sim_matrix[best_id][bundle].sum()
            N += 1
            if type_matrix is not None:
                covered += type_matrix[best_id]
        else:
            break

        active = active[active != best_id]

    bundle = [list_items[idx] for idx in bundle]
    duration = time.time() - start
    info = {"time": duration}

    return bundle, info




