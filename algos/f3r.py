import time
import numpy as np


def f3r(
        list_items,
        rel_vec,
        sim_matrix,
        type_matrix,
        disc_matrix,
        bundle_size,
        p_fair_targets_base,
        gamma,
        epsilon,
        lims_type=None,
):

        start = time.time()

        active = np.arange(len(list_items))
        actives = []
        for k in range(disc_matrix.shape[1]):
                actives.append(np.where(disc_matrix[:, k])[0])
        if type_matrix is not None:
                covered = np.zeros(type_matrix.shape[1])

        bundle = []
        current_rel = 0
        current_sim = 0
        N = 0

        while len(bundle) < bundle_size:

            # Reduce the search space by removing all covered items
            if type_matrix is not None:
                covered_items = active[np.argwhere((type_matrix[active] + covered > lims_type).any(axis=1)).squeeze()]
                active = active[~np.isin(active, covered_items)]
                for k in range(disc_matrix.shape[1]):
                        actives[k] = actives[k][~np.isin(actives[k], covered_items)]

            if len(active) == 0 or len(bundle) == bundle_size:
                break

            z = np.random.uniform(0, 1)

            if z > epsilon:
                k = np.random.choice(np.arange(disc_matrix.shape[1]), p=p_fair_targets_base)
                current_active = actives[k]
            else:
                current_active = active

            if len(current_active) == 0:
                current_active = active


            if N > 0:
                rels = (N / (N + 1)) * current_rel + (1 / N) * (1 - gamma) * rel_vec[current_active]
                if N > 1:
                        sims = (1 / N) * current_sim + (1 / (N * (N-1))) * gamma * sim_matrix[current_active][:, bundle].sum(axis=1)
                else:
                        sims = (1 / N) * current_sim + (1 / N) * gamma * sim_matrix[current_active][:, bundle].sum(axis=1)
                scores = rels + sims
            else:
                scores = (1 - gamma) * rel_vec[current_active]
            best_idx = scores.argmax()
            best_id = current_active[best_idx]

            bundle.append(best_id)
            if N > 0:
                current_rel = (N / (N + 1)) * current_rel + (1 / N) * (1 - gamma) * rel_vec[best_id]
                if N == 1:
                    current_sim = (1 / N) * current_sim + (1 / N) * gamma * sim_matrix[best_id][bundle].sum()
                else:
                    current_sim = (1 / N) * current_sim + (1 / (N * (N - 1))) * gamma * \
                                  sim_matrix[best_id][bundle].sum()
            else:
                current_rel = rel_vec[best_id]
            N += 1
            if type_matrix is not None:
                covered += type_matrix[best_id]

            best_idx_k = [j for j in range(disc_matrix.shape[1]) if best_id in actives[j]][0]
            actives[best_idx_k] = actives[best_idx_k][actives[best_idx_k] != best_id]
            active = active[active != best_id]

        bundle = [list_items[idx] for idx in bundle]
        duration = time.time() - start
        info = {"time": duration}

        return bundle, info
