def score_bundle(bundle, rel_vec, sim_matrix, args):
    """
    Compute bundle quality, relevance and similarity.
    :param bundle: an array with the bundle's item ids.
    :param rel_vec: the user's relevance scores.
    :param sim_matrix: the task similarity matrix.
    :param args: the script arguments.
    :return: Bundle quality, relevance and similarity.
    """
    if len(bundle) == 0:
        return 0, 0, 0

    else:
        relevance = rel_vec.loc[bundle].sum() / len(bundle)
        similarity = 0
        for i in range(len(bundle) - 1):
            sim = sim_matrix.loc[bundle[i], bundle[i+1:]]
            similarity += sim.sum()
        if len(bundle) > 1:
            similarity /= len(bundle) * (len(bundle) - 1)

        quality = (1 - args.gamma) * relevance + args.gamma * similarity

        return quality, relevance, similarity