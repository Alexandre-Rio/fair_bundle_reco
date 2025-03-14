import argparse
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from main import run


def create_args():
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem_config', type=str, default='ml-100k')
    parser.add_argument('--exp_name', type=str, default="exp_fair_wg")

    # Algo parameters
    parser.add_argument('--algo', type=str, default="fair_wg", help="'ilp', 'fair_wg', or 'f3r'")
    parser.add_argument('--gamma', type=float, default=1/3)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--lambda_pfair_update', action='store_true')
    parser.add_argument('--lambda_pfair_start', type=float, default=1.0)
    parser.add_argument('--lambda_pfair_update_every', type=int, default=5)
    parser.add_argument('--compute_optimal', action='store_true')

    return parser.parse_args()


# Parse args and load config
args = create_args()
problem_config = json.load(open(os.path.join("configs", args.problem_config + ".json")))
args.problem_config = problem_config

# Get task data
sim_matrix = pd.read_csv(os.path.join('data', args.problem_config["task"], 'sim_matrix.csv'), index_col=0)
disc_matrix = pd.read_csv(os.path.join('data', args.problem_config["task"], 'disc_matrix.csv'), index_col=0)
rel_matrix = np.load(os.path.join('data', args.problem_config["task"], f'rel_matrix.npy'))
if os.path.exists(os.path.join('data', args.problem_config["task"], 'user_prob_vector.csv')):
    probs_users = pd.read_csv(os.path.join('data', args.problem_config["task"], 'user_prob_vector.csv'), index_col=0)
else:
    probs_users = None
if os.path.exists(os.path.join('data', args.problem_config["task"], 'type_matrix.csv')):
    type_matrix = pd.read_csv(os.path.join('data', args.problem_config["task"], 'type_matrix.csv'), index_col=0)
else:
    type_matrix = None

# Pre-process data
rel_matrix = pd.DataFrame(rel_matrix)
sim_matrix.columns = sim_matrix.columns.astype('int')
disc_matrix.columns = disc_matrix.columns.astype('int')
rel_matrix = rel_matrix.loc[:, disc_matrix.index]
sim_matrix = sim_matrix.loc[disc_matrix.index, disc_matrix.index]
if type_matrix is not None:
    type_matrix = type_matrix.loc[disc_matrix.index]
rel_matrix /= 5.0

if __name__ == '__main__':
    lambda_list = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    data_x, data_y = [], []
    for lambda_pfair_start in lambda_list:
        setattr(args, "lambda_pfair_start", lambda_pfair_start)
        fairness_list, quality_list = [], []
        for seed in range(5):
            setattr(args, "seed", seed)
            bundle_df = run(args)

            fairness = bundle_df['fairness'].iloc[-1]
            quality = bundle_df['quality'].mean()

            fairness_list.append(fairness)
            quality_list.append(quality)

        data_x.append(np.mean(fairness_list))
        data_y.append(np.mean(quality_list))

    sc = plt.scatter(data_x, data_y, c=lambda_list, cmap=get_cmap('spring'))
    plt.title("Fairness-Quality Trade-off")
    plt.xlabel('Fairness')
    plt.ylabel('Quality')
    cbar = plt.colorbar(sc)
    cbar.set_label("lambda")
    plt.show()
