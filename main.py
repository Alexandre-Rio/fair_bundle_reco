import argparse
import os
import numpy as np
import pandas as pd
import time
import json

from algos import ilp, f3r, fair_wg
from utils import score_bundle


def create_args():
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem_config', type=str, default='ml-100k')
    parser.add_argument('--exp_name', type=str, default="")

    # Algo parameters
    parser.add_argument('--algo', type=str, default="ilp", help="'ilp', 'fair_wg', or 'f3r'")
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


def run(args):
    run_name = os.path.join(args.problem_config["task"], args.algo, args.exp_name, f'run__ilp_{time.time()}__seed_{args.seed}')
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Get disc matrix, initialize disc budgets and compute p_fair_targets
    K = disc_matrix.shape[1]
    disc_budgets = np.zeros(K)
    n_rec_items = 0
    n_rec_items_disc = np.zeros(K)
    n_rec_items_opt = 0
    n_rec_items_disc_opt = np.zeros(K)

    # Set type limits
    if "lims_type" in problem_config:
        if isinstance(problem_config["lims_type"], int):
            lims_type = list(np.ones(type_matrix.shape[1]) * problem_config["lims_type"])
        elif isinstance(problem_config["lims_type"], list):
            lims_type = problem_config["lims_type"]
        else:
            raise ValueError("Wrong type for 'lims_type' in config.")
    else:
        lims_type = None

    # Set exposure vector and build adaptive p_fair targets if needed
    p_fair_targets_base = np.array(problem_config["exposure_vector"]).astype('float64')
    p_fair_targets_base /= p_fair_targets_base.sum()  # Normalize to one
    t_vec = np.arange(1, args.problem_config["horizon"] + 1)
    evo_target_func = lambda step: ((args.problem_config["horizon"] - step) / args.problem_config["horizon"]) ** args.alpha
    if args.alpha > 0:
        p_fair_mult = 1 - args.epsilon * (1 + evo_target_func(t_vec))
        p_fair_targets = np.tile(p_fair_targets_base, (args.problem_config["horizon"], 1)) * (1 - args.epsilon * (1 + np.tile(evo_target_func(t_vec), (disc_matrix.shape[1], 1)).T))
    else:
        p_fair_mult = np.ones_like(t_vec) * (1 - args.epsilon)
        p_fair_targets = np.tile(p_fair_targets_base, (args.problem_config["horizon"], 1)) * (1 - args.epsilon)

    # START RUNTIME
    np.random.seed(args.seed)
    lambda_pfair = args.lambda_pfair_start

    # For storage
    bundle_history = []
    opt_bundle_history = []

    for t in range(args.problem_config["horizon"]):

        if t % 10 == 0:
            print(f"Step {t + 1}/{args.problem_config['horizon']}")

        # Sample user
        if probs_users is not None:
            user_idx = np.random.choice(len(probs_users), p=probs_users.to_numpy().squeeze(-1))
        else:
            user_idx = np.random.choice(len(rel_matrix))
        rel_user = rel_matrix.loc[user_idx]
        user_pref_profile = [rel_user[disc_matrix[0] == 0].mean(), rel_user[disc_matrix[0] == 1].mean()]

        # Pre-select/filer items if specified
        ps_idx = rel_user.sort_values().index[-args.M:].to_numpy()

        # Update disc budget
        if t > 0:
            for k in range(K):
                disc_budgets[k] = n_rec_items_disc[k] - p_fair_targets[t, k] * n_rec_items

        # Recommend bundle
        type_matrix_ps = type_matrix.loc[ps_idx].to_numpy() if type_matrix is not None else None
        print(f"User: {user_idx}")
        if args.algo == "ilp":
            bundle, info = ilp(list_items=ps_idx,
                               rel_vec=rel_user.loc[ps_idx].to_numpy(),
                               sim_matrix=sim_matrix.loc[ps_idx, ps_idx].to_numpy(),
                               type_matrix=type_matrix_ps,
                               disc_matrix=disc_matrix.loc[ps_idx].to_numpy(),
                               disc_budgets=disc_budgets,
                               p_fair_targets=p_fair_targets[t],
                               p_fair=True,
                               gamma=args.gamma,
                               bundle_size=args.problem_config["bundle_size"],
                               lims_type=lims_type,
            )
        elif args.algo == "fair_wg":
            bundle, info = fair_wg(
                list_items=ps_idx,
                rel_vec=rel_user.loc[ps_idx].to_numpy(),
                sim_matrix=sim_matrix.loc[ps_idx, ps_idx].to_numpy(),
                type_matrix=type_matrix_ps,
                disc_matrix=disc_matrix.loc[ps_idx].to_numpy(),
                bundle_size=args.problem_config["bundle_size"],
                disc_budgets=disc_budgets / (n_rec_items + 1e-9),
                gamma=args.gamma,
                lambda_pfair=lambda_pfair,
                lims_type=lims_type,
            )
        elif args.algo == "f3r":
            bundle, info = f3r(
                               list_items=ps_idx,
                               rel_vec=rel_user.loc[ps_idx].to_numpy(),
                               sim_matrix=sim_matrix.loc[ps_idx, ps_idx].to_numpy(),
                               type_matrix=type_matrix_ps,
                               disc_matrix=disc_matrix.loc[ps_idx].to_numpy(),
                               bundle_size=args.problem_config["bundle_size"],
                               p_fair_targets_base=p_fair_targets_base,
                               gamma=args.gamma,
                               epsilon=args.epsilon,
                               lims_type=lims_type,
            )
        else:
            raise ValueError('Invalid algorithm name. Valid names are: "ilp", "fair_wg" and "f3r".')

        bundle_quality, bundle_relevance, bundle_similarity = score_bundle(bundle, rel_user, sim_matrix, args)

        # Update disc budgets
        n_rec_items += len(bundle)
        for k in range(K):
            n_rec_items_disc[k] += disc_matrix.loc[bundle, k].sum()
        fairness = 1 - np.max(
            np.maximum(0.0, (p_fair_targets_base - n_rec_items_disc / n_rec_items) / p_fair_targets_base))

        if args.compute_optimal:  # Compute bundle without fairness
            opt_bundle, opt_info = ilp(list_items=ps_idx,
                                       rel_vec=rel_user.loc[ps_idx].to_numpy(),
                                       sim_matrix=sim_matrix.loc[ps_idx, ps_idx].to_numpy(),
                                       type_matrix=type_matrix_ps,
                                       disc_matrix=disc_matrix.loc[ps_idx].to_numpy(),
                                       disc_budgets=disc_budgets,
                                       p_fair_targets=p_fair_targets[t],
                                       p_fair=False,
                                       gamma=args.gamma,
                                       bundle_size=args.problem_config["bundle_size"],
                                       lims_type=lims_type,
            )


            opt_quality, opt_relevance, opt_similarity = score_bundle(opt_bundle, rel_user, sim_matrix, args)

            n_rec_items_opt += len(opt_bundle)
            for k in range(K):
                n_rec_items_disc_opt[k] += disc_matrix.loc[opt_bundle, k].sum()
            fairness_opt = 1 - np.max(np.maximum(0.0, (p_fair_targets_base - n_rec_items_disc_opt / n_rec_items_opt) / p_fair_targets_base))
            relative_fairness = fairness / fairness_opt

            if opt_quality > 0 and opt_relevance > 0 and opt_similarity > 0 and len(opt_bundle) > 0:
                relative_quality = bundle_quality / opt_quality
                relative_relevance = bundle_relevance / opt_relevance
                relative_similarity = bundle_similarity / opt_similarity
                relative_size = len(bundle) / len(opt_bundle)
            else:
                relative_quality = 0
                relative_relevance = 0
                relative_similarity = 0
                relative_size = 0

            opt_bundle_cat = list(disc_matrix.loc[opt_bundle].idxmax(axis=1).values)
            opt_bundle_history.append((user_idx, opt_bundle, opt_info, opt_quality, opt_relevance, opt_similarity, opt_bundle_cat, fairness_opt))

        # Update lambda_p_fair if needed
        if args.algo == "fair_wg":
            if args.lambda_pfair_update and (t + 1) % args.lambda_pfair_update_every == 0:
                if fairness < 1 - args.epsilon:
                    lambda_pfair *= 2
                elif fairness > 1 - args.epsilon / 2:
                    lambda_pfair /= 2

        # Log Results
        bundle_cat = list(disc_matrix.loc[bundle].idxmax(axis=1).values)
        if args.compute_optimal:
            bundle_history.append(
                (user_idx, bundle, info, bundle_quality, bundle_relevance, bundle_similarity, bundle_cat,
                 relative_quality, relative_relevance, relative_similarity, p_fair_mult[t], fairness, relative_fairness,
                 len(bundle), len(opt_bundle), relative_size, user_pref_profile, lambda_pfair))
        else:
            bundle_history.append(
                (user_idx, bundle, info, bundle_quality, bundle_relevance, bundle_similarity, bundle_cat,
                 p_fair_mult[t], fairness, len(bundle), lambda_pfair))

        writer.add_scalar("charts/bundle_quality", bundle_quality, t)
        writer.add_scalar("charts/bundle_relevance", bundle_relevance, t)
        writer.add_scalar("charts/bundle_similarity", bundle_similarity, t)
        writer.add_scalar("charts/fairness", fairness, t)
        writer.add_scalar("charts/lambda_pfair", lambda_pfair, t)
        writer.add_scalar("charts/n_rec_items", n_rec_items, t)
        writer.add_scalar("charts/size", len(bundle), t)
        for k in range(K):
            if n_rec_items > 0:
                writer.add_scalar(f"charts/exp_{k}", n_rec_items_disc[k] / n_rec_items, t)
            writer.add_scalar(f"charts/disc_budgets_{k}", disc_budgets[k] / n_rec_items, t)
            writer.add_scalar(f"charts/p_fair_target_{k}", p_fair_targets[t, k], t)
        if args.compute_optimal:
            writer.add_scalar("charts/relative_relevance", relative_relevance, t)
            writer.add_scalar("charts/relative_size", relative_size, t)
            writer.add_scalar("charts/opt_size", len(opt_bundle), t)
            writer.add_scalar("charts_opt/fairness_opt", fairness_opt, t)
            writer.add_scalar("charts_opt/opt_quality", opt_quality, t)
            writer.add_scalar("charts_opt/opt_relevance", opt_relevance, t)
            writer.add_scalar("charts_opt/opt_similarity", opt_similarity, t)
            writer.add_scalar("charts_opt/n_rec_items_opt", n_rec_items_opt)
            for k in range(K):
                if n_rec_items > 0:
                    writer.add_scalar(f"charts_opt/exp_{k}", n_rec_items_disc_opt[k] / n_rec_items_opt, t)
        writer.add_scalar(f"charts/time", info["time"], t)

    # Store results
    bundle_df = pd.DataFrame(bundle_history)
    if args.compute_optimal:
        bundle_df.columns = ['user', 'bundle', 'info', 'quality', 'relevance', 'similarity', 'categories',
                             'relative_quality', 'relative_relevance', 'relative_similarity', 'fairness_bound',
                             'fairness', 'relative_fairness', 'size', 'opt_size', 'relative_size', 'user_pref_profile', 'lambda']
    else:
        bundle_df.columns = ['user', 'bundle', 'info', 'quality', 'relevance', 'similarity', 'categories', 'fairness_bound', 'fairness', 'size', 'lambda']
    bundle_df.to_csv(f'./runs/{run_name}/bundle_history.csv')
    if args.compute_optimal:
        opt_df = pd.DataFrame(opt_bundle_history)
        opt_df.columns = ['user', 'bundle', 'info', 'quality', 'relevance', 'similarity', 'categories', 'fairness']
        opt_df.to_csv(f'./runs/{run_name}/opt_history.csv')

    # Log args
    with open(os.path.join('runs', run_name, 'args.txt'), 'w') as f:
        f.write(json.dumps(vars(args)))

    writer.close()

    return bundle_df


if __name__ == '__main__':
    run(args)



