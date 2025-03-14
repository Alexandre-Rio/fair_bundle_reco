import gurobipy as gp
from gurobipy import GRB
import numpy as np
from time import time


def ilp(list_items,
        rel_vec,
        sim_matrix,
        type_matrix,
        disc_matrix,
        disc_budgets,
        p_fair_targets,
        p_fair,
        gamma,
        bundle_size=-1,
        lims_type=None
):

    ################################### ILP MODEL ########################################

    start_model = time()
    model = gp.Model()
    model.setParam('OutputFlag', 0)  # No print during optimization

    # Set decision variables
    x = model.addVars(
        range(len(list_items)),
        vtype=GRB.BINARY,
        name="x"
    )

    y = model.addVars(
        [(i, j) for i in range(len(list_items)) for j in range(i + 1, len(list_items))],
        vtype=GRB.BINARY,
        name="y"
    )

    # Set constraints

    ## Complementarity
    if type_matrix is not None:
        for h in range(type_matrix.shape[1]):
            model.addConstr(
                gp.quicksum(x[i] * type_matrix[i, h] for i in range(len(list_items))) <= lims_type[h], name=f"type_{h}"
            )

    # Size constraint
    model.addConstr(
        gp.quicksum(x[i] for i in range(len(list_items))) == bundle_size, name=f"size"
    )

    ## Decision
    for i in range(len(list_items)):
        for j in range(i + 1, len(list_items)):
            model.addConstr(
                y[i, j] - x[i] <= 0, name=f"dec_ub_{i}{j}_{i}"
            )
    for i in range(len(list_items)):
        for j in range(i + 1, len(list_items)):
            model.addConstr(
                y[i, j] - x[j] <= 0, name=f"dec_ub_{i}{j}_{j}"
            )
    for i in range(len(list_items)):
        for j in range(i + 1, len(list_items)):
            model.addConstr(
                x[i] + x[j] - 1 - y[i, j] <= 0, name=f"dec_lb_{i}{j}"
            )


    ## Fairness
    if p_fair:
        if len(disc_matrix.shape) == 1:
            disc_matrix = disc_matrix[:, np.newaxis]
        for k in range(disc_matrix.shape[1]):
            model.addConstr(
                gp.quicksum(x[i] * (p_fair_targets[k] - disc_matrix[i, k]) for i in range(len(list_items))) <= disc_budgets[k], name=f"p_fair_{k}"
            )

    # Set objective function
    rel_expr = 0
    sim_expr = 0
    for i in range(len(list_items)):
        rel_expr += x[i] * rel_vec[i]
        for j in range(i + 1, len(list_items)):
            sim_expr += y[i, j] * sim_matrix[i, j]
    rel_expr /= bundle_size
    sim_expr /= 0.5 * bundle_size * (bundle_size - 1)
    obj_expr = (1 - gamma) * rel_expr + gamma * sim_expr
    model.setObjective(obj_expr, sense=GRB.MAXIMIZE)


    model.update()

    start_optim = time()
    model.optimize()
    end = time()
    print(p_fair * "Fair" + (1 - p_fair) * "Optimal" + f"Optimization duration: {end - start_optim:.2f} seconds")
    print(p_fair * "Fair" + (1 - p_fair) * "Optimal" + f"Optimization duration incl. Model building: {end - start_model:.2f} seconds")

    # Fair bundle
    if model.status == GRB.OPTIMAL:
        obj_value = model.getObjective().getValue()
        print(p_fair * "Fair" + (1 - p_fair) * "Optimal" + f"Objective Value: {obj_value}")
        info = {"solution": True, "optimal": True, "obj_value": obj_value}
        bundle = [list_items[i] for i in range(len(list_items)) if x[i].x == 1]
    elif model.SolCount > 0:
        info = {"solution": True, "optimal": False}
        bundle = [list_items[i] for i in range(len(list_items)) if x[i].x == 1]
    else:
        info = {"solution": False, "optimal": False}
        bundle = []

    info['time'] = end - start_model

    return bundle, info