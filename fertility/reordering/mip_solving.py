import time

from ortools.linear_solver import pywraplp

import numpy as np


import torch


def or_tools_solve(log_x: torch.Tensor, log_y: torch.Tensor, solver):
    if log_x.requires_grad:
        raise ValueError("First argument requires grad but we cannot compute this here!")
    if log_y.requires_grad:
        raise ValueError("Second argument requires grad but we cannot compute this here!")

    x_np = log_x.detach().cpu().numpy()
    y_np = log_y.detach().cpu().numpy()

    x_outs = []
    y_outs = []
    for b in range(x_np.shape[0]):
        x_o, y_o = solve_qap3_lp(x_np[b], y_np[b], solver_name=solver)
        x_outs.append(x_o)
        y_outs.append(y_o)

    x_out = torch.from_numpy(np.stack(x_outs)).to(log_x.device)
    y_out = torch.from_numpy(np.stack(y_outs)).to(log_y.device)
    return torch.log(x_out + 1e-40), torch.log(y_out + 1e-40)


def solve_qap3_lp(log_x, log_y, solver_name="GLOP"):
    n = log_x.shape[0]
    assert log_x.shape == (n, n)
    assert log_y.shape == (n, n, n)

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise ValueError(f"Solver {solver_name} not available")

    force_int = solver_name == "SCIP" # the integer solver

    x = dict()
    y = dict()
    for i in range(n):
        for j in range(n):
            if force_int:
                x[i, j] = solver.IntVar(0, 1, "")
            else:
                x[i, j] = solver.NumVar(0, 1, "")

            for k in range(n):
                if force_int:
                    y[i, j, k] = solver.IntVar(0, 1, "")
                else:
                    y[i, j, k] = solver.NumVar(0, 1, "")

    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n)]) == 1)
        solver.Add(solver.Sum([x[j, i] for j in range(n)]) == 1)

        for j in range(1, n):
            solver.Add(solver.Sum([y[i, j, k] for k in range(n)]) == x[i, j])
            solver.Add(solver.Sum([y[new_i, j, i] for new_i in range(n)]) == x[i, j-1])

    objective_terms = []
    for i in range(n):
        for j in range(n):
            objective_terms.append(x[i, j] * log_x[i, j])
            for k in range(n):
                objective_terms.append(log_y[i, j, k] * y[i, j, k])

    solver.Maximize(solver.Sum(objective_terms))

    status = solver.Solve()

    out_x = np.zeros((n, n))
    out_y = np.zeros((n, n, n))

    for i in range(n):
        for j in range(n):
            out_x[i, j] = x[i, j].solution_value()
            for k in range(n):
                out_y[i, j, k] = y[i, j, k].solution_value()

    return out_x, out_y


if __name__ == '__main__':
    from fertility.reordering.bregman_for_perm import prepare_scores, make_random_permutation, make_instance
    # ini, tr, fin = make_instance(4, max_val=20)

    torch.manual_seed(17)

    ini, tr, fin = make_random_permutation(3, seed=43, max_val=20)

    # np_ini = ini.squeeze(0).numpy()
    # np_tr = tr.squeeze(0).numpy()
    # np_fin = fin.squeeze(0).numpy()

    # r = solve_qap3_mip(np_ini, np_tr, np_fin, solver_name="Clp")

    log_x, log_y = prepare_scores(ini, tr, fin)

    log_x = torch.randn((1, 3, 3)) * 50
    log_y = torch.randn((1, 3, 3, 3)) * 10
    # still need to adjust temperature if

    r_x, r_y = solve_qap3_lp(log_x.squeeze(0).numpy(), log_y.squeeze(0).numpy(), solver_name="Clp")

    print(r_x, r_y)
    print("log_x")
    print(log_x)
    print("log_y")
    print(log_y)
