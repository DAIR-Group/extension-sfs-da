import cvxpy as cp

def solve(A, delta, B1, B2):
    x = cp.Variable(A.shape[0])
    objective = cp.Minimize(0.5 * cp.quad_form(x, A)  + delta.T @ x)
    constraints = [B1 @ x <= 0, B2 @ x == 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
    # prob.solve(solver=cp.PROXQP, backend='sparse', eps_abs=1e-10, eps_rel=1e-10, verbose=False)
    return x.value.reshape(-1,1), prob.constraints[0].dual_value.reshape(-1,1), prob.constraints[1].dual_value.reshape(-1,1)