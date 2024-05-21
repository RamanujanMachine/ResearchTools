import sympy as sp
from ramanujan.cmf import CMF
from sympy.abc import n


def trajectory_scan(C : CMF,traj_space, start, iterations = 100):
    r"""
    scans trajectories and returns useful information
    Given a CMF, a List of trajectories, and a starting point, this function:
    Calculates the trajectory matrix of each trajectory in the list
    Modifies the starting point as to prevent singularities
    converts the matrix to a PCF
    calculates the limit and the delta of the PCF
    returns the limit, precision, delta, pcf, and starting point for each trajectory.
    """
    return [analyze_trajectory(C,traj,start,iterations) for traj in traj_space]


def analyze_trajectory(C : CMF,traj, start, iterations=100):
    r"""
    extracts and returns useful information from a trajectory of a CMF.
    Given a CMF, a trajectory, and a starting point, this function:
    Calculates the trajectory matrix
    Modifies the starting point as to prevent singularities
    converts the matrix to a PCF
    calculates the limit and the delta of the PCF
    returns the limit, precision, delta, pcf, and starting point for the trajectory.
    """
    Mtraj = C.trajectory_matrix(traj, start) #traj matrix
    MtrajDet = Mtraj.det() #find Determinant
    Mtraj_denom = sp.lcm_list([sp.denom(elem) for elem in Mtraj]) #find Denominator
    sing_list = sp.solve(sp.denom(MtrajDet))+sp.solve(sp.numer(MtrajDet))+sp.solve(Mtraj_denom) # list of singular offets.
    offset = max(0,max([r for r in sing_list if r.is_integer])) # finds maximal integer root over 0.
    if offset!=0:
        start = {k:(start[k]+offset*traj[k]) for k in C.axes()}
        Mtraj = Mtraj.subs({n:n+offset})
        Mtraj_denom = Mtraj_denom.subs({n:n+offset})
    Mtraj_norm = (Mtraj*Mtraj_denom).normalize()
    traj_pcf = Mtraj_norm.as_pcf().pcf
    lim_obj = traj_pcf.limit(iterations)
    lim = lim_obj.as_rounded_number()
    prec = lim_obj.precision()
    delta = traj_pcf.delta(iterations)
    return (start,traj_pcf,lim,prec,delta)

