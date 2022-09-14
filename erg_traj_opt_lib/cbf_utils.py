def sdf2cbf(f, constr, alpha):
    return lambda x, u: constr(f(x,u)) - (1.-alpha) * constr(x)