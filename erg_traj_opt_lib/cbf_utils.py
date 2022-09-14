def sdf2cbf(f, constr):
    return lambda x, u, alpha: constr(f(x,u)) - (1.-alpha) * constr(x)