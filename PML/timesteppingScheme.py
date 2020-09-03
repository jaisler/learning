from firedrake import *

# -------------------------- #
# Newmark time-stepping scheme - time integration method (first timestep)
def Newmark_scheme_initial(beta, gamma, m, a, v, u_prevs, v_i, f0, f1, qr_x, dt):

    LHS = m + dt*dt*beta*a # left-hand side
    RHS = (m - dt*dt*(0.5-beta)*a)*u_prevs[0] + dt*m*v_i + dt*dt*(beta*f1+(0.5-beta)*f0)*v*dx(rule=qr_x) # right-hand side
    return LHS, RHS

# -------------------------- #
# Newmark time-stepping scheme - time integration method (step n)
def Newmark_scheme_update_rhs(beta, gamma, m, a, v, u_prevs, f0, f1, f2, qr_x, dt):

    RHS = ((2.0*m - dt*dt*(0.5-2.0*beta+gamma)*a)*u_prevs[1] - (m + dt*dt*(0.5+beta-gamma)*a)*u_prevs[0] 
        + dt*dt*(beta*f2+(0.5-2.0*beta+gamma)*f1+(0.5+beta-gamma)*f0)*v*dx(rule=qr_x)) # right-hand side
    return RHS
