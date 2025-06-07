"""Computing CLVs along a trajectory by Ginelli's Algorithm.
We do not evolve the system, but only use the points on the trajectory to evolve
the perturbations in the tangent space."""

from jax import jvp,vmap,jit
from functools import partial
import jax.numpy as jnp
import numpy as np
import os
from numpy.linalg import norm,inv
from ode_solvers import rk4_solver as solver
#from scipy.integrate import solve_ivp 
from testbed_models import L86_gcm
qr_decomp=jnp.linalg.qr

# Compute the Jacobian of the system from the model
model=L86_gcm
model_name=L86_gcm.__name__
model_dim=5
num_clv=5



# When the number of clvs is less than the jacobian dimension, we do not compute the full jacobian, but use jvp 
@jit
def model_plus_tangent_propagator(X_):
    "model(x) + jacobian(x)V for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(1), out_axes=(None,1))((in_tangents,)) 
    Y_=X_.at[:,0].set(y)
    Z_=Y_.at[:,1:].set(out_tangents)
    return Z_

# Reorthonormalization step of the output tangent vectors is done using this function
@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:])  # QR decomposition
    S=jnp.diag(jnp.sign(jnp.diag(R_)))
    temp_=x_.at[:,1:].set(Q_@S)
    return temp_,Q_@S,S@R_

# The backward evolution computing the clv coefficients 
def backward_evolution(x_,y_k):
    """Generates coeffients at the previous time step from the current time step"""
    temp_C=inv(x_)@y_k
    #row_sums = temp_.sum(axis=1,keepdims=True)
    row_sums=np.linalg.norm(temp_C,ord=2,axis=0,keepdims=True)
    temp_C = temp_C / row_sums
    return temp_C

# Forward transient

# Load the trajectory of the system
dt=0.01
dt_solver=0.01
n_iters=int(dt/dt_solver)
startpt=0 # starting point of the forward transient
qrstep=1

# Change to data path

base_type='state'
data_path='/home/shashank/Projects/low-order-climate/data'
os.chdir(data_path)


base_traj=np.load('Multiple_trajectories_N=10_gap={}_ti=0.0_tf=100.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))

# orthogonal perturbations at t=0
X_start1=jnp.zeros((model_dim,num_clv+1))
X_start=X_start1.at[:,1:].set(np.eye(model_dim)[:,:num_clv])  

#X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
my_solver=jit(partial(solver,rhs_function=model_plus_tangent_propagator,time_step=dt_solver))

# Solver for three regions
n1=2000  # number of time steps in forward transient
n2=5000   # number of time steps in the interval
n3=2000  # number of time steps in backward transient

#Store local growth rates, the G, R and C matrices
local_growth_rates=np.zeros((int((n2+n3)/qrstep),num_clv))
Store_G=np.zeros((int(n2/qrstep),model_dim,num_clv))
Store_R=np.zeros((int((n2+n3)/qrstep),num_clv,num_clv))
Store_C=np.zeros((int(n2/qrstep),num_clv,num_clv))

# The forward transient
for i in range(n1):
    X_start=X_start.at[:,0].set(base_traj[i])
    for j in range(n_iters):
       X_stop=my_solver(x_initial=X_start)
       X_start=X_stop
    #local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=1))/dt
    if(np.mod(i,qrstep)==qrstep-1):
        X_start,_,_=Output_tangent_QR(X_stop)

print('Transient is over, moving on to the interval in question')

# Interval where we sample the clvs.
for i in range(n2):
    X_start=X_start.at[:,0].set(base_traj[n1+i])
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    if(np.mod(i,qrstep)==qrstep-1):
        X_start,Store_G[int(i/qrstep)],Store_R[int(i/qrstep)]=Output_tangent_QR(X_stop)
        local_growth_rates[int(i/qrstep)]=np.log(np.diag(Store_R[int(i/qrstep)]))

print('The matrics of backward lyapunov vectors stored, moving on to backward transient part')

# #Free Up space
np.save('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_G)  
del Store_G  

# Backward transent interval begins, where going forward over the backward transient interval
# Store Q and R matrices
for i in range(n3):
    X_start=X_start.at[:,0].set(base_traj[n1+n2+i])
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    if(np.mod(i,qrstep)==qrstep-1):
        X_start,_,Store_R[int(n2/qrstep)+int(i/qrstep)]=Output_tangent_QR(X_stop)
        local_growth_rates[int(n2/qrstep)+int(i/qrstep)]=np.log(np.diag(Store_R[int(n2/qrstep)+int(i/qrstep)]))

# Free up space
np.save('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),local_growth_rates)
del local_growth_rates

print('Moving backward on the backward transient part, to converge to the clvs')   

# A generic upper triangular matirx for the backward transient
C_tilde_initial=np.zeros((num_clv,num_clv))
for j in range(num_clv):
    C_tilde_initial[j,j:]=np.ones(num_clv-j)
C_tilde=C_tilde_initial
print(C_tilde)

#  Going backwards over the backward transient via backward evolution using stored R-inverse
for i in range(int(n3/qrstep)):
    l=int(n2/qrstep)+int(n3/qrstep)-1-i
    C_tilde=backward_evolution(Store_R[l],C_tilde)

# CLV on the interval of interest 
print('Sampling the clvs over the interval of interest')

# # Sampling the clvs(i.e. in terms of coefficients when expressed in BLV) 
for i in range(int(n2/qrstep)):
    l=int(n2/qrstep)-1-i
    C_tilde=backward_evolution(Store_R[l],C_tilde)
    Store_C[l]=C_tilde

print('Sampled, now storing all data')
np.save('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_C)
print('Job done')
