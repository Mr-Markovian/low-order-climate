"""A vectorized code to compute multiple initial conditions at simultaneously. The points stored have
the shape [dimension,no. of intial conditions]"""
from jax import vmap,jit 
from functools import partial
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import rk4_solver as solver
from testbed_models import L86_gcm
import os

seed_num=3
np.random.seed(seed_num)

# these parameters now give you the mode, which is only a function of x, the state vector
model=L86_gcm
model_dim=5
num_initials=10

data_path='/home/shashank/Projects/low-order-climate/data'


# Generate intial conditions from a gaussian with variance =epsilon  
epsilon=0.1
X_o=np.random.multivariate_normal(np.zeros(model_dim),epsilon*np.eye(model_dim),size=num_initials).T

T_start=0.0
T_stop=50.0
dt=0.01
dt_solver=0.01
iters_delta_t=int(dt/dt_solver)
Trajs=np.zeros((int((T_stop-T_start)/dt),model_dim,num_initials))

@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    f_X= vmap(model,in_axes=(1),out_axes=(1))(X_) 
    return f_X    

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

@jit
def integrate_forward(X_):
    #forecast_ensemble=lax.fori_loop(0,iters_delta_t,my_solver,last_analysis_ensemble)
    X=X_
    for i in range(iters_delta_t):
        X=my_solver(x_initial=X)
    return X

#X_now=np.expand_dims(X_o,axis=1)
X_now=X_o
Trajs[0]=X_now
for i in range(1,Trajs.shape[0]):
    X_next=integrate_forward(X_now)
    Trajs[i]=X_next
    X_now=X_next

os.chdir(data_path)

#Save the last point of the solution
print(Trajs[-1,:])
np.save('Initial_condition_on_att_{}.npy'.format(model.__name__),Trajs[-1])
print('job done')