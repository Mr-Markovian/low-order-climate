"""A vectorized code to compute multiple trajectories at simultaneously. The input has the shape
[time,dimension,no. of intial conditions]"""
from jax import vmap,jit 
from functools import partial
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ode_solvers import rk4_solver as solver
from testbed_models import L86_gcm
import os
import matplotlib.pyplot as plt


# these parameters now give you the mode, which is only a function of x, the state vector
model=L86_gcm
model_dim=5
num_initials=10

data_path='/home/shashank/Projects/low-order-climate/data'

os.chdir(data_path)
#X_o=np.expand_dims(np.load('Initial_condition_on_att_{}.npy'.format(model.__name__)),axis=1)
X_o=np.load('Initial_condition_on_att_{}.npy'.format(model.__name__))

T_start=0.0
T_stop=500.0
dt=0.01
dt_solver=0.01
iters_delta_t=int(dt/dt_solver)

@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    f_X= vmap(model,in_axes=(1),out_axes=(1))(X_) 
    return f_X    

Trajs=np.zeros((int((T_stop-T_start)/dt),model_dim,num_initials))
my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
@jit
def integrate_forward(X_):
    #forecast_ensemble=lax.fori_loop(0,iters_delta_t,my_solver,last_analysis_ensemble)
    X=X_
    for i in range(iters_delta_t):
        X=my_solver(x_initial=X)
    return X

print(X_o.shape)
X_now=X_o
Trajs[0]=X_now
for i in range(1,Trajs.shape[0]):
    X_next=integrate_forward(X_now)
    Trajs[i]=X_next
    X_now=X_next

#Save the solution
print(Trajs[-1])
print(Trajs.shape)
np.save('Multiple_trajectories_N={}_gap={}_ti={}_tf={}_dt_{}_dt_solver={}.npy'.format(num_initials,dt,T_start,T_stop,dt,dt_solver),Trajs[:,:,0])
print('job done')


fig = plt.figure(figsize=(14,10))
  
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
# plotting
#my_scatter=ax.scatter(base_traj10[:,0],base_traj10[:,1],base_traj10[:,2],c=cosines,cmap=colormap,s=15)
#for i in range(num_initials):
ax.plot(Trajs[:,0,0],Trajs[:,1,0],Trajs[:,2,0],c='blue',lw=1)
ax.plot(Trajs[:,0,1],Trajs[:,1,1],Trajs[:,2,1],c='black',lw=2)
ax.plot(Trajs[:,0,2],Trajs[:,1,2],Trajs[:,2,2],c='red',lw=3)

ax.set_xlabel(r'$X\to$')
ax.set_ylabel(r'$Y\to$')
ax.set_zlabel(r'$Z\to$')
#ax.set_zticks([10,20,30,40])
ax.text(0,0,45,r'$\sigma={}$'.format(0.0),fontsize=25)
#ax.set_title(r'$\sigma={}$'.format(sigma))
ax.grid(False)
#ax.set_title(r'$\dot x = 10 (y - x)\  ; \dot y = x (28 - z) - y \ ;  \dot z = x y - \frac{8}{3} z$')
#ax.text(0,-25,0,r'$\dot x = 10 (y - x) \; \dot y = x (28 - z) - y;  \dot z = x y - \frac{8}{3} z$',fontsize=20)
#fig.colorbar(my_plot,shrink=0.5, aspect=5)
#plt.legend()
plt.tight_layout()
#plt.savefig('enhanced_L63_attr_12_sigma={}_spacing={}.pdf'.format(sigma,dt*qrstep),dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()