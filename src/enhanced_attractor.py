import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
import matplotlib.cm as cm1
colormap=cm.batlowK
#colormap=cm1.tab20b
from matplotlib.colors import Normalize

# A colormap in dark mode
#colormap=cm.vanimo
#os.chdir('L63_BLV_test1')
mpl.rcParams['lines.markersize']=15
mpl.rcParams['axes.titlesize']=25
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
#mpl.rcParams['ztick.labelsize']=20
mpl.rcParams['axes.labelsize']=15

spr=1
dt=0.01   

model_dim=5
model_name='L86_gcm'
base_type='state'
num_clv=5
coord=['$x^{(1)}$','$x^{(2)}$','$x^{(3)}$']
sigma=0.0
dt=0.01
dt_solver=0.01
n_iters=int(dt/dt_solver)
startpt=0 # starting point of the forward transient
qrstep=1

# Adding colors to the arrows in 2d and 3d. 

data_path='/home/shashank/Projects/low-order-climate/data'
plots_path='/home/shashank/Projects/low-order-climate/figures'
os.chdir(data_path)
#base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[10000:20000]


C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
print(np.mean(np.load('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type)),axis=0))


#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
start_idx=2000 # starting point of the interval of clv( 25000+10000)
base_traj=np.load('Multiple_trajectories_N=10_gap={}_ti=0.0_tf=100.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[2000:7000]
base_traj10=base_traj[::spr]

V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  

# Cosines between the 1st and 2nd clv
cosines=np.zeros((G.shape[0]))
for i in range(G.shape[0]):
    cosines[i]=np.dot(V[i,:,1],V[i,:,2])

print(C.shape)
print(G.shape)
print(base_traj10.shape)
# Regime change in L63, by using CLVs
#plot the state and the trajectory

os.chdir(plots_path)

# fig = plt.figure(figsize=(15,10))
  
# # syntax for 3-D projection11
# ax = plt.axes(projection ='3d')
# # plotting
# #my_scatter=ax.scatter(base_traj10[:,0],base_traj10[:,1],base_traj10[:,2],c=cosines[::spr],cmap=colormap,s=15)
# ax.plot(base_traj1[:,0],base_traj1[:,1],base_traj1[:,2],c='grey',lw=1)
# ax.plot(base_traj[:,0],base_traj[:,1],base_traj[:,2],c='black',lw=1)
# plt.show()
# plt.savefig('comparing_trajs.pdf')


fig = plt.figure(figsize=(15,10))
  
# syntax for 3-D projection11
ax = plt.axes(projection ='3d')
# plotting
my_scatter=ax.scatter(base_traj10[:,0],base_traj10[:,1],base_traj10[:,2],c=cosines[:],cmap=colormap,s=30)
ax.plot(base_traj10[:,0],base_traj10[:,1],base_traj10[:,2],c='grey',lw=1)
ax.set_xlabel(r'$x^{(1)}\to$',fontsize=20)
ax.set_ylabel(r'$x^{(2)}\to$',fontsize=20)
ax.set_zlabel(r'$x^{(3)}\to$',fontsize=20)
#ax.set_zticks([10,20,30,40])
#ax.text(0,0,40,r'$\sigma={}$'.format(sigma),fontsize=20)
ax.text(0,0,40,r'$\langle C_1, C_2 \rangle$'.format(sigma),fontsize=30)
#ax.view_init(elev=16.)

#ax.set_title(r'$\sigma={}$'.format(sigma))
ax.grid(False)
#ax.set_title(r'$\dot x = 10 (y - x)\  ; \dot y = x (28 - z) - y \ ;  \dot z = x y - \frac{8}{3} z$')
#ax.text(0,-25,0,r'$\dot x = 10 (y - x) \; \dot y = x (28 - z) - y;  \dot z = x y - \frac{8}{3} z$',fontsize=20)
#fig.colorbar(my_scatter,shrink=0.5, aspect=5)
#plt.legend()
#plt.tight_layout()
plt.savefig('enhanced_L63_0.18_attr_01_sigma={}_spacing={}.pdf'.format(sigma,dt*spr),dpi=500, bbox_inches='tight', pad_inches=0.2)
plt.show()

# # Vector enhanced attractor
# colors=cosines.ravel()
# norm = Normalize()
# norm.autoscale(colors)

# fig = plt.figure(figsize=(16,10))
# ax = plt.axes(projection ='3d')
# # plotting
# my_quiver=ax.quiver(base_traj[:,0],base_traj[:,1],base_traj[:,2],V[:,0,0],V[:,1,0],V[:,2,0],cmap=colormap)
# ax.plot(base_traj[:,0],base_traj[:,1],base_traj[:,2],c='grey',lw=1)
# ax.set_xlabel(r'$X\to$')
# ax.set_ylabel(r'$Y\to$')
# ax.set_zlabel(r'$Z\to$')
# ax.grid(False)
# fig.colorbar(my_scatter,shrink=0.5, aspect=5)
# plt.legend()
# plt.tight_layout()
# plt.savefig('enhanced_L63_attr_12_vec.png',dpi=300, bbox_inches='tight', pad_inches=0)
