import jax.numpy as jnp
from jax import vmap

def model_vectorized(ode_rhs, X_):
    """
    Model vectorization using vmap to compute multiple forward passes.
    """
    f_X = vmap(ode_rhs, in_axes=(1), out_axes=(1))(X_)
    return f_X

#Lorenz-86 with default values: 
def L86_gcm(X,a=0.25,b=4.0,F0=8.0,F1=0.021685,G0=1.0,G1=0.01,Tav=30,gamma=30,delta0=7.8e-7,delta1=9.6e-8,ka=1.8e-4,kw=1.8e-5,w=1.3e-4,zeta=1.1e-3):
    """Function to be used for compuation of ode in scipy.integrate.solve_ivp"""
    # sailinity and temperature coupling function
    f= lambda T1, S1, : w*T1 - zeta *S1
    x,y,z,T,S=X
    dx_dt= -y**2 - z**2 -a * x +(a * F0+ F1 *T )
    dy_dt=x*y -b*x*z-y +G0 + G1*(Tav-T)
    dz_dt=b*x*y+x*z-z
    dT_dt=ka*(gamma*x-T)-f(T,S)*T - kw*T
    dS_dt=delta0+ delta1*(y**2+z**2)-f(T,S)*S-kw*S
    return jnp.array([dx_dt,dy_dt,dz_dt,dT_dt,dS_dt])


# L96 2-level model, 
def L96_2level(t,x_,d_x,d_y,F_x,c=10,b=10,h=1):
    "2 scale lorenz 96 model, the input is combined vector x_=[x y] "
    X=x_[0:d_x]
    Y=x_[d_x:].reshape(d_x,d_y)
    dx_dt=-jnp.roll(X,-1)*(jnp.roll(X,-2)-jnp.roll(X,1))-X+F_x-(h*c/b)*jnp.sum(Y,axis=1)
    dy_dt=-c*b*(jnp.roll(Y,1,axis=1)*(jnp.roll(Y,2,axis=1)-jnp.roll(Y,-1,axis=1)))-c*Y+(h*c/b)*jnp.tile(X,(d_y,1)).T
    dx=jnp.zeros_like(x_)
    dx[0:d_x]=dx_dt
    dx[d_x:]=dy_dt.flatten()
    return dx

# L96 single level model with forcing=10
def L96(x_,forcing=8.):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(jnp.roll(x_,-1)-jnp.roll(x_,2))*jnp.roll(x_,1)-x_+forcing
    return dx_dt

#Lorenz-63 with default values: sigma=10, rho=28, beta=8/3
def L63(x,sigma=10.,rho=28.,beta=8./3):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    return jnp.array([sigma*(x[1]-x[0]),x[0]*(rho-x[2])-x[1],x[0]*x[1]-beta*x[2]])
    

