# low-order-climate

This project contains details about a dynamical systems study on a small toy model of coupled ocean atomosphere dynamics, given by the following equations.


```python
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


This is a precursor to a more complicated model- a PDE based one. Which could be a PhD problem in iself.

The link to the actual paper introducing them is here: 
https://www.tandfonline.com/doi/abs/10.3402/tellusa.v53i5.12229

What I want to do is to openup this loops, to understand closely the dynamics of the LVs, and see the imapct on the ocean variables and the atmosphere variables seperately

Again, I don't have much time to work on this, but I will try to add more details as I go along.

