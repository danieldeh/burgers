import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Solve nonlinear Burgers equation with stochastic forcing

# du/dt + u du/dx = \nu d2u/dx2 + f_t(x)

def main():
    # Parameters
    L = 2 * np.pi  # Domain length [0, L]
    T = 10  # Solution integrated until [0, T]
    nu = 0.1  # Viscosity

    nx = 101  # Number of grid points
    nt = 10000  # Number of timesteps

    s = 100  # Save solution every s timesteps

    # Initial condition
    x = np.linspace(0, L, nx)
    u = np.sin(x * 2 * np.pi / L)

    # Derived quantities
    dt = T / nt
    dx = L / nx

    # Numerical scheme stability check
    print("Viscous CFL is %f" % (nu * dt / dx / dx))
    print("Initial Advection CFL is %f" % (dt * np.max(u) / dx))
    print("MacCormack stability CFL is %f" % (dt * (np.max(u) * dx + 2 * nu) / dx / dx))

    # Define forcing function
    forcing = source(x, nx, nt)

    # Solve
    # us = ftcs(u, nu, nx, nt, dx, dt, forcing, s)
    us = macCormack(u, nu, nx, nt, dx, dt, forcing, s)

    # Save gif of solution

    def animate(n):
        line.set_data(x, us[n, :])
        # line.set_data(x, forcing[:, n])
        return line

    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(-2, 2))
    line = ax.plot([], [])[0]
    anim = animation.FuncAnimation(fig, animate, frames=len(us), interval=100)
    writergif = animation.PillowWriter(fps=30)
    anim.save('burgersSolution.gif', writer=writergif)


# Simulate Brownian motion
def random_walk(k, nt, mu=0, sigma=1):
    """ Simulates k random walks"""
    W_t = np.ones([nt, len(k)])

    for i in range(1, nt):
        yi = np.random.normal(mu, sigma, len(k))
        W_t[i, :] = W_t[i - 1, :] + yi / np.sqrt(nt)

    return W_t


def source(x, nx, nt, kmin=1, kmax=3, mu=0, sigma=1):
    """ Defines stochastic forcing function f_t(x) = \\sum_{k=kmin}^kmax \\sin(k \\pi x) d W_t^k"""
    k = np.arange(kmin, kmax + 1)
    forcing = np.zeros([nx, nt])
    w = random_walk(k, nt, mu, sigma)
    for i in range(len(k)):
        forcing += 1 * np.sin(k[i] * np.pi * x).reshape([-1, 1]) * w[:, i].reshape([-1, 1]).transpose()
    return forcing


def ftcs(u, nu, nx, nt, dx, dt, forcing, s):
    """
    Forward in time, First-order upwind difference for convective term, centered in space for diffusion term.
    Note that this scheme is for testing only - it is far too diffusive to be used in production and is
    unconditionally unstable for low \nu
    """
    # Allocate memory for solution
    us = np.ones(nx) * u

    for i in range(nt):

        # phi2 = 2*np.pi * np.random.rand()
        # phi3 = 2*np.pi*np.random.rand()
        # KForce = 2*np.pi/L
        # F = np.sqrt(dt) * (np.cos(2*KForce * x + phi2) + np.cos(3*KForce*x+phi3))

        un = u.copy()

        # Finite difference for interior points
        u[1:-1] = un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) + nu * dt / dx ** 2 * \
                  (un[2:] - 2 * un[1:-1] + un[:-2]) + dt * forcing[1:-1, i]

        # Boundary conditions (Periodic)
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-1]) + nu * dt / dx ** 2 * \
               (un[1] - 2 * un[0] + un[-1]) + dt * forcing[0, i]
        u[-1] = un[-1] - un[-1] * dt / dx * (un[-1] - un[-2]) + nu * dt / dx ** 2 * \
                (un[0] - 2 * un[-1] + un[-2]) + dt * forcing[-1, i]

        # Save solution every s timesteps
        if i % s == 0:
            us = np.vstack((us, u))

        # Check stability criterion
        if dt * np.max(u) / dx > 1:
            print("Advection CFL is %f for t = %g" % (dt * np.max(u) / dx, dt * i))
    return us


def macCormack(u, nu, nx, nt, dx, dt, forcing, s):
    """ macCormack Scheme for solving 1D Burger's"""
    # Allocate memory for solution
    us = np.ones(nx) * u

    def f(uin):
        cons = 0.5 * uin ** 2
        return cons

    for i in range(nt):

        u_star = u.copy()
        un = u.copy()

        # Predictor

        u_star[1:-1] = un[1:-1] - dt / dx * (f(un[2:]) - f(un[1:-1])) + nu * dt / dx / dx * \
                       (un[2:] - 2 * un[1:-1] + un[:-2]) + dt * forcing[1:-1, i]

        # Periodic Boundary Conditions
        u_star[0] = un[0] - dt / dx * (f(un[1]) - f(un[0])) + nu * dt / dx / dx * \
                    (un[1] - 2 * un[0] + un[-1]) + dt * forcing[0, i]

        u_star[-1] = un[-1] - dt / dx * (f(un[0]) - f(un[-1])) + nu * dt / dx / dx * \
                     (un[0] - 2 * un[-1] + un[-2]) + dt * forcing[-1, i]

        # Corrector

        u[1:-1] = 0.5 * (un[1:-1] + u_star[1:-1] - dt / dx * (f(u_star[1:-1]) - f(u_star[:-2])) + \
                         nu * dt / dx / dx * (u_star[2:] - 2 * u_star[1:-1] + u_star[:-2]) + \
                         dt * forcing[1:-1, i])

        # Periodic Boundary conditions
        u[0] = 0.5 * (un[0] + u_star[0] - dt / dx * (f(u_star[0]) - f(u_star[-1])) + \
                      nu * dt / dx / dx * (u_star[1] - 2 * u_star[0] + u_star[-1]) + \
                      dt * forcing[0, i])

        u[-1] = 0.5 * (un[-1] + u_star[-1] - dt / dx * (f(u_star[-1]) - f(u_star[-2])) + \
                       nu * dt / dx / dx * (u_star[0] - 2 * u_star[-1] + u_star[-2]) + \
                       dt * forcing[-1, i])

        # Save solution every s timesteps
        if i % s == 0:
            us = np.vstack((us, u))

        # Check stability criterion
        if dt * (np.max(u) * dx + 2 * nu) / dx / dx > 1:
            print("MacCormack stability CFL is %f for t = %g" % (dt * (np.max(u) * dx + 2 * nu) / dx / dx, dt * i))
    return us


if __name__ == '__main__':
    main()
