import numpy as np
import matplotlib.pyplot as plt


def GDA(func, x0, y0, eta_x, eta_y, T):
    gamma = 10
    x = x0
    y = y0
    x_traj = [x]
    y_traj = [y]
    x_grad = []
    y_grad = []
    for t in range(int(T)):
        func.calc(x, y)
        x = x - eta_x * func.fx / gamma
        y = y + eta_y * func.fy
        x_grad.append(np.linalg.norm(func.fx))
        y_grad.append(np.linalg.norm(func.fy)) 
        x_traj.append(x)
        y_traj.append(y)
    return x_traj, y_traj, x_grad, y_grad


def FR(func, x0, y0, eta_x, eta_y, T):
    x = x0
    y = y0
    x_traj = [x]
    y_traj = [y]
    for t in range(int(T)):
        func.calc(x, y)
        x = x - eta_x * func.fx
        y = y + eta_y * func.fy + eta_x * func.fyx * func.fx / func.fyy
        x_traj.append(x)
        y_traj.append(y)
    return x_traj, y_traj


# Local Sympletic Surgery
# def LSS(func, x0, y0, eta_x, eta_y, T):
#     x = x0
#     y = y0
#     x_traj = [x]
#     y_traj = [y]
#     v = np.array([1, 1])
#     xi2 = 1e-4
#     for t in range(int(T)):
#         func.calc(x, y)
#         J = np.array([[func.fxx, func.fyx], [- func.fxy, - func.fyy]])
#         omega = np.array([func.fx, - func.fy])
#         delta_z = omega + np.exp(-xi2 * np.linalg.norm(J.T @ v)**2) * J.T @ v
#         lambd = 1e-3 * (1 - np.exp(-np.linalg.norm(omega)**2))

#         delta_v = J.T @ J @ v + lambd * v - J.T @ omega
#         x = x - eta_x * delta_z[0]
#         y = y - eta_y * delta_z[1]
#         v = v - 0.001 * delta_v
#         x_traj.append(x)
#         y_traj.append(y)
#     return x_traj, y_traj


if __name__ == "__main__":
    from test_examples.example_FR_paper import g1, g2, g3

    f = g1()

    x0 = 4
    y0 = 5
    eta_x = 1e-3
    eta_y = 1e-3
    T = 1e4

    x_GDA, y_GDA, _, _ = GDA(f, x0, y0, eta_x, eta_y, T)

    T = 1e3

    x_FR, y_FR = FR(f, x0, y0, eta_x, eta_y, T)

    fig = plt.figure()
    v = np.linspace(-10, 10, 200)
    x, y = np.meshgrid(v, v)
    z = f.eval(x, y)
    plt.contourf(x, y, z, 25)

    plt.plot(np.array(x_GDA), np.array(y_GDA), linewidth=2, color="green", label="GDA")
    plt.plot(x_GDA[-1], y_GDA[-1], "p", color="green", markersize=14)

    plt.plot(np.array(x_FR), np.array(y_FR), linewidth=2, color="red", label="FR")
    plt.plot(x_FR[-1], y_FR[-1], "p", color="red", markersize=14)

    fig.suptitle(" f")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xticks(np.arange(-10, 10, 4))
    plt.yticks(np.arange(-10, 10, 4))
    plt.colorbar()
    plt.axis([-10, 10, -10, 10])
    plt.legend()
    plt.show()
