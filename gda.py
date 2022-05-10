import numpy as np
import matplotlib.pyplot as plt


def GDA(func, x0, y0, eta_x, eta_y, T, gamma=1, reccord=True):
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
        if reccord:
            x_traj.append(x)
            y_traj.append(y)
    if reccord:
        return x_traj, y_traj, x_grad, y_grad
    else:
        return x, y, x_grad, y_grad

def FR(func, x0, y0, eta_x, eta_y, T, reccord=True):
    x = x0
    y = y0
    x_traj = [x]
    y_traj = [y]
    x_grad = []
    y_grad = []
    for t in range(int(T)):
        func.calc(x, y)
        x = x - eta_x * func.fx
        ridge = func.fyx * func.fx / func.fyy if np.isscalar(x) else  np.linalg.solve(func.fyy, func.fyx @ func.fx)
        y = y + eta_y * func.fy + eta_x * ridge
        x_grad.append(np.linalg.norm(func.fx))
        y_grad.append(np.linalg.norm(func.fy)) 
        if reccord:
            x_traj.append(x)
            y_traj.append(y)
    if reccord:
        return x_traj, y_traj, x_grad, y_grad
    else:
        return x, y, x_grad, y_grad


if __name__ == "__main__":
    from test_examples.examples import g1, g2, g3, g4

    examples = [g1(), g2(), g3(), g4()]
    names = ["g1", "g2", "g3", "g4"]

    eta_x = 1e-3
    eta_y = 1e-3
    T = 1e4

    for i in range(4):

        f = examples[i]
        x0 = 5
        y0 = 4
        if i == 3:
            x0 = -4
            y0 = -5
            eta_x = 1e-4
            eta_y = 1e-4
            T = 7e4


        x_GDA, y_GDA, x_GDA_grad, y_GDA_grad = GDA(f, x0, y0, eta_x, eta_y, T)


        x_FR, y_FR, x_FR_grad, y_FR_grad = FR(f, x0, y0, eta_x, eta_y, T)

        fig = plt.figure()
        v = np.linspace(-10, 10, 200)
        x, y = np.meshgrid(v, v)
        z = f.eval(x, y)
        plt.contourf(x, y, z, 25)

        plt.plot(np.array(x_GDA), np.array(y_GDA), linewidth=2, color="green", label="GDA")
        plt.plot(x_GDA[-1], y_GDA[-1], "p", color="green", markersize=14)

        plt.plot(np.array(x_FR), np.array(y_FR), linewidth=2, color="red", label="FR")
        plt.plot(x_FR[-1], y_FR[-1], "p", color="red", markersize=14)


        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.xticks(np.arange(-10, 10, 4))
        plt.yticks(np.arange(-10, 10, 4))
        plt.colorbar()
        plt.axis([-10, 10, -10, 10])
        plt.legend()
        plt.savefig("plots/contours_" + names[i] + ".png")


        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(np.array(x_GDA_grad), label="GDA")
        ax1.plot(np.array(x_FR_grad), label="FR")

        ax2.plot(np.array(y_GDA_grad), label="GDA")
        ax2.plot(np.array(y_FR_grad), label="FR")
        ax2.set_xlabel("iterations")
        ax1.set_ylabel("x grad")
        ax2.set_ylabel("y grad")
        ax1.grid()
        ax2.grid()
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 10})
        plt.savefig("plots/log_plot_" + names[i] + ".png")