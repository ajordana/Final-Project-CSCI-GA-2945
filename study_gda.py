import numpy as np
import matplotlib.pyplot as plt
from gda import GDA, FR


from test_examples.examples import g1, g2, g3, g4

ex = 3

if ex == 4:
    f = g4()
    x0 = -4
    y0 = -5
    T = 8e4
    gammas = [1, 2, 4, 8, 16]

if ex == 3:
    f = g3()
    x0 = 5
    y0 = 4
    T = 2e5
    gammas = [1, 4, 6, 16, 32]



eta_x = 1e-3
eta_y = 1e-3


x_GDA1, y_GDA1, x_GDA_grad1, y_GDA_grad1 = GDA(f, x0, y0, eta_x, eta_y, T, gamma=gammas[0])
x_GDA2, y_GDA2, x_GDA_grad2, y_GDA_grad2 = GDA(f, x0, y0, eta_x, eta_y, T, gamma=gammas[1])
x_GDA3, y_GDA3, x_GDA_grad3, y_GDA_grad3 = GDA(f, x0, y0, eta_x, eta_y, T, gamma=gammas[2])
x_GDA4, y_GDA4, x_GDA_grad4, y_GDA_grad4 = GDA(f, x0, y0, eta_x, eta_y, T, gamma=gammas[3])
x_GDA5, y_GDA5, x_GDA_grad5, y_GDA_grad5 = GDA(f, x0, y0, eta_x, eta_y, T, gamma=gammas[4])

fig = plt.figure()
v = np.linspace(-6, 6, 500)
x, y = np.meshgrid(v, v)
z = f.eval(x, y)
plt.contourf(x, y, z, 25)
plt.plot(np.array(x_GDA1), np.array(y_GDA1), linewidth=2, label="gamma = " + str(gammas[0]) )
plt.plot(np.array(x_GDA2), np.array(y_GDA2), linewidth=2, label="gamma = " + str(gammas[1]) )
plt.plot(np.array(x_GDA3), np.array(y_GDA3), linewidth=2, label="gamma = " + str(gammas[2]) )
plt.plot(np.array(x_GDA4), np.array(y_GDA4), linewidth=2, label="gamma = " + str(gammas[3]) )
plt.plot(np.array(x_GDA5), np.array(y_GDA5), linewidth=2, label="gamma = " + str(gammas[4]) )
 
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.colorbar()
plt.legend()

if ex == 3:
    plt.xticks(np.arange(-4, 6, 2))
    plt.yticks(np.arange(-4, 6, 2))
    plt.axis([-4, 6, -4, 6])
    plt.savefig("plots/contours3_inf_study.png")

if ex == 4:
    plt.xticks(np.arange(-6, 6, 2))
    plt.yticks(np.arange(-6, 6, 2))
    plt.axis([-6, 6, -6, 6])
    plt.savefig("plots/contours4_inf_study.png")


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(x_GDA_grad1), label="gamma = " + str(gammas[0]))
ax2.plot(np.array(y_GDA_grad1), label="gamma = " + str(gammas[0]))
ax1.plot(np.array(x_GDA_grad2), label="gamma = " + str(gammas[1]))
ax2.plot(np.array(y_GDA_grad2), label="gamma = " + str(gammas[1]))
ax1.plot(np.array(x_GDA_grad3), label="gamma = " + str(gammas[2]))
ax2.plot(np.array(y_GDA_grad3), label="gamma = " + str(gammas[2]))
ax1.plot(np.array(x_GDA_grad4), label="gamma = " + str(gammas[3]))
ax2.plot(np.array(y_GDA_grad4), label="gamma = " + str(gammas[3]))
ax1.plot(np.array(x_GDA_grad5), label="gamma = " + str(gammas[4]))
ax2.plot(np.array(y_GDA_grad5), label="gamma = " + str(gammas[4]))
ax2.set_xlabel("iterations")
ax1.set_ylabel("x grad")
ax2.set_ylabel("y grad")
ax1.grid()
ax2.grid()
ax1.set_yscale('log')
ax2.set_yscale('log')


handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 10})

if ex == 3:
    plt.savefig("plots/log_plot3_inf_study.png")
if ex == 4:
    plt.savefig("plots/log_plot4_inf_study.png")
plt.show()