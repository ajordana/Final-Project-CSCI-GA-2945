import numpy as np
import matplotlib.pyplot as plt
from gda import GDA, FR
from test_examples.point_mass import point_mass


f = point_mass()
x0 = np.zeros(f.fx.shape)
y0 = np.zeros(f.fy.shape)
p = np.linspace(0, f.target[0], f.T)
for i in range(f.T):
    y0[4 * i] = p[i]

eta_x = 2e-4
eta_y = 2e-4

T = 5e4


u_GDA, s_GDA, x_grad_GDA, y_grad_GDA = GDA(f, x0, y0, eta_x, eta_y, T, reccord=False)
u_FR, s_FR, x_grad_FR, y_grad_FR = FR(f, x0, y0, eta_x, eta_y, T, reccord=False)


states_GDA = [f.x0] + [s_GDA[i * f.nx : (i + 1) * f.nx] for i in range(f.T)]
controls_GDA = [u_GDA[i * f.nu : (i + 1) * f.nu] for i in range(f.T)]
next_states_GDA = [f.dyn(states_GDA[i], controls_GDA[i]) for i in range(f.T)]

states_FR = [f.x0] + [s_FR[i * f.nx : (i + 1) * f.nx] for i in range(f.T)]
controls_FR = [u_FR[i * f.nu : (i + 1) * f.nu] for i in range(f.T)]
next_states_FR = [f.dyn(states_FR[i], controls_FR[i]) for i in range(f.T)]


plt.figure()
for t in range(f.T):
    if t == 0:
        plt.plot(
            np.array([states_GDA[t][0], next_states_GDA[t][0]]),
            np.array([states_GDA[t][1], next_states_GDA[t][1]]),
            "g",
            label="GDA",
        )
        plt.plot(
            np.array([states_FR[t][0], next_states_FR[t][0]]),
            np.array([states_FR[t][1], next_states_FR[t][1]]),
            "r",
            label="FR",
        )
    else:
        plt.plot(
            np.array([states_GDA[t][0], next_states_GDA[t][0]]),
            np.array([states_GDA[t][1], next_states_GDA[t][1]]),
            "g",
        )
        plt.plot(
            np.array([states_FR[t][0], next_states_FR[t][0]]),
            np.array([states_FR[t][1], next_states_FR[t][1]]),
            "r",
        )

plt.xlabel(r"$p_x$")
plt.ylabel(r"$p_y$")
plt.grid()
plt.title("Position trajectory")
plt.legend()

plt.savefig("plots_pm/trajectory.png")


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
for t in range(f.T):
    if t == 0:
        ax1.plot(
            np.array([t, t + 1]),
            np.array([states_GDA[t][0], next_states_GDA[t][0]]),
            "g",
            label="GDA",
        )
        ax1.plot(
            np.array([t, t + 1]),
            np.array([states_FR[t][0], next_states_FR[t][0]]),
            "r",
            label="FR",
        )
    else:
        ax1.plot(
            np.array([t, t + 1]),
            np.array([states_GDA[t][0], next_states_GDA[t][0]]),
            "g",
        )
        ax1.plot(
            np.array([t, t + 1]), np.array([states_FR[t][0], next_states_FR[t][0]]), "r"
        )

    ax2.plot(
        np.array([t, t + 1]), np.array([states_GDA[t][1], next_states_GDA[t][1]]), "g"
    )
    ax3.plot(
        np.array([t, t + 1]), np.array([states_GDA[t][2], next_states_GDA[t][2]]), "g"
    )
    ax4.plot(
        np.array([t, t + 1]), np.array([states_GDA[t][3], next_states_GDA[t][3]]), "g"
    )

    ax1.plot(
        np.array([t, t + 1]), np.array([states_FR[t][0], next_states_FR[t][0]]), "r"
    )
    ax2.plot(
        np.array([t, t + 1]), np.array([states_FR[t][1], next_states_FR[t][1]]), "r"
    )
    ax3.plot(
        np.array([t, t + 1]), np.array([states_FR[t][2], next_states_FR[t][2]]), "r"
    )
    ax4.plot(
        np.array([t, t + 1]), np.array([states_FR[t][3], next_states_FR[t][3]]), "r"
    )

ax1.set_ylabel(r"$p_x$")
ax2.set_ylabel(r"$p_y$")
ax3.set_ylabel(r"$v_x$")
ax4.set_ylabel(r"$v_y$")
ax1.legend()

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.savefig("plots_pm/state_traj.png")


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(controls_FR)[:, 0], "r", label="FR")
ax2.plot(np.array(controls_FR)[:, 1], "r", label="FR")
ax1.plot(np.array(controls_GDA)[:, 0], "g", label="GDA")
ax2.plot(np.array(controls_GDA)[:, 1], "g", label="GDA")
ax1.grid()
ax2.grid()
ax2.set_xlabel("Time")
ax1.set_ylabel(r"$u_x$")
ax2.set_ylabel(r"$u_y$")
ax1.legend()
plt.savefig("plots_pm/control_traj.png")


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(x_grad_GDA), "g", label="GDA")
ax2.plot(np.array(y_grad_GDA), "g", label="GDA")

ax1.plot(np.array(x_grad_FR), "r", label="FR")
ax2.plot(np.array(y_grad_FR), "r", label="FR")

ax2.set_xlabel("iterations")
ax1.set_ylabel(r"$\Vert \nabla_x f \Vert$")
ax2.set_ylabel(r"$\Vert \nabla_y f \Vert$")
ax1.grid()
ax2.grid()
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.legend()
plt.savefig("plots_pm/log_plots.png")


plt.show()
