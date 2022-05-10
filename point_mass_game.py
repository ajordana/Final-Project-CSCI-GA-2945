import numpy as np
import matplotlib.pyplot as plt
from gda import GDA, FR
from test_examples.point_mass import point_mass


f = point_mass()
x0 = np.zeros(f.fx.shape)
y0 = np.zeros(f.fy.shape)
p = np.linspace(0, f.target[0], f.T)
for i in range(f.T):
    y0[4*i] = p[i]

eta_x = 2e-4
eta_y = 2e-4

T = 1e5

# x_GDA, y_GDA, x_grad, y_grad = GDA(f, x0, y0, eta_x, eta_y, T)
u, s, x_grad, y_grad = GDA(f, x0, y0, eta_x, eta_y, T, reccord=False)


states = [f.x0] + [s[i * f.nx : (i + 1) * f.nx] for i in range(f.T)]
controls = [u[i * f.nu : (i + 1) * f.nu] for i in range(f.T)]
next_states = [f.dyn(states[i], controls[i])for i in range(f.T)]

print(states)

plt.figure()
for t in range(f.T):
    plt.plot(np.array([states[t][0], next_states[t][0]]), np.array([states[t][1], next_states[t][1]]), 'black')
# plt.plot(np.array(states)[:, 0], np.array(states)[:, 1], label="DDP Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.title("Position trajectory")


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
for t in range(f.T):
    ax1.plot(np.array([t, t+1]), np.array([states[t][0], next_states[t][0]]), 'black')
    ax2.plot(np.array([t, t+1]), np.array([states[t][1], next_states[t][1]]), 'black')
    ax3.plot(np.array([t, t+1]), np.array([states[t][2], next_states[t][2]]), 'black')
    ax4.plot(np.array([t, t+1]), np.array([states[t][3], next_states[t][3]]), 'black')
ax1.set_ylabel("$p_x$")
ax2.set_ylabel("$p_y$")
ax3.set_ylabel("$v_x$")
ax4.set_ylabel("$v_y$")
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(controls)[:, 0], label="u_x")
ax2.plot(np.array(controls)[:, 1], label="u_y")
ax1.grid()
ax2.grid()
ax2.set_xlabel("Time")
ax1.set_ylabel("$u_x$")
ax2.set_ylabel("$u_y$")



fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(x_grad))
ax2.plot(np.array(y_grad))


ax2.set_xlabel("iterations")
ax1.set_ylabel("x grad")
ax2.set_ylabel("y grad")
ax1.grid()
ax2.grid()
ax1.set_yscale('log')
ax2.set_yscale('log')



plt.show()


