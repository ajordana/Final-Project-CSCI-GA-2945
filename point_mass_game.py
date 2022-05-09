import numpy as np
import matplotlib.pyplot as plt
from gda import GDA
from test_examples.point_mass import point_mass


f = point_mass()
x0 = np.zeros(f.fx.shape)
y0 = np.zeros(f.fy.shape)

eta_x = 1e-4
eta_y = 1e-4

T = 1e5

x_GDA, y_GDA, x_grad, y_grad = GDA(f, x0, y0, eta_x, eta_y, T)

u = x_GDA[-1]
s = y_GDA[-1]

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


fig, (ax1, ax2) = plt.subplots(2, 1)
for t in range(f.T):
    ax1.plot(np.array([t, t+1]), np.array([states[t][0], next_states[t][0]]), 'black')
    ax2.plot(np.array([t, t+1]), np.array([states[t][1], next_states[t][1]]), 'black')
ax1.set_xlabel("x")
ax2.set_ylabel("y")
ax1.grid()
ax2.grid()

plt.figure()
plt.plot(np.array(controls)[:, 0], label="u_x")
plt.plot(np.array(controls)[:, 1], label="u_y")
plt.xlabel("Time")
plt.ylabel("Control")
plt.grid()
plt.title("Control inputs")
plt.legend()



plt.figure()
plt.plot(np.array(x_grad), label="x grad")
plt.plot(np.array(y_grad), label="y grad")
plt.grid()
plt.legend()



plt.show()

