import numpy as np


class point_mass_model:
    def __init__(self):
        self.T = 10
        self.nx = 4
        self.nu = 2
        self.dt = 0.1
        self.x0 = np.zeros(4)
        self.Q = 0.1 * np.diag(np.array([1, 1, 0.01, 0.01]))
        self.R =  1e-1 * np.diag(np.array([1, 1]))
        self.target = np.array([1, 0, 0, 0])
        self.ter_coeff = 200
        self.P = 1e-3 * np.eye(self.nx)
        self.Pinv = 1e3 * np.eye(self.nx)

    def dyn(self, s, u):
        dpx = s[2]
        dpy = s[3]
        dvx = u[0]
        dvy = u[1] 
        return s + np.array([dpx, dpy, dvx, dvy]) * self.dt

    def dynx(self, s, u):
        D = np.zeros((self.nx, self.nx))
        D[0, 2] = 1
        D[1, 3] = 1
        return np.eye(self.nx) + D * self.dt

    def dynu(self, s, u):
        D = np.zeros((self.nx, self.nu))
        D[2, 0] = 1
        D[3, 1] = 1
        return D * self.dt

    def cost(self, s, u=None):
        cost = 0.1 / ((0.1 * s[1] + 1.0) ** 10)
        if u is None:
            s_ = s - self.target
            return 0.5 * self.ter_coeff * np.vdot(s_, self.Q @ s_)
        else:
            s_ = s - self.target
            return cost + 0.5 * np.vdot(s_, self.Q @ s_) + 0.5 * np.vdot(u, self.R @ u)

    def costx(self, s, u=None):
        lx = np.array([0, -0.1 / (1 + 0.1 * s[1]) ** 11, 0, 0]) 
        if u is None:
            return self.ter_coeff * self.Q @ (s - self.target)
        else:
            return lx + self.Q @ (s - self.target)

    def costxx(self, s, u=None):
        lxx = np.zeros((self.nx, self.nx))
        lxx[1, 1] =  0.11 / (0.1 * s[1] + 1.0) ** 12 
        if u is None:
            return self.ter_coeff * self.Q
        else:
            return lxx + self.Q

    def costu(self, s, u):
        return self.R @ u

    def costuu(self, s, u):
        return self.R


class point_mass(point_mass_model):
    def __init__(self):
        point_mass_model.__init__(self)

        self.allocate_data()

    def calc(self, x, y):
        self.f = self.eval(x, y)
        self.eval_grad(x, y)
        self.eval_hessians(x, y)
        print(np.linalg.norm(self.fx))
        print(np.linalg.norm(self.fy))
        # self.fxx = "not implemented"
        # self.fyy = "not implemented"
        # self.fyx = "not implemented"
        self.fxy = self.fyx

    def eval_hessians(self, u, s):
        states = [self.x0] + [s[i * self.nx : (i + 1) * self.nx] for i in range(self.T)]
        controls = [u[i * self.nu : (i + 1) * self.nu] for i in range(self.T)]
        w_list = [
            states[t + 1] - self.dyn(states[t], controls[t]) for t in range(self.T)
        ]
        dinx_list = [self.dynx(states[t], controls[t]) for t in range(self.T)]
        dinu_list = [self.dynu(states[t], controls[t]) for t in range(self.T)]

        # state * control
        for t in range(self.T):
            if t > 0:
                self.fyx[(t-1) * self.nx : t * self.nx, t * self.nu : (t + 1) * self.nu] = - dinx_list[t].T @ self.Pinv @ dinu_list[t]
            self.fyx[t * self.nx : (t + 1) * self.nx, t * self.nu : (t + 1) * self.nu] =  self.Pinv @ dinu_list[t]

        # state * state
        for t in range(1, self.T):            
            self.fyy[(t-1) * self.nx : t * self.nx, (t-1) * self.nx : t * self.nx] = self.costxx(states[t], controls[t]) - self.Pinv - dinx_list[t].T @ self.Pinv @ dinx_list[t]

            self.fyy[(t-1) * self.nx : t * self.nx, t * self.nx : (t+1) * self.nx] = dinx_list[t].T @ self.Pinv 
            self.fyy[t * self.nx : (t+1) * self.nx, (t-1) * self.nx : t * self.nx] = self.Pinv @ dinx_list[t]

        self.fyy[-self.nx :, -self.nx :] = self.costxx(states[-1]) - self.Pinv 

    def eval_grad(self, u, s):
        states = [self.x0] + [s[i * self.nx : (i + 1) * self.nx] for i in range(self.T)]
        controls = [u[i * self.nu : (i + 1) * self.nu] for i in range(self.T)]
        w_list = [
            states[t + 1] - self.dyn(states[t], controls[t]) for t in range(self.T)
        ]
        dinx_list = [self.dynx(states[t], controls[t]) for t in range(self.T)]

        for t in range(self.T):
            self.fx[t * self.nu : (t + 1) * self.nu] = (
                self.costu(states[t], controls[t])
                + self.dynu(states[t], controls[t]).T @ self.Pinv @ w_list[t]
            )

        for t in range(1, self.T):
            self.fy[(t - 1) * self.nx : t * self.nx] = self.costx(states[t], controls[t])+ dinx_list[t].T @ self.Pinv @ w_list[t] - self.Pinv @ w_list[t - 1]
            
        self.fy[-self.nx:] = self.costx(states[-1]) - self.Pinv @ w_list[-1]

    def eval(self, u, s):
        states = [self.x0] + [s[i * self.nx : (i + 1) * self.nx] for i in range(self.T)]
        controls = [u[i * self.nu : (i + 1) * self.nu] for i in range(self.T)]
        J = 0
        for t in range(self.T):
            J += self.cost(states[t], controls[t])
            w = states[t + 1] - self.dyn(states[t], controls[t])
            J += -np.vdot(w, self.Pinv @ w)
        J += self.cost(states[-1])
        return J

    def allocate_data(self):
        self.f = 0
        self.fx = np.zeros(self.nu * self.T)
        self.fy = np.zeros(self.nx * self.T)
        self.fxx = np.zeros((self.nu * self.T, self.nu * self.T))
        self.fyy = np.zeros((self.nx * self.T, self.nx * self.T))
        self.fyx = np.zeros((self.nx * self.T, self.nu * self.T))


if __name__ == "__main__":
    pm = point_mass()
