import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        # Calculate reset gate (r), update gate (z), nd new candidate hidden state (n)
        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh ))

        # Calculate current hidden state (h_t)
        h_t = (1 - self.z) * self.n + self.z * h_prev_t 
        self.h_t = h_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        x = self.x.reshape(self.d, 1)
        hidden = self.hidden.reshape(self.h, 1)
        # #    when computing self.dWs...
        # # 2) Transpose all calculated dWs...
        # # 3) Compute all of the derivatives
        # # 4) Know that the autograder grades the gradients in a certain order, and the
        # #    local autograder will tell you which gradient you are currently failing.

        # # ADDITIONAL TIP:
        # # Make sure the shapes of the calculated dWs and dbs  match the
        # # initalized shapes accordingly

        dL_dz =  self.z_act.backward(delta * (self.hidden - self.n))
        dL_dn = self.h_act.backward(delta * (1-self.z))
        dL_dr = (self.Wnh @ self.hidden + self.bnh) * self.r_act.backward(dL_dn)
    
        self.dWrx = (x @ dL_dr).T
        self.dWzx = (x @ dL_dz).T
 
        self.dWnx = (x @ dL_dn).T
        self.dWrh = (hidden @ dL_dr).T
        self.dWzh = (hidden @ dL_dz).T
        self.dWnh =  (hidden @ (dL_dn * self.r)).T
        self.dbrx = dL_dr.ravel()
        self.dbzx = dL_dz.ravel()
        self.dbnx = dL_dn.ravel()
        self.dbrh = dL_dr.ravel()
        self.dbzh = dL_dz.ravel()
        self.dbnh = dL_dn * self.r

        dx = dL_dr @ self.Wrx + dL_dn @ self.Wnx + dL_dz @ self.Wzx
        dh_prev_t = dL_dr @ self.Wrh + dL_dz @ self.Wzh + np.multiply(delta, self.z) + (dL_dn * self.r) @ self.Wnh

        dx = dx.reshape(1, self.d)
        dh_prev_t = dh_prev_t.reshape(1, self.h)

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
