# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers[::2] # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            
            # TODO: calculate updates for bias

            # TODO: Perform weight and bias updates with weight decay


            # TODO: Calculate updates for weight
            grad_W = layer.dLdW
            # grad_W += self.weight_decay * layer.W  # Apply weight decay

            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * grad_W
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (grad_W ** 2)

            m_hat_W = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[layer_id] / (1 - self.beta2 ** self.t)
            
            # TODO: calculate updates for bias
            grad_b = layer.dLdb
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * grad_b
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (grad_b ** 2)

            m_hat_b = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

            # TODO: Perform weight and bias updates

            W_t_1 = layer.W.copy()
            b_t_1 = layer.b.copy()

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

            layer.W -= W_t_1 * self.lr * self.weight_decay
            layer.b -= b_t_1 * self.lr * self.weight_decay


