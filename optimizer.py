"""
Minimal SGD optimizer for the RBM in ``my_RBM_tf2_test.py``.

NOTE: the original ``optimizer.py`` was not part of the uploaded sources, so this
file reconstructs the interface that ``RBM.train`` / ``data_collection_a*.py``
expect:

    optimus = Optimizer(machine, learning_rate)
    ...
    optimus.fit()          # consumes machine.grad_dict, updates machine.model_dict

``RBM.parallel_cd`` / ``RBM.contr_divergence`` return the CD *ascent* direction
for the log-likelihood, so the update is a plus sign:

    theta <- theta + lr * grad
"""


class Optimizer:
    def __init__(self, machine, learning_rate=0.1, momentum=0.0):
        self.machine = machine
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity = {}

    def fit(self):
        grads = self.machine.grad_dict
        for key, value in self.machine.model_dict.items():
            g = grads[key]
            if self.momentum:
                v = self._velocity.get(key, 0.0)
                v = self.momentum * v + g
                self._velocity[key] = v
                g = v
            self.machine.model_dict[key] = value + self.learning_rate * g
        self.machine.update_model()
