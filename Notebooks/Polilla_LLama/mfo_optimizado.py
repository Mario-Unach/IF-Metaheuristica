# Notebooks/polilla_llama/mfo_optimizer.py
import numpy as np
import pickle

class MothFlameOptimizer:
    def __init__(self, objective_function, dim=4, n_moths=15, max_iter=20):
        self.obj_func = objective_function
        self.dim = dim
        self.n_moths = n_moths
        self.max_iter = max_iter
        self.best_pos = None
        self.best_score = np.inf

    def optimize(self, bounds):
        # bounds es una lista de tuplas [(min, max), ...] para cada dimensión
        moth_pos = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], (self.n_moths, self.dim)
        )
        
        for iteration in range(self.max_iter):
            for i in range(self.n_moths):
                fitness = self.obj_func(moth_pos[i])
                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_pos = np.copy(moth_pos[i])
            
            # Aquí va la lógica de vuelo en espiral (vuelo hacia las llamas)
            # ...
        return self.best_pos, self.best_score

    def save_state(self, filepath):
        """Guarda los mejores pesos/parámetros encontrados"""
        state = {'best_pos': self.best_pos, 'best_score': self.best_score}
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.best_pos = state['best_pos']
            self.best_score = state['best_score']