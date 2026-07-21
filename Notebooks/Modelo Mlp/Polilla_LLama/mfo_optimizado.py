# Notebooks/polilla_llama/mfo_optimizer.py
import numpy as np
import pickle
import math

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
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        moth_pos = np.random.uniform(lb, ub, (self.n_moths, self.dim))
        moth_fitness = np.zeros(self.n_moths)
        
        flame_pos = np.copy(moth_pos)
        flame_fitness = np.ones(self.n_moths) * np.inf
        
        for iteration in range(1, self.max_iter + 1):
            # Evaluar fitness de cada polilla
            for i in range(self.n_moths):
                # Mantener dentro de los límites
                moth_pos[i] = np.clip(moth_pos[i], lb, ub)
                moth_fitness[i] = self.obj_func(moth_pos[i])
            
            # Ordenar las polillas según fitness para convertirlas en llamas
            if iteration == 1:
                sort_idx = np.argsort(moth_fitness)
                flame_fitness = moth_fitness[sort_idx]
                flame_pos = moth_pos[sort_idx]
            else:
                # Combinar polillas y llamas anteriores y quedarse con los n_moths mejores
                double_pos = np.vstack((flame_pos, moth_pos))
                double_fitness = np.concatenate((flame_fitness, moth_fitness))
                
                sort_idx = np.argsort(double_fitness)
                flame_fitness = double_fitness[sort_idx][:self.n_moths]
                flame_pos = double_pos[sort_idx][:self.n_moths]
            
            # Actualizar el mejor global
            if flame_fitness[0] < self.best_score:
                self.best_score = flame_fitness[0]
                self.best_pos = np.copy(flame_pos[0])
            
            # Número de llamas a considerar (decrece con las iteraciones)
            flame_no = round(self.n_moths - iteration * ((self.n_moths - 1) / self.max_iter))
            
            a = -1 + iteration * ((-1) / self.max_iter) # Decrece linealmente de -1 a -2
            
            for i in range(self.n_moths):
                for j in range(self.dim):
                    # Elige a qué llama acercarse
                    if i < flame_no:
                        distance_to_flame = abs(flame_pos[i, j] - moth_pos[i, j])
                        # Ecuación de espiral logarítmica
                        t = (a - 1) * np.random.rand() + 1
                        moth_pos[i, j] = distance_to_flame * math.exp(1.5 * t) * math.cos(t * 2 * math.pi) + flame_pos[i, j]
                    else:
                        distance_to_flame = abs(flame_pos[flame_no - 1, j] - moth_pos[i, j])
                        t = (a - 1) * np.random.rand() + 1
                        moth_pos[i, j] = distance_to_flame * math.exp(1.5 * t) * math.cos(t * 2 * math.pi) + flame_pos[flame_no - 1, j]
            
            print(f"Iteración {iteration}/{self.max_iter} - Mejor Score: {self.best_score:.4f}")
            
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