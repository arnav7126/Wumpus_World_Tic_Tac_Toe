import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
class WumpusBayesianNetwork:
    def __init__(self, size):
        self.size = size
        self.model = BayesianNetwork()
        self.inference = None
        self.build_network()
        
    def build_network(self):
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                pit_node = f"Pit_{i}_{j}"
                breeze_node = f"Breeze_{i}_{j}"
                self.model.add_nodes_from([pit_node, breeze_node])

                for ni, nj in self.get_valid_neighbors(i, j):
                    neighbor_breeze = f"Breeze_{ni}_{nj}"
                    self.model.add_edge(pit_node, neighbor_breeze)
        
        self.add_cpds()
        
    def add_cpds(self):
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                pit_node = f"Pit_{i}_{j}"
                breeze_node = f"Breeze_{i}_{j}"
                if i == 1 and j == 1:
                    cpd_pit = TabularCPD(variable=pit_node, variable_card=2, values=[[1.0], [0.0]])
                else:
               
                    cpd_pit = TabularCPD(variable=pit_node, variable_card=2, values=[[0.8], [0.2]])
                self.model.add_cpds(cpd_pit)
                neighbors = [f"Pit_{ni}_{nj}" for ni, nj in self.get_valid_neighbors(i, j)]

                if neighbors:
                    num_neighbors = len(neighbors)
                    evidence_card = [2] * num_neighbors
                    parent_states = list(itertools.product([0, 1], repeat=num_neighbors))
                    breeze_values = [0 if sum(state) == 0 else 1 for state in parent_states]

                    cpd_breeze = TabularCPD(
                        variable=breeze_node, variable_card=2,
                        values=[[1 - v for v in breeze_values], breeze_values],
                        evidence=neighbors,
                        evidence_card=evidence_card
                    )
                else:
                    cpd_breeze = TabularCPD(variable=breeze_node, variable_card=2, values=[[0.5], [0.5]])

                self.model.add_cpds(cpd_breeze)

     
        if not self.model.check_model():
            raise ValueError("Bayesian model is inconsistent!")

        self.inference = VariableElimination(self.model)
        
    def get_valid_neighbors(self, i, j):
        """Get valid neighboring cells within grid bounds."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 1 <= ni <= self.size and 1 <= nj <= self.size:
                neighbors.append((ni, nj))
        return neighbors
    
    def infer_pit_probability(self, cell, evidence):
        if cell in evidence:
            return float(evidence[cell])  
        
        try:
           result = self.inference.query(variables=[cell], evidence=evidence)
           base_probability = result.values[1]  
           unknown_cells = []
           known_safe_count = 0
           total_pit_prob = 0
           for i in range(1, self.size + 1):
                for j in range(1, self.size + 1):

                    current_cell = f"Pit_{i}_{j}"
                    if cell == "Pit_1_1":
                        return self.inference.query(variables=[cell], evidence=evidence).values[1]
                    if current_cell in evidence:
                        if evidence[current_cell] == 0:  
                            known_safe_count += 1  
                    else:
                        unknown_cells.append(current_cell)

           num_unknowns = len(unknown_cells)

           if num_unknowns > 0:
                adjusted_probability = base_probability + (known_safe_count * 0.02)
                adjusted_probability = min(adjusted_probability, 0.9)  
           else:
                adjusted_probability = base_probability  

           return adjusted_probability  
        
        except Exception as e:
            print(f"Inference failed for {cell}: {e}")
            return None
class WumpusWorld:
    def __init__(self, size):
        self.size = size
        self.grid = [['' for _ in range(size)] for _ in range(size)]
        self.agent_position = (1, 1)  
        self.performance = 0
        self.init_world()

    def init_world(self):
        self.grid[self.size - 1][0] = 'V'  
        g_x, g_y = self.get_random_empty_cell()
        self.grid[g_x][g_y] = 'G'
        self.gold_position = (g_x + 1, g_y + 1)  
        w_x, w_y = self.get_random_empty_cell()
        self.grid[w_x][w_y] = 'W'

        pit_positions = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != (self.size - 1, 0) and random.random() < 0.2 and self.grid[i][j] == '':
                    self.grid[i][j] = 'P'
                    pit_positions.append((i, j))
        for x, y in pit_positions:
            self.add_adjacent_marker(x, y, 'B')
        self.add_adjacent_marker(w_x, w_y, 'S')
        start_x, start_y = self.size - 1, 0  
        adjacent_cells = [(start_x, 1), (start_x - 1, 0)]  
        has_breeze = any(self.grid[x][y] == 'P' for x, y in adjacent_cells if 0 <= x < self.size and 0 <= y < self.size)
        has_stench = any(self.grid[x][y] == 'W' for x, y in adjacent_cells if 0 <= x < self.size and 0 <= y < self.size)
        if has_breeze:
            self.grid[start_x][start_y] += 'B'
        if has_stench:
            self.grid[start_x][start_y] += 'S'


    def get_random_empty_cell(self):
        while True:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x][y] == '' and (x, y) != (self.size - 1, 0):
                return x, y

    def add_adjacent_marker(self, x, y, marker):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and 'V' not in self.grid[nx][ny]:
                if marker not in self.grid[nx][ny]:  
                    self.grid[nx][ny] += marker

    def get_cell_status(self, x, y):
        real_x, real_y = self.size - x, y - 1  
        cell = self.grid[real_x][real_y]
        return [
            'Breeze' if 'B' in cell else 'None',
            'Pit' if 'P' in cell else 'None',
            'Stench' if 'S' in cell else 'None',
            'Gold' if 'G' in cell else 'None',
            'Wumpus' if 'W' in cell else 'None'
        ]

    def move_agent(self, new_x, new_y):
        old_x, old_y = self.agent_position
        real_old_x, real_old_y = self.size - old_x, old_y - 1  
        real_new_x, real_new_y = self.size - new_x, new_y - 1
        self.grid[real_old_x][real_old_y] = self.grid[real_old_x][real_old_y].replace('V', '')
        self.agent_position = (new_x, new_y)
        self.grid[real_new_x][real_new_y] += 'V'
        self.performance -= 1  

    def display_world(self):
        for row in self.grid:
            print(row)


class Agent:

    MAX_LIVES=50
    def __init__(self,world, bayesian_network):
        self.position = (1, 1)  
        self.previous_position = (1, 1) 
        self.pit_falls = 0  
        self.wumpus_eaten = 0  
        self.visited = set()
        self.world = world
        self.bayesian_network = bayesian_network
        self.evidence = {}

    def perceive(self, world):
        x, y = self.position
        percepts = self.world.get_cell_status(x, y)
        if 'Breeze' in percepts:
            self.evidence[f"Breeze_{x}_{y}"] = 1
        else:
            self.evidence[f"Breeze_{x}_{y}"] = 0
            
        return percepts

    def decide_move(self, world_size,current_x, current_y, step):
        self.visited.add((current_x, current_y))
        neighbors = [(ni, nj) for ni, nj in self.bayesian_network.get_valid_neighbors(current_x, current_y) if (ni, nj) not in self.visited]
        if 'Pit' not in self.world.get_cell_status(current_x, current_y):
            self.evidence[f"Pit_{current_x}_{current_y}"] = 0  
        new_breeze_value = 1 if 'Breeze' in self.world.get_cell_status(current_x, current_y) else 0
        self.evidence[f"Breeze_{current_x}_{current_y}"] = new_breeze_value
        if new_breeze_value == 0:
                for ni, nj in self.bayesian_network.get_valid_neighbors(current_x, current_y):
                    self.evidence[f"Pit_{ni}_{nj}"] = 0 
        x, y = self.position
        possible_moves = []

        if x > 1: possible_moves.append((x - 1, y))  # Down
        if x < world_size: possible_moves.append((x + 1, y))  # Up
        if y > 1: possible_moves.append((x, y - 1))  # Left
        if y < world_size: possible_moves.append((x, y + 1))  # Right
        self.plot_pit_probability(step)
        return random.choice(possible_moves)
    
    
    def plot_pit_probability(self, step):
        pit_probabilities = np.zeros((self.world.size, self.world.size))
        for i in range(1, self.world.size + 1):
                for j in range(1, self.world.size + 1):
                    cell = f"Pit_{i}_{j}"
                    prob = self.bayesian_network.infer_pit_probability(cell, self.evidence)
                    pit_probabilities[self.world.size - i, j - 1] = prob if prob is not None else 0.5  

        plt.figure(figsize=(6, 6))
        sns.heatmap(pit_probabilities, annot=True, cmap="Reds", cbar=True, linewidths=0.5, vmin=0, vmax=1)

        plt.title(f"Pit Probability Heatmap - Step {step}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.xticks(np.arange(self.world.size) + 0.5, np.arange(1, self.world.size + 1))
        plt.yticks(np.arange(self.world.size) + 0.5, np.arange(self.world.size, 0, -1))  

        plt.savefig(f"heatmap_step_{step}.png")
        plt.close()  
    def update_position(self, new_x, new_y):
  
        self.previous_position = self.position  
        self.position = (new_x, new_y)

    def revert_position(self):
  
        self.position = self.previous_position
        self.pit_falls += 1 
        self.check_game_over()

    def respawn(self):
 
        self.position = (1, 1)
        self.wumpus_eaten += 1  
        self.check_game_over()

    def check_game_over(self):
        if self.pit_falls >= self.MAX_LIVES or self.wumpus_eaten >= self.MAX_LIVES:
            print("\nYou have crossed the maximum number of lives (50). Game Over!")
            print(f"Final Performance Score: {world.performance}")
            print(f"Agent fell into pits {self.pit_falls} times.")
            print(f"Agent was eaten by the Wumpus {self.wumpus_eaten} times.")
            exit()
        
if __name__ == "__main__":
    size = int(input("Enter the grid size (N x N): "))
    wumpus_network = WumpusBayesianNetwork(size)
    # Initialize world and agent
    world = WumpusWorld(size)
    agent = Agent(world, wumpus_network)
    current_x, current_y = 1, 1  
    step = 0
    # Start game loop
    while True:
        step += 1
        print("\n=== Current World State ===")
        world.display_world()

        status = agent.perceive(world)
        print(f"\nAgent at {agent.position} perceives: {status}")
        
        if 'Gold' in status:
            print("Agent collected the Gold! Exiting...")
            world.performance += 1000  
            print(f"Final Performance Score: {world.performance}")
            print(f"Agent fell into pits {agent.pit_falls} times.")
            print(f"Agent was eaten by the Wumpus {agent.wumpus_eaten} times.")
            break


        if 'Pit' in status:
            print("Agent fell into a Pit! Moving back to the previous position.")
            world.performance -= 1000  
            agent.revert_position()
            continue
        

        if 'Wumpus' in status:
            print("💀 Agent was eaten by the Wumpus! Respawning at (1,1)...")
            world.performance -= 1000  
            agent.respawn()
            continue

 
        new_x, new_y = agent.decide_move(world.size,current_x, current_y,step)
        print(f"Agent moves to: ({new_x}, {new_y})")

        current_x, current_y = new_x, new_y
        agent.update_position(new_x, new_y)
        world.move_agent(new_x, new_y)
        print(f"Performance Score: {world.performance}")
        print(f"Times fallen into pits: {agent.pit_falls}")
        print(f"Times eaten by Wumpus: {agent.wumpus_eaten}")