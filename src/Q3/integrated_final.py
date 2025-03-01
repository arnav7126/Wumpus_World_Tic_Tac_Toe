import time
import random
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Import Tic-Tac-Toe functions
from tic_tac_toe_only_llm_final import play_game
# Import Wumpus World classes
from wumpus_final import WumpusWorld, WumpusBayesianNetwork, Agent


def main():
    # Initialize Wumpus World and Bayesian Network
    grid_size = int(input("Enter the grid size for Wumpus World (N x N): "))
    wumpus_network = WumpusBayesianNetwork(grid_size)
    world = WumpusWorld(grid_size)
    agent = Agent(world, wumpus_network)
    
    # Initialize agent position
    current_x, current_y = 1, 1  
    step = 0
    
    # Run Tic-Tac-Toe + Wumpus World Integration
    for trial in range(50):
        print(f"\nTrial {trial + 1}: Playing Tic-Tac-Toe\n")
        result = play_game(3, mode="LLM_vs_LLM")  # Play one Tic-Tac-Toe trial (3x3 board)
        if result == "Gemini":
            print("\nGemini won! Agent making a best move in Wumpus World...")
            step += 1
            world.display_world()
            status = agent.perceive(world)
            print(f"\nAgent at {agent.position} perceives: {status}")
            
            if 'Gold' in status:
                print("Agent collected the Gold! Exiting...")
                world.performance += 1000
                break
            if 'Pit' in status:
                print("Agent fell into a Pit! Game Over.")
                world.performance -= 1000
                break
            if 'Wumpus' in status:
                print("Agent was eaten by the Wumpus! Game Over.")
                world.performance -= 1000
                break
            
            # Decide next move based on Bayesian inference
            new_x, new_y = agent.decide_move(world.size, current_x, current_y, step)
            print(f"Agent moves to: ({new_x}, {new_y})")
            
            agent.update_position(new_x, new_y)
            world.move_agent(new_x, new_y)
            current_x, current_y = new_x, new_y
            
            print(f"Performance Score: {world.performance}")
        else:
            print("\nGroq won! Agent making a random move in Wumpus World...")
            step += 1
            world.display_world()
            status = agent.perceive(world)
            print(f"\nAgent at {agent.position} perceives: {status}")
            
            if 'Gold' in status:
                print("Agent collected the Gold! Exiting...")
                world.performance += 1000
                print(f"Performance Score: {world.performance}")
                break
            if 'Pit' in status:
                print("Agent fell into a Pit! Game Over.")
                world.performance -= 1000
                print(f"Performance Score: {world.performance}")
                break
            if 'Wumpus' in status:
                print("Agent was eaten by the Wumpus! Game Over.")
                world.performance -= 1000
                print(f"Performance Score: {world.performance}")
                break
            
            # Decide next move based on Bayesian inference
            new_x, new_y = agent.decide_random_move(world.size, current_x, current_y, step)
            print(f"Agent moves to: ({new_x}, {new_y})")
            
            agent.update_position(new_x, new_y)
            world.move_agent(new_x, new_y)
            current_x, current_y = new_x, new_y
            
            print(f"Performance Score: {world.performance}")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
