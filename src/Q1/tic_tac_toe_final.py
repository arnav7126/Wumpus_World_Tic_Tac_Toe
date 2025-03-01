import google.generativeai as genai
import anthropic 
import openai 
from llama_cpp import Llama
import requests
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
                #####API KEYS USED< GEMINI AND GROQ>
GROQ_API_KEY = "<insert-api-key-here>"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def initializeBoard(n):
    board = [[" " for _ in range(n)] for _ in range(n)]
    return board

######### GEMINI API MOVES CODE THAT CAN BE REPLACED AS MENTIONED IN THE README###


def get_gemini_moves(board, last_move, gemini_history,max_attempts=3,delay=20):         ####WE are feeding (valid moves) and previous move of opponent LLM as natural language to the LLM
     valid_moves = [(i, j) for i in range(len(board)) for j in range(len(board)) if board[i][j] == " "]
     if last_move is not None:
        first_element = last_move[0]
        second_element = last_move[1]
        last_move_str = f"{first_element},{second_element}"
     else:
        first_element = "N/A,opponent hasnt played a move yet"
        second_element = "N/A,opponent hasnt played a move yet"
        last_move_str = "None"
     history_str = ", ".join([f"({r},{c})" for r, c in gemini_history]) if gemini_history else "None" 
     prompt_given_to_gemini = (
        "### Tic-Tac-Toe Game - Your Turn ###\n\n"
        "Current Board State:\n"
        f"{board}\n\n"
        f"Last move played: {last_move}\n\n"
        "You are playing as 'O'. Your goal is to win the game or block your opponent ('X').\n\n"
        f"all the moves played by you is:{gemini_history}, now play the move such that it is included in {valid_moves}, and it makes order such that- 1,1 2,2 .... till n,n or for any number x the order x,1 x,2 till x,n and 1,x 2,x till n,x"
        "### Rules:\n"
        f"1. Play inside the dimensions of the board only and win immediately if possible. Last move played by opponent LLM is on row number {first_element} and column number {second_element}\n"
        "2. The board is represented as a 2D grid, where each cell is either empty (' '), occupied by an 'X', or occupied by an 'O'.\n"
        "3. The goal is to get your marks in a row, column, or diagonal before your opponent does.\n"
        "4. Respond with a valid move in 'row,column' format (e.g., '1,2').\n"
        "5. DO NOT include explanations, extra text, greetings, or comments in your response.\n"
        "6. You MUST choose an empty spot. If all spots are taken, respond with 'None'.\n"
        "7. If you make an invalid move, your turn will be retried.\n"
        "8. Prioritize winning moves. If no winning move is available, block your opponent.\n\n"
        f"### Allowed Moves: {valid_moves}\n"
        "You can only play one of these valid moves.\n\n"
        "### Valid Example Moves:\n"
        " - '0,2' (if that cell is empty)\n"
        " - '2,1' (if that cell is empty)\n"
        " - 'None' (only if no valid moves remain)\n\n"
        "WARNING: If you respond incorrectly, your turn will be retried.\n\n"
        "Now, enter your move:"
    )
     headers = {"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type": "application/json"}
     payload = {"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt_given_to_gemini}],"max_tokens": 50}
     attempts = 0
     while attempts < max_attempts:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                move = data["choices"][0]["message"]["content"].strip()
                return move
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Resource exhausted error encountered (Groq). Retrying in {delay} seconds (attempt {attempts+1}/{max_attempts})...")
                time.sleep(delay)
                attempts += 1
            else:
                raise
     raise Exception("Max retries reached. Unable to get a move from Groq.")    ####API limit permanently reached, gotta change API code

#####################################################################################################################################3



def get_groq_moves(board, last_move,groq_history,max_attempts=3,delay=20):
    valid_moves = [(i, j) for i in range(len(board)) for j in range(len(board)) if board[i][j] == " "]
    if last_move is not None:
        first_element = last_move[0]
        second_element = last_move[1]
        last_move_str = f"{first_element},{second_element}"
    else:
        first_element = "N/A,opponent hasnt played a move yet"
        second_element = "N/A,opponent hasnt played a move yet"
        last_move_str = "None"
    history_str = ", ".join([f"({r},{c})" for r, c in groq_history]) if groq_history else "None" 
    prompt_given_to_groq = (
        "### Tic-Tac-Toe Game - Your Turn ###\n\n"
        "Current Board State:\n"
        f"{board}\n\n"
        f"Last move played: {last_move}\n\n"
        "You are playing as 'O'. Your goal is to win the game or block your opponent ('X').\n\n"
        f"all the moves played by you is:{groq_history}, now play the move such that it is included in {valid_moves}, and it makes order such that- 1,1 2,2 .... till n,n or for any number x the order x,1 x,2 till x,n and 1,x 2,x till n,x"
        "### Rules:\n"
        f"1. Play inside the dimensions of the board only and win immediately if possible. Last move played by opponent LLM is on row number {first_element} and column number {second_element}\n"
        "2. The board is represented as a 2D grid, where each cell is either empty (' '), occupied by an 'X', or occupied by an 'O'.\n"
        "3. The goal is to get your marks in a row, column, or diagonal before your opponent does.\n"
        "4. Respond with a valid move in 'row,column' format (e.g., '1,2').\n"
        "5. DO NOT include explanations, extra text, greetings, or comments in your response.\n"
        "6. You MUST choose an empty spot. If all spots are taken, respond with 'None'.\n"
        "7. If you make an invalid move, your turn will be retried.\n"
        "8. Prioritize winning moves. If no winning move is available, block your opponent.\n\n"
        f"### Allowed Moves: {valid_moves}\n"
        "You can only play one of these valid moves.\n\n"
        "### Valid Example Moves:\n"
        " - '0,2' (if that cell is empty)\n"
        " - '2,1' (if that cell is empty)\n"
        " - 'None' (only if no valid moves remain)\n\n"
        "WARNING: If you respond incorrectly, your turn will be retried.\n\n"
        "Now, enter your move:"
    )
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type": "application/json"}
    payload = {"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt_given_to_groq}],"max_tokens": 50}
    attempts = 0
    while attempts < max_attempts:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                move = data["choices"][0]["message"]["content"].strip()
                return move
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Resource exhausted error encountered (Groq). Retrying in {delay} seconds (attempt {attempts+1}/{max_attempts})...")
                time.sleep(delay)
                attempts += 1
            else:
                raise
    raise Exception("Max retries reached. Unable to get a move from Groq.")

def get_human_move(board, last_move, marker):                    ### when human is playing the game we can get input our move in the valid format here
    valid_moves = [(i, j) for i in range(len(board)) for j in range(len(board)) if board[i][j] == " "]
    print("\nCurrent Board State:")
    print_board(board)
    print(f"Last move played: {last_move}")
    print(f"You are playing as '{marker}'.")
    print("Valid moves:", valid_moves)
    move = input("Enter your move in 'row,column' format: ")
    return move

def print_board(board):      ####for printing the board
    for row in board:
        print(" | ".join(row))
        print("-" * (4 * len(board) - 1))

def is_valid_move(board, row, col):               ###basic rules for validity of a tic tac toe move .... empty space and respect dimensions of board
    return 0 <= row < len(board) and 0 <= col < len(board) and board[row][col] == " "

def check_winner(board):          ##chcking winner board condition
    n = len(board)
    for i in range(n):
        if all(board[i][0] != " " and board[i][j] == board[i][0] for j in range(n)):
            return board[i][0]
        if all(board[0][i] != " " and board[j][i] == board[0][i] for j in range(n)):
            return board[0][i]
    if all(board[0][0] != " " and board[i][i] == board[0][0] for i in range(n)):
        return board[0][0]
    if all(board[0][n-1] != " " and board[i][n-1-i] == board[0][n-1] for i in range(n)):
        return board[0][n-1]
    return None  

def play_game(n, mode="LLM_vs_LLM", human_marker=None):          ####whole logic of playing the game
    board = initializeBoard(n)
    print_board(board)
    if mode == "LLM_vs_LLM":
        players = [("Gemini", "X", "LLM"), ("Groq", "O", "LLM")]
    elif mode == "LLM_vs_Human":
        if human_marker == "X":
            players = [("Human", "X", "Human"), ("Groq", "O", "LLM")]
        else:
            players = [("Gemini", "X", "LLM"), ("Human", "O", "Human")]
    else:
        print("Invalid mode selected.")
        return None
    gemini_history = []
    groq_history = []
    last_move = None
    for turn in range(n * n):
        player_name, marker, p_type = players[turn % 2]
        print(f"{player_name}'s Turn ({marker})\n")
        while True:
            if p_type == "LLM":
                if player_name == "Gemini":
                    move = get_gemini_moves(board, last_move,gemini_history,max_attempts=3,delay=20)
                else: 
                    move = get_groq_moves(board, last_move,groq_history,max_attempts=3,delay=20)
            else:
                move = get_human_move(board, last_move, marker)
            try:
                row, col = map(int, move.split(","))
                if is_valid_move(board, row, col):
                    board[row][col] = marker
                    last_move = (row, col)
                    
                    if p_type == "LLM":
                        if player_name == "Gemini":
                            gemini_history.append((row, col))
                        else:
                            groq_history.append((row, col))
                    break

                else:
                    print("Invalid move received! Try again.")
            except Exception as e:
                print("Error in move format! Try again.", e)

        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"{player_name} ({winner}) wins, letsgooooo!\n")
            return player_name 

    print("It's a draw, but groq deserves a win coz it held on even after starting second(fair enough ig)")
    return "Draw"
def run_experiments(n, trials=50):
    outcomes = {"Gemini": 0, "Groq": 0}

    for i in range(trials):
        print(f"Running game {i+1} of {trials}...\n")
        result = play_game(n, mode="LLM_vs_LLM")            ##starts playing <LLM,LLM> games

        if result == "Gemini":
            outcomes["Gemini"] += 1                        
        else:
            outcomes["Groq"] += 1                        ####draw condition is counted as win for LLM2 because it had a disadvantage of starting second

        if (i + 1) % 3 == 0:
            print("Pausing for 20 seconds to avoid overloading errors :)))...\n")    ###too many API errors were happening while running large number of trials , so delays are necessary
            time.sleep(20)

    with open("Exercise1.json", "w") as f:                       ####getting the json file of the outcomes
        json.dump(outcomes, f, indent=4)

def load_results(filename="Exercise1.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found. Using sample data.")
        return {"Gemini": 20, "Groq": 30}

    
def create_binomial_distribution_chart(results, trials=50):
    player1_wins = results["Gemini"]
    player2_wins = results["Groq"]
    total_games = player1_wins + results["Groq"]
    observed_probability = player1_wins / total_games if total_games > 0 else 0.5
   
    fair_probability = 0.5
    x = np.arange(0, trials + 1)
    binomial_pmf = scipy.stats.binom.pmf(x, trials, fair_probability)
  
    plt.figure(figsize=(10, 6))
    
    plt.bar(x, binomial_pmf, color='aqua', alpha=0.7, label='Expected Distribution')
    
    plt.axvline(x=player1_wins, color='red', linestyle='-.', 
              label=f'Observed wins (Player 1): {player1_wins}')
    plt.axvline(x=player2_wins, color='red', linestyle='--', 
              label=f'Observed wins (Player 2): {player2_wins}')
    
    plt.xlabel('Number of Player 1 Wins')
    plt.ylabel('Probability')
    plt.title('Binomial Distribution of Tic Tac Toe Outcomes')
    plt.legend()
   
    textbox_content = (f'Total valid games: {total_games}\n'
                      f'Player 1 wins: {player1_wins}\n'
                      f'Player 2 wins: {player2_wins}\n'
                      f'Observed probability: {observed_probability:.3f}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.75, 0.95, textbox_content, transform=plt.gca().transAxes,
           fontsize=10, verticalalignment='top', bbox=props)
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("tic_tac_toe_binomial_distribution.png")
    plt.show()


if __name__ == "__main__":
    mode_input = input("Select game mode:\n1 - LLM vs LLM\n2 - LLM vs Human\nEnter 1 or 2: ").strip()
    board_size = int(input("Enter board size (e.g., 3 for 3x3): "))
    if mode_input == "1":
        run_experiments(board_size, trials=50)
        
        results = load_results()
        
        create_binomial_distribution_chart(results, trials=50)
    elif mode_input == "2":
        human_marker = input("Choose your marker ('X' or 'O'): ").strip().upper()
        play_game(board_size, mode="LLM_vs_Human", human_marker=human_marker)
    else:
        print("Invalid mode selection.")