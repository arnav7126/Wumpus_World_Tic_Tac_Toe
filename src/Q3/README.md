Merging of Tic_Tac_Toe & Wumpus World

1)Overview

In this, we are merging the 2 questions as follows:

If LLM1 wins the tic_tac_toe match, then the agent makes the best move. However, if LLM2 wins the tic_tac_toe match, then the agent makes a random move. At each move, the probability heat map is generated and saved in the same folder as where you are running the file.

A sample run is shown below:

![alt text](<Screenshot 2025-03-01 at 11.55.36 PM.png>)

Here, after the first match is over, LLM2 has won, hence the agent makes a random move in the wumpus world.

2)Running the Code

To run the merged code, type the following command:

    python3 integrated_final.py
