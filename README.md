# 4200TermProject

1. Data Acquisition & Preprocessing
Source: Thousands of high-level competitive matches scrubbed from LeagueShogi.

Cleaning: Raw logs are parsed and converted into standardized SFEN (Shogi Forsyth-Edwards Notation).

Purpose: This dataset provides the "ground truth" for the CNN to recognize board states and initializes the policy network with human-like intuition.

2. CNN Training (Vision)
The vision system is trained using two specialized models to ensure high-speed inference during live games:

Occupancy Model: A binary classifier (trainOccupancy.py) that determines if a square contains a piece.

Classification Model: A multi-class CNN (trainCNN.py) that identifies the specific piece type and its promotion status.

Dataset: Uses detection_debug_monitor_1.png and thousands of augmented screen-captured samples to handle different Shogi clients and resolutions.

3. Self-Play & MCTS (The Engine)
Once the models understand the board, the engine improves through autonomous play:

MCTS (Monte Carlo Tree Search): Used in selfPlayTrain.py to explore the game tree.

Reinforcement Learning: The agent plays against versions of itself, using the scrubbed league data as a baseline to avoid "random" early-stage play.

Move Encoding: High-efficiency conversion of board states into tensors via moveEncoding.py to maximize training throughput.
