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

## Running the Code ##
To have the project run, the CNN classifiers will need to be trained on images taken from the screen recorder. 
To start got to https://lishogi.org/editor. Once there, launch the screen_capture.py file
Next, fill the board up with on kind of piece and then click on the debugging menu and press d. Do this for all piece types, an empty board, and the side hands. 
Once the images are collected, move them into the correct template folder and run the CNNs

Next, download https://drive.google.com/file/d/1QpnprcMNT2RTVUrNOsrzDrU_Mi7Snesp/view?usp=drive_link. This is the fully trained model.
Now you can run launcher.py, and you'll have a working shoji engine. 
