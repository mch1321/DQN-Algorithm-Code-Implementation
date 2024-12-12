import torch
"""
Feel free to change these hyper-parameters!
TARGET_UPDATE: update rate of the target network
BATCH_SIZE: number of transitions sampled from the replay buffer
GAMMA: determines the importance of future rewards - low gamma may make agent short-sighted, high gamma makes agent overestimate long-term rewards
EPSILON_DECAY: controls rate of exponential decay of epsilon - a higher value means a slower decay
"""

DEVICE = torch.device('cpu')
SCREEN_WIDTH = 600
TARGET_UPDATE = 10
EPOCHS = 10000 #500 #100 #min 480 for heatmap
BATCH_SIZE = 64 #128 #32
GAMMA = 0.92 #0.999 #0.92
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 500 #200 #50
FIXED_EPSILON = 0.3
