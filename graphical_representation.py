#graphical representation of results

import matplotlib.pyplot as plt
import environment
import numpy as np
import seaborn as sns
import torch


def sum_durations(durations):
    """
    Calculates the sum of total durations.
    Parameters:
    durations (list): A list containing the durations (same as rewards) per episode.
    """
    sum = 0
    for num in durations:
        sum += num
    print("SUM: ", sum)
    

def learning_curve(durations, runs=10):
    """
    Plots the learning curve of the agent.
    Parameters:
    durations (list): A list containing the durations (same as rewards) per episode.
    """
    # Moving average calculation
    def moving_average(data, window_size):
        cumulative_sum = [0] + [sum(data[:i]) for i in range(1, len(data) + 1)]
        return [(cumulative_sum[i + window_size] - cumulative_sum[i]) / window_size for i in range(len(data) - window_size + 1)]
    
    # Plot the durations
    plt.figure(figsize=(10, 6))
    plt.plot(durations, label='Episode Duration')

    # Plot the moving average
    if len(durations) > 10:
        ma = moving_average(durations, window_size=10)
        plt.plot(range(10, len(durations) + 1), ma, label='Moving Average (window=10)', color='red')
    
    # Labels and title
    plt.title("Learning Curve")
    plt.xlabel("Training epochs")
    plt.ylabel("Duration")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def loss_function(losses):
    """
    Plots the loss function.
    Parameters:
    losses (list): A list containing output of loss function for all time instances.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses)
    plt.title("Loss Function")
    plt.xlabel("Time steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def Q_values(q_values_total):
    """
    Plots the Q-values over episodes.
    Parameters:
    q_values_total (list): A list of lists, where each inner list contains Q-values for a single episode.
    """
    max_q_values = [max(np.max(q_values) for q_values in episode) for episode in q_values_total]
    min_q_values = [min(np.min(q_values) for q_values in episode) for episode in q_values_total]
    mean_q_values = [np.mean([np.mean(q_values) for q_values in episode]) for episode in q_values_total]

    plt.figure(figsize=(10, 6))
    episodes = range(len(q_values_total))
    plt.plot(episodes, max_q_values, label='Max Q-Value', color='green')
    plt.plot(episodes, min_q_values, label='Min Q-Value', color='red')
    plt.plot(episodes, mean_q_values, label='Mean Q-Value', color='blue')
    plt.fill_between(episodes, min_q_values, max_q_values, color='gray', alpha=0.2, label='Q-Value Range')
    plt.title("Q-Values Over Episodes")
    plt.xlabel("Training epochs")
    plt.ylabel("Q-Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def action_distribution(actions):
    """
    Plots the frequency of actions taken by the agent.
    Parameters:
    actions (list): A list of actions taken by the agent over all epochs.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=actions, palette='Set2')
    plt.title("Action Distribution")
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def cp_pa_visualisation(cart_positions, pole_angles):
    """
    Plots the cart position and pole angle during an episode.
    Parameters:
    cart_positions (list): A list containing positions of the cart along an episode.
    pole_angles (list): A list containing the angles of the pole along an episode.
    """
    # Generate a range of episodes
    episodes = range(len(cart_positions))


    # Plot cart positions and pole angles
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, cart_positions, label='Cart Position', alpha=0.8)
    plt.plot(episodes, pole_angles, label='Pole Angle', alpha=0.8)

    plt.title('Cart-Pole: Position and Angle Over Time')
    plt.xlabel('Epoch Time Steps')
    plt.ylabel('Environment State')
    plt.legend()
    plt.grid(True)
    plt.show()


def cart_pole_state(cart_positions_epoch, pole_angles_epoch):
    """
    Plots the cart position and pole angle for one episode on dual y-axes.
    Parameters:
    - cart_positions_epoch (list): A list containing positions of the cart along an episode.
    - pole_angles_epoch (list): A list containing the angles of the pole along an episode.
    """
    steps = range(len(cart_positions_epoch))  # Steps within the episode

    # Creates a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plots Cart Position on the first y-axis
    ax1.plot(steps, cart_positions_epoch, label="Cart Position", color="blue")
    ax1.set_xlabel("Epoch Time Steps")
    ax1.set_ylabel("Cart Position", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    #ax1.set_ylim(-4.8, 4.8)  # CartPole limits for cart position

    # Creates a twin y-axis for Pole Angle
    ax2 = ax1.twinx()
    ax2.plot(steps, pole_angles_epoch, label="Pole Angle", color="green")
    ax2.set_ylabel("Pole Angle (radians)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Title and grid
    plt.title("Cart Position and Pole Angle Over Steps")
    fig.tight_layout()  # Adjusts layout to prevent overlap
    plt.grid(True)
    plt.show()





####### es feo
##def scatter_plot(agent, state_list, cart_positions_list, pole_angles_list):
##    """
##    Plots a scatter plot of the .
##
##    Parameters:
##    - cart_positions_epoch (list): A list containing positions of the cart along an episode.
##    - pole_angles_epoch (list): A list containing the angles of the pole along an episode.
##    """
##
##    world = environment.init()
##    world.reset()
##
##    # State space bounds for CartPole
##    state_space_bounds = [(-2.4*10, 2.4*10),  # Cart position
##                          (-0.5*10, 0.5*10)]  # Pole angle (radians)
##
##    # Number of bins to discretize each dimension of the state space
##    n_bins_x = 10
##    n_bins_y = 48
##
##    # Discretize the state space
##    cart_positions = np.linspace(*state_space_bounds[0], n_bins_y)
##    pole_angles = np.linspace(*state_space_bounds[1], n_bins_x)
##
##    policy_grid = np.full((n_bins_x, n_bins_y), np.nan)
##    
##    grid_values = []     
##    for i in range(len(state_list)):
##        q_values = agent.policy_net(state_list[i])
##        action = torch.argmax(q_values).item()
##        
##        cart_position = cart_positions_list[i]
##        pole_angle = pole_angles_list[i]
##
##        cart_index = int(cart_position*10)
##        pole_index = int(pole_angle*10)
##    
##        grid_values.append([cart_index, pole_index, action])
##        
##    #print("Grid values: ", grid_values)
##
##        
##    # Create a dictionary to store actions at each coordinate
##    action_map = {}
##    
##
##    # Aggregate actions for each point
##    for x, y, action in grid_values:
##        if (x, y) not in action_map:
##            action_map[(x, y)] = []
##        action_map[(x, y)].append(action)
##
##    # Now calculate the average action for each point
##    averaged_actions = []
##    x_coords = []
##    y_coords = []
##
##    for (x, y), actions in action_map.items():
##        averaged_action = np.mean(actions)  # Average action value
##        averaged_actions.append(averaged_action)
##        x_coords.append(x)
##        y_coords.append(y)
##
##    # Plot the scatter plot
##    plt.figure(figsize=(6, 6))
##    scatter = plt.scatter(x_coords, y_coords, c=averaged_actions, cmap="plasma", edgecolor="k", s=300)
##    plt.colorbar(scatter)  # Show color scale
##    plt.title('Scatter Plot of Actions with Averaging')
##    plt.xlabel('X Coordinate')
##    plt.ylabel('Y Coordinate')
##    plt.show()



def policy_heatmap(agent, state_list, cart_positions_list, pole_angles_list):
    """
    Plots heatmap of policy at each discretized state (cart position and pole angle).
    Parameters:
    - agent (Agent class): The model's agent class.
    - state_list (list): A list containing all the states sampled during training.
    - cart_positions_list (list): A list containing positions of the cart along all epochs.
    - pole_angles_list (list): A list containing the angles of the pole along all epochs.
    """

    # Initialise environment
    world = environment.init()
    world.reset()

    # State space bounds for CartPole
    state_space_bounds = [(-2.4*10, 2.4*10),  # Cart position
                          (-0.5*10, 0.5*10)]  # Pole angle (radians)

    # Number of bins to discretize each dimension of the state space
    n_bins_x = 10
    n_bins_y = 48

    # Discretize the state space
    cart_positions = np.linspace(*state_space_bounds[0], n_bins_y)
    pole_angles = np.linspace(*state_space_bounds[1], n_bins_x)

    policy_grid = np.full((n_bins_x, n_bins_y), np.nan)
    
    grid_values = []     
    for i in range(len(state_list)):
        q_values = agent.policy_net(state_list[i])
        action = torch.argmax(q_values).item()
        
        cart_position = cart_positions_list[i]
        pole_angle = pole_angles_list[i]
        
        cart_index = int(cart_position*10) 
        pole_index = int(pole_angle*10)
    
        grid_values.append([cart_index, pole_index, action])

    # Create a dictionary to store actions at each coordinate
    action_map = {}

    # Aggregate actions for each point
    for x, y, action in grid_values:
        if (x, y) not in action_map:
            action_map[(x, y)] = []
        action_map[(x, y)].append(action)

    # Determine the bounds of the grid
    x_coords, y_coords = zip(*action_map.keys())
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Create a 2D grid initialized with NaN
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    heatmap = np.full((grid_height, grid_width), np.nan)

    # Fill the heatmap with averaged values
    for (x, y), actions in action_map.items():
        avg_action = np.mean(actions)
        heatmap[y - y_min, x - x_min] = avg_action

    # Plot the heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="plasma", origin="lower", extent=(x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5))
    plt.colorbar(label='Action Value')
    plt.title('Heatmap of Averaged Actions')
    plt.xlabel('Cart position')
    plt.ylabel('Pole angle (rad)')
    plt.show()





def policy_heatmap2(agent, state_list, cart_positions_list, pole_angles_list):
    """
    Plots heatmap of policy at each discretized state (cart position and pole angle).
    Parameters:
    - agent (Agent class): The model's agent class.
    - state_list (list): A list containing all the states sampled during training.
    - cart_positions_list (list): A list containing positions of the cart along all epochs.
    - pole_angles_list (list): A list containing the angles of the pole along all epochs.
    """

    # Initialise environment
    world = environment.init()
    world.reset()

    # State space bounds for CartPole
    state_space_bounds = [(-2.4*10, 2.4*10),  # Cart position
                          (-0.5*10, 0.5*10)]  # Pole angle (radians)

    # Number of bins to discretize each dimension of the state space
    n_bins_x = 10
    n_bins_y = 48

    # Discretize the state space
    cart_positions = np.linspace(*state_space_bounds[0], n_bins_y)
    pole_angles = np.linspace(*state_space_bounds[1], n_bins_x)

    policy_grid = np.full((n_bins_x, n_bins_y), np.nan)
    
    grid_values = []     
    for i in range(len(state_list)):
        q_values = agent.policy_net(state_list[i])
        action = torch.argmax(q_values).item()
        
        cart_position = cart_positions_list[i]
        pole_angle = pole_angles_list[i]
        
        cart_index = int(cart_position*10) 
        pole_index = int(pole_angle*10)
    
        grid_values.append([cart_index, pole_index, action])

    # Create a dictionary to store actions at each coordinate
    action_map = {}

    # Aggregate actions for each point
    for x, y, action in grid_values:
        if (x, y) not in action_map:
            action_map[(x, y)] = []
        action_map[(x, y)].append(action)

    # Determine the bounds of the grid
    x_coords, y_coords = zip(*action_map.keys())
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Create a 2D grid initialized with NaN
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    heatmap = np.full((grid_height, grid_width), 0.5)

    # Fill the heatmap with averaged values
    for (x, y), actions in action_map.items():
        avg_action = np.mean(actions)
        heatmap[y - y_min, x - x_min] = avg_action

    # Plot the heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="plasma", origin="lower", extent=(x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5))
    plt.colorbar(label='Action Value')
    plt.title('Heatmap of Averaged Actions')
    plt.xlabel('Cart position')
    plt.ylabel('Pole angle (rad)')
    plt.show()
    
    


def results_subplot(durations, losses, q_values_total, cart_positions_epoch, pole_angles_epoch):
    """
    Creates a single figure with subplots for:
    1. Learning curve
    2. Loss function
    3. Q-values
    4. Cart-pole state of 1 episode (cart position and pole angle)

    Parameters:
    - durations (list): A list containing the durations (same as rewards) per episode.
    - losses (list): A list containing output of loss function for all time instances.
    - q_values_total (list): A list of lists, where each inner list contains Q-values for a single episode.
    - cart_positions_epoch (list): A list containing positions of the cart along an episode.
    - pole_angles_epoch (list): A list containing the angles of the pole along an episode.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # 4 subplots on 2x2 grid

    #### Learning Curve ####
    
    def moving_average(data, window_size):
        cumulative_sum = [0] + [sum(data[:i]) for i in range(1, len(data) + 1)]
        return [(cumulative_sum[i + window_size] - cumulative_sum[i]) / window_size for i in range(len(data) - window_size + 1)]

    # Plot the durations
    axes[0,0].plot(durations, color='blue')
    
    # Plot the moving average
    if len(durations) > 10:
        ma = moving_average(durations, window_size=10)
        axes[0,0].plot(range(10, len(durations) + 1), ma, label='Moving Average (window=10)', color='red')
    
    axes[0,0].set_title("Learning Curve")
    axes[0,0].set_xlabel("Training epochs")
    axes[0,0].set_ylabel("Duration")
    axes[0,0].grid(True)
    axes[0,0].legend()


    #### Loss Function ####
    axes[0,1].plot(losses, color='red')
    axes[0,1].set_title("Loss Function")
    axes[0,1].set_xlabel("Time steps")
    axes[0,1].set_ylabel("Loss")
    axes[0,1].grid(True)
    

    #### Q-values ####
    episodes = range(len(q_values_total))
    mean_q_values = [np.mean([np.mean(q_values) for q_values in episode]) for episode in q_values_total]
    axes[1,0].plot(episodes, mean_q_values, color='blue')
    axes[1,0].set_title("Q-Values Over Episodes")
    axes[1,0].set_xlabel("Training epochs")
    axes[1,0].set_ylabel("Q-Values")
    axes[1,0].legend()
    axes[1,0].grid(True)



    #### Cart Position and Pole Angle ####
    steps = range(len(cart_positions_epoch))
    
    axes[1,1].plot(steps, cart_positions_epoch, color="blue", alpha=0.8)
    axes[1,1].set_ylabel("Cart Position", color="blue")
    axes[1,1].tick_params(axis="y", labelcolor="blue")

    ax2 = axes[1,1].twinx()
    ax2.plot(steps, pole_angles_epoch, label="Pole Angle", color="green", alpha=0.8)
    ax2.set_ylabel("Pole Angle (radians)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(-0.5, 0.5)  # Example limits for pole angle
    
    axes[1,1].set_title("Cart Position and Pole Angle over 1 epoch")
    axes[1,1].grid(True)
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    

    fig.tight_layout()  # Adjust layout to prevent overlap

    

    
