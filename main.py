import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import environment #file
import model #file
import graphical_representation as gr #file
import torch
from itertools import count
from settings import DEVICE, SCREEN_WIDTH, TARGET_UPDATE, EPOCHS #file
import time

# Environment Setup
world = environment.init()
world.reset()

# Log the duartion, loss function, Q-values, actions taken for distribution, states and cart position and pole angle for each iteration
durations = []
losses = []
q_values_total = []
actions_taken = []
cart_positions_epoch = []
pole_angles_epoch = []
state_list = []
cart_positions_list = []
pole_angles_list = []



# Training
agent = model.Agent(DEVICE)

for i in range(EPOCHS):
	# Initialize environment	
	print("epoch: ", i)
	world.reset()
	
	epoch_q_values = []

	# Get current state
	last_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
	current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
	state = current_screen - last_screen
	

	for t in count():
		# Select and perform an action
		action = agent.select_action(state)
		state_list.append(state) # Log state
		actions_taken.append(action.item())  # Log action
		q_values = agent.policy_net(state)  # Get Q-values for the current state
	
		epoch_q_values.append(q_values.detach().cpu().numpy())  # Log Q-values
		_, reward, done, _ = world.step(action.item()) # Agent performs action
		
		# Extract cart position and pole angle from environment state
		cart_position = world.state[0]
		pole_angle = world.state[2]
		cart_positions_list.append(cart_position) # Log cart position
		pole_angles_list.append(pole_angle) # Log pole angle
		
		if i == (EPOCHS-1): # if final epoch
			# Append cart position and pole angle of last epoch to visualise
			cart_positions_epoch.append(cart_position)
			pole_angles_epoch.append(pole_angle)
			
			#print("cart positions and pole angle: ", cart_position, pole_angle)
	
		reward = torch.tensor([reward], device=DEVICE)
		

		# Observe new state
		last_screen = current_screen
		current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)

		if not done:
			next_state = current_screen - last_screen
		else:
			next_state = None

		# Store the transition in memory
		agent.remember(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Optimize the target network
		loss = agent.optimize_model()
		if loss is not None:
			losses.append(loss.item())
		

                # Render the environment to visualise
		world.render()
		time.sleep(0.01) # slight delay for better visualisation

		if done:
			durations.append(t + 1) # Log durations
			q_values_total.append(epoch_q_values)  # Log Q-values for this episode
			break
		
	## Target Update method used in paper [1]
	if i % TARGET_UPDATE == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())
	
	"""
	## More recent Target Update method [10] (you can use this instead of the above!)
	for target_param, policy_param in zip(agent.target_net.parameters(),agent.policy_net.parameters()):
		target_param.data.copy_(0.99 * policy_param.data + (1-0.99) * target_param.data)
	"""
                

world.render()
world.close()


print("durations: ", durations)
gr.sum_durations(durations)
gr.policy_heatmap(agent, state_list, cart_positions_list, pole_angles_list)
gr.policy_heatmap2(agent, state_list, cart_positions_list, pole_angles_list)
#gr.scatter_plot(agent, state_list, cart_positions_list, pole_angles_list)
#gr.policy_heatmap2(agent, state_list, cart_positions_list, pole_angles_list)
#gr.results_subplot(gr.learning_curve(durations), gr.loss_function(losses), gr.Q_values(q_values_total), gr.cart_pole_state(cart_positions, pole_angles))
gr.results_subplot(durations, losses, q_values_total, cart_positions_epoch, pole_angles_epoch)
# Plot learning curve 
print("Learning curve durations: ")
gr.learning_curve(durations)
print("Loss function: ")
gr.loss_function(losses)
print("Q-values: ")
gr.Q_values(q_values_total)
print("Plotting action distribution")
gr.action_distribution(actions_taken)
print("Cart position and Pole Angle across final episode")
gr.cp_pa_visualisation(cart_positions_epoch, pole_angles_epoch)
gr.cart_pole_state(cart_positions_epoch, pole_angles_epoch)
