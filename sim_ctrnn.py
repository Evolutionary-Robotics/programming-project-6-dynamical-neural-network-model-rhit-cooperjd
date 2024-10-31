import ctrnn
import matplotlib.pyplot as plt
import numpy as np

# Parameters
size = 10
duration = 10000
stepsize = 0.01
duration_stepsize = 1000

def set_duration(desired_duration):
    global duration
    global duration_stepsize

    duration = desired_duration
    duration_stepsize = duration/10

# Data
time = np.arange(0.0,duration,stepsize)
outputs = np.zeros((len(time),size))
states = np.zeros((len(time),size))
steps = np.zeros(int(duration/duration_stepsize))
neurons = [""]*(int(duration/duration_stepsize))

for i in range(size):
    neurons[i] = f"Neuron {i+1}"

for j in range(int(duration/duration_stepsize)):
    steps[j] = j + 1

# Initialization
nn = ctrnn.CTRNN(size)

# Neural parameters at random
nn.randomizeParameters()

# nn.setConstantParameters()

# Initialization at zeros or random
nn.initializeState(np.zeros(size))

# Run simulation
step = 0
for t in time:
    nn.step(stepsize)
    states[step] = nn.States
    outputs[step] = nn.Outputs
    step += 1

# How much is the neural activity changing over time
activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
print("Overall activity: ",activity)

def plot_neuron_activity(time, outputs):
# Plot activity
    plt.plot(time,outputs)
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.title("Neural output activity")
    plt.show()

def plot_neuron_state_changes_line_graph(time, states):
# Plot neuron state changes as a line graph
    plt.plot(time,states)
    plt.xlabel("Time")
    plt.ylabel("States")
    plt.title("Neural state activity")
    plt.show()

def plot_neuron_state_changes_heatmap(duration, duration_stepsize, neurons, size, states):
# Plot neuron state changes
# This heat map code was aided in its creation by official plt documentation and Claude AI
    fig, ax = plt.subplots()
    sample_times = np.arange(0, duration+duration_stepsize, duration_stepsize, dtype=int)
    sampled_states = states[sample_times]
    im = ax.imshow(sampled_states.T, cmap='viridis')
    ax.set_xlabel('Sample Times')
    ax.set_ylabel('Neuron i')
    ax.set_xticks(np.arange(len(sample_times)))
    ax.set_xticklabels(sample_times, rotation=45)
    ax.set_yticks(np.arange(size))
    ax.set_yticklabels(neurons)
    cbar = plt.colorbar(im)
    cbar.set_label('Neuron State')
    ax.set_title("How Neuron i Activation Changes Over Time")
    plt.show()

def print_out_params(nn):
    print("========== Parameters for Trial=============")
    print(f"Weights: {nn.Weights}")
    print(f"Biases: {nn.Biases}")
    print(f"Time Constants: {nn.TimeConstants}")
    print("============================================")

print_out_params(nn)
plot_neuron_state_changes_heatmap(duration, duration_stepsize, neurons, size, states)

# Save CTRNN parameters for later
nn.save("ctrnn")
