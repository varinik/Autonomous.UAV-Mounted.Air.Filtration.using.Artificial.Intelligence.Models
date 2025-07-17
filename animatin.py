import numpy as np
import pandas as pd
import scipy.io
import scipy.spatial
import math

#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import animation


from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import pickle
import math


# Configuration
SCENE_SIZE = 15
PLUME_SIZE = 40
DECAY_FACTOR = 0.9
NUM_FRAMES = 50
DRONE_STEP = 6
SEARCH_RADIUS = 6
DETECTION_THRESHOLD = 5.0  # Lowered for sensitivity

Q = 1000000.0    # Stronger source
u = 0.3      # Wind speed
H = 0       # Height breaks z symmetry

ay, by = 1, 0.6 # Lateral spread
az, bz = 1, 0.8

# define models:

class HMM3DWithViterbi:
    def __init__(self, n_states, nn_hidden_layers=50, max_iter=1000):
        self.n_states = n_states

        # uniform transition probabilities
        self.transition_matrix = np.ones((n_states, n_states)) / n_states

        # Uniform initial probabilities
        self.initial_probs = np.ones(n_states) / n_states

        #MLPRegressor used for non-linear emmission probabilities
        self.nn_models = [MLPRegressor(hidden_layer_sizes=nn_hidden_layers, max_iter=max_iter) for _ in range(n_states)]  # One NN per state

    # implements the viterbi algorithm to find the most probable sequence of hidden states
    def viterbi(self, observations):
        # number of observations in the path
        n_observations = len(observations)
        # stores the log-probabilities of all the paths that end in hidden state j at time t
        log_probs = np.zeros((n_observations, self.n_states))
        # stores the best path
        paths = np.zeros((n_observations, self.n_states), dtype=int)

        # defining the probabilities for the first observation
        emission_probs = self._emission_probabilities(observations[0])
        log_probs[0] = np.log(self.initial_probs + 1e-9) + np.log(emission_probs + 1e-9)

        # loop to go through all possible paths and find the best one based on the emission probabilities
        for t in range(1, n_observations):
            for j in range(self.n_states):
                transition_probs = log_probs[t - 1] + np.log(self.transition_matrix[:, j] + 1e-9)
                best_prev_state = np.argmax(transition_probs)
                log_probs[t, j] = transition_probs[best_prev_state] + np.log(self._emission_probabilities(observations[t])[j] + 1e-9)
                paths[t, j] = best_prev_state

        # backtracking to give the most probable path
        best_last_state = np.argmax(log_probs[-1])
        best_path = [best_last_state]
        for t in range(n_observations - 1, 0, -1):
            best_last_state = paths[t, best_last_state]
            best_path.append(best_last_state)
        return list(reversed(best_path))

    # function to train the model
    def fit(self, all_observations):
        # preparing the data
        X_by_state = [[] for _ in range(self.n_states)]
        y_by_state = [[] for _ in range(self.n_states)]
        sum = 0
        total = 0
        all_data = np.vstack(all_observations)
        # kmeans clustering used to see how many hidden states are needed for the specific data
        kmeans = KMeans(n_clusters=self.n_states, n_init=10).fit(all_data)
        initial_states = kmeans.labels_

        # associates parts of the trajectory with certain hidden states
        for trajectory in all_observations:
            states = self.viterbi(trajectory)[:-2]
            for t, state in enumerate(states):
                X_by_state[state].append(trajectory[t])
                y_by_state[state].append(trajectory[t + 1])

        # trains the emission probabilties model using the data
        for state in range(self.n_states):
            X = np.array(X_by_state[state])
            y = np.array(y_by_state[state])
            if len(X) > 10:
                x = 10
                self.nn_models[state].fit(X, y)

            # used to find the error from what the HMM predicted
            for i, x_val in zip(y, X):
                norm_i = i / (np.linalg.norm(i) + 1e-9)
                norm_point = self.predict(x_val) / (np.linalg.norm(self.predict(x_val)) + 1e-9)
                sum += scipy.spatial.distance.euclidean(norm_i, norm_point)
                total += 1
            # print(total)
            print("training loss: ", sum/total)

        # updates starting transission matrix with the information learned from the training (used ot update the transimission matrix from being uniform to something that can accurately represent the data)
        for i in range(len(states) - 1):
            self.transition_matrix[states[i], states[i + 1]] += 1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

    # predict function
    def predict(self, observations):
        if observations.ndim == 1:
            observations = [observations]
        hidden_states = self.viterbi(observations)
        last_hidden_state = hidden_states[-1]
        return self.nn_models[last_hidden_state].predict([observations[-1]])[-1]

    # defining the emission probabilities
    def _emission_probabilities(self, observation):
        emissions = []
        for state in range(self.n_states):
            try:
                predicted = self.nn_models[state].predict([observation])
                likelihood = np.exp(-np.linalg.norm(predicted - observation))
            except Exception:
                likelihood = 1e-6
            emissions.append(likelihood)
        emissions = np.array(emissions)
        return emissions / emissions.sum()

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=5, output_size=3):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        output = self.fc(out[:, :, :])
        return output   


# import models:
with open("pre-trained models/traj_hybrid_x.pkl", 'rb') as file:
    x_model = pickle.load(file)
with open("pre-trained models/traj_hybrid_y.pkl", 'rb') as file:
    y_model = pickle.load(file)
with open("pre-trained models/traj_hybrid_z.pkl", 'rb') as file:
    z_model = pickle.load(file)
with open("pre-trained models/hmm.pkl", 'rb') as file:
    hmm = pickle.load(file)

source = load_model('pre-trained models/lstm_source_pred.keras') 

lstm = LSTM(3, 128, 4, 3)
lstm.load_state_dict(torch.load('pre-trained models/trajectory_lstm.pth')) # replace with location of files


# Setup plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pollution_field = np.zeros((SCENE_SIZE, SCENE_SIZE, SCENE_SIZE))
source_positions = []

his = []

# Random source path

# Parameters

df = pd.read_csv('circle_path.txt', sep=',', header=None, names=['X', 'Y', 'Z'])

# Extract position columns (X, Y, Z)

path = df[['X', 'Y', 'Z']].to_numpy()

# print(path)

start_pos = path[0]

def source_pos(t):
    pos = path[t]
    return tuple(np.clip(pos, 0, SCENE_SIZE - 1))

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def add_plume(p_field, x0, y0, z0, Q, sigma):
    x_min = clamp(x0 - PLUME_SIZE // 2, 0, SCENE_SIZE - 1)
    x_max = clamp(x0 + PLUME_SIZE // 2, 0, SCENE_SIZE - 1)
    y_min = clamp(y0 - PLUME_SIZE // 2, 0, SCENE_SIZE - 1)
    y_max = clamp(y0 + PLUME_SIZE // 2, 0, SCENE_SIZE - 1)
    z_min = clamp(z0 - PLUME_SIZE // 2, 0, SCENE_SIZE - 1)
    z_max = clamp(z0 + PLUME_SIZE // 2, 0, SCENE_SIZE - 1)

    x = np.arange(x_min, x_max + 1)
    y = np.arange(y_min, y_max + 1)
    z = np.arange(z_min, z_max + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    dx = X - x0
    dy = Y - y0
    dz = Z - z0
    exponent = -(dx**2 + dy**2 + dz**2) / (2 * sigma**2)
    plume = Q * np.exp(exponent)

    p_field[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] += plume

# Drones (closer to center)
drone_positions = [
    np.array(path[0], dtype=float),
    np.array([path[0][0]+2, path[0][1]+2, path[0][2]+2], dtype=float),
    np.array([path[0][0]+3, path[0][1]+3, path[0][2]+3], dtype=float)
]
drone_trails = [[], [], []]
drone_colors = ['red', 'green', 'purple']
drone_readings = [0.0, 0.0, 0.0]

epsilon = 1e-200
def find_conc(bit_map, len_map, conc):
    found = []
    for i in range(len_map):
        for j in range(len_map):
            for k in range(len_map):
                if(bit_map[i][j][k] <= conc + epsilon and conc - epsilon <= bit_map[i][j][k]):
                    found.append([i, j, k])
    return found

def conc_match(bit_map, conc, i, j, k, shift):
    trial = [[i-shift, j-shift, k-shift], 
             [i-shift, j-shift, k+shift], 
             [i-shift, j+shift, k-shift], 
             [i-shift, j+shift, k+shift], 

             [i+shift, j-shift, k-shift], 
             [i+shift, j-shift, k+shift], 
             [i+shift, j+shift, k-shift], 
             [i+shift, j+shift, k+shift]]
    
    output = []

    for i, j, k in trial:
        if(bit_map[i][j][k] == conc):
            output.append([i, j, k])
    
    return output

def get_source(bit_map, concentrations):
    c1, c2, c3 = concentrations
    possible = find_conc(bit_map=bit_map, len_map=51, conc=c1)
    # print(possible)

    # c2:
    c2_works = []
    updated_possible = []
    for i, j, k in possible:
        output = conc_match(bit_map=bit_map, conc=c2, i=i, j=j, k=k, shift=2)
        for x,y,z in output:
            c2_works.append([i, j, k, x, y, z])

    
    final = []
    for i, j, k, i2, j2, k2 in c2_works:
        output = conc_match(bit_map=bit_map, conc=c3, i=i, j=j, k=k, shift=3)
        for x,y,z in output:
            final.append([i, j, k, i2, j2, k2, x, y, z])

    # reformat input
    # print(final)
    # print(possible)
    sadf = input()
    final = final[0]
    # all xs
    x = [final[0],final[3],final[6]]
    # all ys
    y = [final[1],final[4],final[7]]
    # all zs
    z = [final[2],final[5],final[8]]
    return x, y, z


def plume_function(x0, y0, z0, x, y, z):
    dx = max(abs(x - x0), 0.0001)
    dy = max(abs(y - y0), 0.0001)
    dz = max(abs(z - z0), 0.0001)

    s_y = ay * dx**by
    s_z = az * dx**bz
    s_y = max(s_y, 0.00002)
    s_z = max(s_z, 0.00002)

    term1 = Q / (2 * np.pi * s_y * s_z * u)
    term2 = np.exp(-dy**2 / (2 * s_y**2))
    term3 = np.exp(-(dz - H)**2 / (2 * s_z**2))
    c = term1 * term2 * term3
    return c

def create_bit_map():
    source = [50, 50, 50]
    bit_map = [[[0 for _ in range(200)] for _ in range(200)] for _ in range(200)]

    for i in range(0, 200):
        for j in range(0, 200):
            for k in range(0, 200):
                bit_map[i][j][k] = plume_function(source[0], source[1], source[2], i, j, k)
    return bit_map

def hybrid_pred(his):
    print(his)
    his_tensor = torch.tensor(his, dtype=torch.float32)
    his_tensor_lstm = torch.tensor([his], dtype=torch.float32)

    # LSTM and HMM predictions
    pred_lstm = lstm(his_tensor_lstm).detach().numpy()[:, -1]  # shape: (1, 3)
    pred_hmm = np.array(hmm.predict(his_tensor)).reshape(1, 3)

    # Combine predictions for hybrid model
    X_stack = np.hstack([pred_lstm[:, 0].reshape(-1, 1), pred_hmm[:, 0].reshape(-1, 1)])
    Y_stack = np.hstack([pred_lstm[:, 1].reshape(-1, 1), pred_hmm[:, 1].reshape(-1, 1)])
    Z_stack = np.hstack([pred_lstm[:, 2].reshape(-1, 1), pred_hmm[:, 2].reshape(-1, 1)])

    # pred_x = abs(x_model.predict([X_stack[-1]])[-1])
    # pred_y = abs(y_model.predict([Y_stack[-1]])[-1])
    # pred_z = abs(z_model.predict([Z_stack[-1]])[-1])
    pred_x = int(np.round(abs(x_model.predict([X_stack[-1]])[-1])))
    pred_y = int(np.round(abs(y_model.predict([Y_stack[-1]])[-1])))
    pred_z = int(np.round(abs(z_model.predict([Z_stack[-1]])[-1])))

    # Generate new drone positions around hybrid prediction
    drone_positions_2 = [
        np.array([pred_x, pred_y, pred_z], dtype=float),
        np.array([pred_x + 2, pred_y + 2, pred_z + 2], dtype=float),
        np.array([pred_x + 3, pred_y + 3, pred_z + 3], dtype=float)
    ]
    # xx = x[0]
    # print("drone1 pos: ", drone_positions_2[0])
    # print("drone2 pos: ", drone_positions_2[1])
    # print("drone3 pos: ", drone_positions_2[2])

    return drone_positions_2


def update_drone(pos, drone_positions, bit_map, history):
    his = history.copy()
    x0, y0, z0 = pos
    drone_points = [pos.astype(int) for pos in drone_positions.copy()]
    # print("printing: ", drone_points)

    concentrations = [
        plume_function(x0, y0, z0, drone_points[0][0], drone_points[0][1], drone_points[0][2]),
        plume_function(x0, y0, z0, drone_points[1][0], drone_points[1][1], drone_points[1][2]),
        plume_function(x0, y0, z0, drone_points[2][0], drone_points[2][1], drone_points[2][2])
    ]
    # print("concentrations: ", concentrations)

    dx, dy, dz = get_source(bit_map, concentrations)
    input_sample=np.expand_dims([dx, dy, dz], axis=0)
    lstm_output = source.predict(np.array(input_sample))
    rounded = np.round(lstm_output[0])

    x = dx[0]
    y = dy[0]
    z = dz[0]

    result = [
        (50 - x) * int(rounded[0]),
        (50 - y) * int(rounded[1]),
        (50 - z) * int(rounded[2])
    ]
    
    pred_source = [a + b for a, b in zip(drone_points[0], result)]
    his.append(pred_source)
    # print("pred_source: ", pred_source)
    print("actual source: ", x0, y0, z0)
    actual_drone_pos = [
        pred_source,
        [pred_source[0] + 2, pred_source[1] + 2, pred_source[2] + 2], 
        [pred_source[0] + 3, pred_source[1] + 3, pred_source[2] + 3]
    ]

    print("history: ", his)
    predicted = hybrid_pred(his)
    print("predicted: ", predicted)
    return predicted, actual_drone_pos, pred_source, his





def update(frame):
    global pollution_field, source_positions, drone_readings, bit_map, his, drone_positions

    ax.cla()
    ax.set_facecolor('white')
    ax.set_xlim(0, SCENE_SIZE)
    ax.set_ylim(0, SCENE_SIZE)
    ax.set_zlim(0, SCENE_SIZE)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(colors='black')
    ax.grid(color='gray', linestyle='--', alpha=0.3)

    pollution_field *= DECAY_FACTOR
    x0, y0, z0 = source_pos(frame)
    source_positions.append((x0, y0, z0))

    Q_var = 60 + 100 * np.abs(np.sin(frame * 0.2))  # Higher emissions
    sigma_var = 10 + 10 * np.abs(np.cos(frame * 0.15))
    # add_plume(pollution_field, x0, y0, z0, Q_var, sigma_var)

    # threshold = 0.5 * pollution_field.max()
    # idxs = np.where(pollution_field >= threshold)
    # pollution_vals = pollution_field[idxs]

    # if pollution_vals.size > 0:
    #     pollution_norm = (pollution_vals - threshold) / (pollution_vals.max() - threshold)
    #     pollution_norm = np.clip(pollution_norm, 0, 1)
    #     colors = np.zeros((pollution_norm.size, 4))
    #     colors[:, 2] = 1.0
    #     colors[:, 3] = 0.01 + 0.01 * pollution_norm
    #     ax.scatter(idxs[0], idxs[1], idxs[2], color=colors, marker='o', s=1)

    ax.scatter(x0, y0, z0, color='red', s=150, marker='*', label='Source')
    if len(source_positions) > 1:
        xs, ys, zs = zip(*source_positions)
        ax.plot(xs, ys, zs, color='black', linewidth=1.5, alpha=0.7, label='Source Path')

    info = [f'Time step: {frame}', f'Q = {Q_var:.1f}   σ = {sigma_var:.1f}']

    drone_positions, hybrid, pred_source, history = update_drone([x0, y0, z0],drone_positions, bit_map, his)
    his = history.copy()
    ax.scatter(*np.array(pred_source), color='orange', s=150, marker='*', label='Pred_Source')
    # print(hybrid)
    drone_positions = np.array(drone_positions)
    for i in range(3):
        x, y, z = drone_positions[i].astype(int)
        pollution_val = pollution_field[x, y, z]
        drone_readings[i] = pollution_val
        drone_trails[i].append(drone_positions[i].copy())

        trail = np.array(drone_trails[i])
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=drone_colors[i], linewidth=1.5, alpha=0.7)

        marker = '^' if pollution_val >= DETECTION_THRESHOLD else 'o'
        ax.scatter(*drone_positions[i], color=drone_colors[i], s=100, marker=marker, label=f'Drone {i+1}')
        info.append(f'Drone {i+1}: {pollution_val:.1f} {"🔵" if pollution_val >= DETECTION_THRESHOLD else ""}')

    print(f"Frame {frame} - Drone readings: {[round(r,1) for r in drone_readings]}")

    ax.set_xlabel('X (Downwind)')
    ax.set_ylabel('Y (Crosswind)')
    ax.set_zlabel('Z (Altitude)')
    ax.set_title('3D Pollution Plume with Tracking Drones', color='black')
    ax.text2D(0.05, 0.95, "\n".join(info), transform=ax.transAxes, fontsize=12, color='black')

bit_map = create_bit_map()
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=500, blit=False)
plt.legend()
plt.show()
# print(update_drone(path[0], drone_positions, bit_map, []))
# print(update_drone(path[1], drone_positions, bit_map, []))