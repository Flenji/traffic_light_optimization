# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:20:12 2023

@author: hanne
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def load_object(filename, path =""):
    
    filename = os.path.join(path,filename)
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj
        
def save_object(obj ,filename, path = ""):
    filename = os.path.join(path,filename)
    with open (filename, "wb") as file:
        pickle.dump(obj, file)
    

def plot_learning_curve(scores, epsilons, filename, path ="", mean_over = 10):
    
    filename = os.path.join(path,filename)
    N = int(len(scores)/mean_over)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[t*mean_over:t*mean_over+mean_over])

    eps_avg = np.empty(N)
    for t in range(N):
        eps_avg[t] = np.mean(epsilons[t*mean_over:t*mean_over+mean_over])


    x = np.arange(N) * mean_over
    
    """R = len(scores)%mean_over
    N = len(scores) - (R+mean_over)
    x = np.arange(N)
    
    
    running_avg = np.empty(N)
    eps_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[t:t+mean_over])
        eps_avg[t] = np.mean(epsilons[t:t+mean_over])"""
    
    fig, ax1 = plt.subplots()

    # Plot the first graph with the first y-axis (left)
    ax1.plot(x, running_avg, 'b')
    ax1.set_xlabel('learning_steps')
    ax1.set_ylabel('Reward', color='b')
    ax1.tick_params('y', colors='b')

    # Create a second y-axis on the right side
    ax2 = ax1.twinx()

    # Plot the second graph with the second y-axis (right)
    ax2.plot(x, eps_avg, 'r')
    ax2.set_ylabel('Epsilon', color='r')
    ax2.tick_params('y', colors='r')


    #plt.plot(running_avg)
    #plt.plot(eps_avg)
    #plt.xlabel("Traffic")
    #plt.ylabel("Reward")
    plt.title("Fumo Reward")

    plt.savefig(filename)