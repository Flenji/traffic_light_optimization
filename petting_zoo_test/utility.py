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
    
def plot_learning_curves(scores, epsilons, nrows, ncols,  filename, figsize=(15, 8),  path = "", mean_over = 10):
    
    n_agents = len(scores)
    
    N = int(len(epsilons)/mean_over)
    
    running_avg = dict.fromkeys(scores.keys())
    
    for agent in scores:
        
        running_avg[agent] = np.empty(N)
        for t in range(N):
            running_avg[agent][t] = np.mean(scores[agent][t*mean_over:t*mean_over+mean_over])
    
    eps_avg = np.empty(N)
    for t in range(N):
        eps_avg[t] = np.mean(epsilons[t*mean_over:t*mean_over+mean_over])
        
    x = np.arange(N) * mean_over
    
    # Create a figure and a 2D array of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharey=True, sharex= True) 
    
    
    eps_color = "#991203"
    axes2 = []
    for i, agent in enumerate(running_avg.keys()):
        
        row_idx = i // nrows
        col_idx = i % ncols
        
        ax = axes[row_idx, col_idx]
        ax.plot(x, running_avg[agent])
        ax.title.set_text(f"Agent {agent}")
        
        
        axes2.append(ax.twinx())
        ax2 = axes2[i]
        
        ax2.plot(x, eps_avg, color = eps_color, alpha = 0.8)
        ax2.axis("off")
        
        if col_idx == ncols-1:
            ax2.axis("on")
            ax2.set_ylabel("epsilon")
            ax2.yaxis.label.set_color(eps_color)
            ax2.tick_params(axis='y', colors=eps_color)
        else:
            ax.set_ylabel("Average Reward")
            
        if row_idx == nrows-1:
            ax.set_xlabel("Learning Steps")
            
        if i == len(running_avg)-1 and col_idx != ncols-1:
           ax2.axis("on")
           ax2.set_ylabel("epsilon")
           ax2.yaxis.label.set_color(eps_color)
           ax2.tick_params(axis='y', colors=eps_color) 
           
           axes[row_idx,-1].axis("off")
        
    filename = os.path.join(path,filename)
    plt.savefig(filename+".png")
    
def createPath(*args):
    return os.path.join(*args)


def get_time_formatted(seconds):
    seconds = int(seconds)
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    formatted = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
    return formatted
    