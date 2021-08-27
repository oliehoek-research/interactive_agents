# NOTE: Copied from: https://colab.research.google.com/drive/19O0Uv6K2I3FZZqTL6tuNr2gpJbAMC854#scrollTo=pb1MI1eYKox9
# NOTE: Simple script applying otherplay to the coordination game - not sufficient for the linguistic coordination repeated game - just uses a stateless action distribution

import torch  # NOTE: All their code uses pytorch
from os import path

import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn import init
from torch.optim import Adam, SGD
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from itertools import product

from google.colab import files


# NOTE: Seems to construct the payoff matrix for the coordination game
def get_payoff_values(n_dim, shuffle):
    payoff_values =torch.zeros(n_dim, n_dim, requires_grad=False)  # NOTE: Initialize N x N matrix of zeros
    payoff_values[0,0]=0.9  # NOTE: This seems to be the focal point (unique low-scoring action)
    if shuffle:
        perm = np.random.permutation(range(1,n_dim))  # NOTE: Optionally permute the payoff matrix - seems to be how they implement otherplay in this setting
    else:
        perm = np.array(range(1, n_dim ))
        perm = np.concatenate([np.zeros(1,dtype=int), perm])
    for s in range(1,n_dim):
        payoff_values[s,perm[s]] += 1.0  # NOTE: Set payoffs for all other actions to 1
    return payoff_values


def entropy(pi):  # NOTE: Pretty self explanatory, computes the Shannon entropy of a discrete distribution
    return -(torch.log(pi)*pi).sum()


def get_theta(dims, std_val):  # NOTE: What does this do?
    return init.normal_(torch.zeros(dims[0], dims[1], requires_grad=True), std=std_val)


def get_thetas(n_dim, std_val, JAL):  # NOTE: What does this do?
    if JAL:
        dims =[n_dim**2, 1]
        theta= get_theta(dims,std_val)
        return theta
    dims =[n_dim, 1]
    return [get_theta(dims,std_val) for _ in [0,1]]
  

def get_objective(thetas_dec, payoff_values, l_pen, JAL):  # NOTE: Computes the payoff for a given set of logits
    if JAL:
        p1 = torch.softmax(thetas_dec,0) 
        p_s_comma_a = torch.reshape(p1, [int(p1.shape[0]**0.5), -1] )  # NOTE: Computes a joint distribution directly - JAL seems to refer to learning a correlated solution
    else:
        thetas_1 = thetas_dec[0] 
        thetas_2 = thetas_dec[1]
        p1 = torch.softmax(thetas_1,0)  # NOTE: seems to compute an action distribution from logit values
        p2 = torch.softmax(thetas_2,0)
        p_s_comma_a =p1.matmul(torch.reshape(p2,[1,-1]))  # NOTE: Computes the joint action distribution (may not always be a product of individual distributions)

    ent = entropy(p_s_comma_a.reshape([-1])) 
    r1 = (p_s_comma_a * payoff_values).sum() + ent * l_pen  # NOTE: Compute entropy-reguarized loss
    r2 = r1
    rewards = [r1, r2] 
    losses = [-r1, -r2]
    return rewards, losses


def train_policy(thetas_dec, JAL, other_play, l_pen, t_max, optim, lr ):
    payoff_values = get_payoff_values(n_dim, other_play)  # NOTE: Seems to construct a single instance of the coordination game
    returns = np.zeros((t_max))  # NOTE: Stores a history of returns up to thr max numbers of epsodes

    # NOTE: Initialize optimizers
    if optim == 'sgd':
        optimizer = [ SGD( [par], lr=lr) for par in thetas_dec ]
    else:
        optimizer = [ Adam( [par], lr=lr) for par in thetas_dec ]
    
    for i in range(t_max):  # NOTE: Outer loop, evaluation  intervals
        for ii, opt in enumerate( optimizer ):  # NOTE: Inner loop, training episodes
            if other_play:
                payoff_values = get_payoff_values(n_dim, other_play)  # NOTE: Only if using otherplay, regenerate the matrix each episode
            opt.zero_grad()
            rewards, losses =get_objective(thetas_dec, payoff_values, l_pen,  JAL)  # NOTE: Compute the losses
            losses[ii].backward()
            opt.step()  # NOTE: The gradient descent step in the logit strategy space
        r_s = [rewards[0].data.numpy() for loss in rewards]
        returns[i] = r_s[0]
    return thetas_dec, returns


def evaluate(policy_0, policy_1, n_dim):
    payoff_values = get_payoff_values(n_dim, False)
    if JAL:
        p1 = torch.softmax(policy_0,0) 
        p1 = torch.reshape(p1, [n_dim, n_dim] ).sum(-1).reshape([-1,1])
        p2 = torch.softmax(policy_1,0) 
        p2 = torch.reshape(p2, [n_dim, n_dim] ).sum(0).reshape([-1,1])
        thetas_dec = [torch.log(p1), torch.log(p2)]
    else:
        thetas_dec = [policy_0, policy_1 ]
    rewards, _ = get_objective(thetas_dec, payoff_values, 0,  False)
    return rewards[0].data.numpy()


def get_cross_play(policies, pops, n_dim):
    results = np.zeros((pops, pops))
    for i in range(pops):
        for j in range(i):
            results[i, j ] = evaluate(policies[i], policies[j], n_dim)
    results = results + results.T
    results = results.sum(-1)/(pops-1)
    return results


# NOTE: Seems to train a population for cross-evaluation
def pop_training( other_play, pops, t_max ):
    n_dim = 10
    lr = 0.05
    l_pen = -0.02

    optim  = 'adam'
    policies = []

    for i in range(pops):
        policies.append(  get_thetas(n_dim, 1, JAL))
    payoff_values = get_payoff_values(n_dim, other_play)
    returns = np.zeros((pops,t_max))
    cross_play = np.zeros((pops,t_max))

    if optim == 'sgd':
        optimizers = [ SGD( [par], lr=lr) for par in policies ]
    else:
        optimizers = [ Adam( [par], lr=lr) for par in policies ]
    for i in range(t_max):
        for pop in range(pops):
            if other_play:
                payoff_values = get_payoff_values(n_dim, other_play)
            optimizers[pop].zero_grad()
            rewards, losses =get_objective(policies[pop], payoff_values, l_pen,  JAL)
            losses[0].backward()
            optimizers[pop].step()
            r_s = [rewards[0].data.numpy() for loss in rewards]
            returns[pop, i] = r_s[0]
        cross_plays =  get_cross_play(policies, pops, n_dim)
        cross_play[:, i] = cross_plays
    return returns, cross_play


pops = 30
t_max = 1000
JAL = True
returns = []
cross_plays = []
for op in [True, False]:
    re, cp = pop_training(op, pops, t_max)
    returns.append(re)
    cross_plays.append(cp)


# Plotting

plt.figure(figsize=(7,3.5))

n_runs = pops
n_readings = t_max
mode_labels = ['Other-Play', 'Self-Play' ]

colors = ['#d62728', '#1f77b4']

plt.subplot(1,2,1)

x_vals = np.arange(n_readings)
for op in [0,1]:
    vals = returns[op]
    y_m = vals.mean(0)
    y_std = vals.std(0) / ( n_runs**0.5 )
    plt.plot( x_vals, y_m, colors[op],label = mode_labels[ op ])
    plt.fill_between(x_vals, y_m + y_std ,y_m - y_std,alpha = 0.3)
    plt.ylim([0, 1.1])
    plt.xlim([0, n_readings])

plt.legend()
plt.title('Training')
plt.xlabel('Epoch')
plt.ylabel('Reward')

plt.subplot(1,2,2)

x_vals = np.arange(n_readings)
for op in [0,1]:
    vals = cross_plays[op]
    y_m = vals.mean(0)
    y_std = vals.std(0) / ( n_runs**0.5 )
    plt.plot( x_vals, y_m, colors[op],label = mode_labels[ op ])
    plt.fill_between(x_vals, y_m + y_std ,y_m - y_std, color= colors[op],alpha = 0.3)
    plt.ylim([0, 1.1])
    plt.xlim([0, n_readings])

# plt.legend()
None

plt.title('Testing (Zero-Shot)')
plt.xlabel('Epoch')
# plt.ylabel('Reward')


plt.savefig('matrix_game.pdf')

plt.show()

# Uncomment line below if you would like to download the plot to your local machine.
# files.download("matrix_game.pdf") 