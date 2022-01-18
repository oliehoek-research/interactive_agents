import numpy as np

from coordination_game import CoordinationGame

class QLearner:

    def __init__(self, 
            num_actions, 
            epsilon=0.05, 
            lr=0.1,
            initial=0.0,
            random_argmax=False):
        self._num_actions = num_actions
        self._epsilon = epsilon
        self._lr = lr
        self._random_argmax = random_argmax
        
        self._Q = np.full(self._num_actions, initial)
    
    def act(self):
        if np.random.random() < self._epsilon:
            return np.random.randint(self._num_actions)
        elif self._random_argmax:
            actions = np.argwhere(self._Q == self._Q.max())
            return np.random.choice(actions.reshape(-1))
        else:
            return self._Q.argmax()
    
    def learn(self, action, reward):
        self._Q[action] += self._lr * (reward - self._Q[action])

    @property
    def policy(self):
        policy = np.zeros(self._num_actions)

        if self._random_argmax:
            actions = np.argwhere(self._Q == self._Q.max()).reshape(-1)
            policy[actions] = 1.0 / len(actions)
        else:
            policy[self._Q.argmax()] = 1.0
        
        return policy

    @property
    def Q(self):
        return self._Q

SEEDS = 10
ROUNDS = 10000

if __name__ == "__main__":
    game = CoordinationGame(num_actions=10, focal_point=True, other_play=True, focal_payoff=0.9)

    means_total = 0
    policies_total = []
    for player in range(game.num_players):
        policies_total.append(np.zeros(game.num_actions(player)))

    for seed in range(SEEDS):
        np.random.seed(seed)

        learners = []
        for player in range(game.num_players):
            learners.append(QLearner(
                game.num_actions(player), 
                epsilon=0.1, 
                lr=0.1, 
                initial=0.0, 
                random_argmax=False))
        
        print(f"Q-learning - seed {seed}")

        for round in range(ROUNDS):
            actions = [learner.act() for learner in learners]
            rewards = game.play(actions)

            for player, learner in enumerate(learners):
                learner.learn(actions[player], rewards[player])
        
        policies = [learner.policy for learner in learners]
        mean = game.means(policies)[0]
        means_total += mean

        print(f"mean joint payoff: {mean}")

        for player, policy in enumerate(policies):
            policies_total[player] += policy
            print(f"  policy {player}: {policy}:")
    
    print(f"\nAverage joint payoff: {means_total / SEEDS}")
    for player, policy in enumerate(policies_total):
        print(f"    mean policy {player}: {policy / SEEDS}")
