import numpy as np

class CoordinationGame:

    def __init__(self, 
            num_players=2,
            num_actions=10,
            noise=0.0,
            focal_point=False,
            focal_payoff=0.9,
            other_play=False):
        self._num_players = num_players
        self._num_actions = num_actions
        self._noise = noise
        self._focal_point = focal_point
        self._focal_payoff = focal_payoff
        self._other_play = other_play

    @property
    def num_players(self):
        return self._num_players
    
    def num_actions(self, player):
        return self._num_actions

    def play(self, actions):
        actions = actions.copy()
        if self._other_play:
            for player in range(1, self._num_players):
                if self._focal_point and 0 != actions[player]:
                        actions[player] = np.random.randint(1, self._num_actions)
                elif not self._focal_point:
                    actions[player] = np.random.randint(self._num_actions)

        if not all([a == actions[0] for a in actions[1:]]):
            reward = 0.0
        elif 0 == actions[0] and self._focal_point:
            reward = self._focal_payoff
        else:
            reward = 1.0
            
            if self._noise > 0.0:
                reward += self._noise * np.random.random()
        
        return np.full(self._num_players, reward)
    
    def means(self, policies):
        dist = np.ones(self._num_actions)
        for policy in policies:
            dist *= policy

        mean = self._focal_point if self._focal_point else 1.0
        mean *= dist[0]
        mean += np.sum(dist[1:])
        
        return np.full(self._num_players, mean)
