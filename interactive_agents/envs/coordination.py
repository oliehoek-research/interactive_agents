from gymnasium.spaces import Discrete, Box
import numpy as np

from .common import SyncEnv

class Coordination(SyncEnv):
    """
    The N-player repeated coordination game.  Players observe all other player's
    previous actions as concatenated one-hot vectors.

    Supports 'focal point' actions that have slightly smaller payoffs, and
    supports permuting action IDs used to implement Other-Play.  Also allows
    us to specify that other agents should all play a fixed action that is
    selected randomly at the start of each episode.
    """

    def __init__(self, config={}):
        self._num_stages = config.get("stages", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)
        self._focal_point = config.get("focal_point", False)
        self._focal_payoff = config.get("focal_payoff", 0.9)
        self._noise = config.get("payoff_noise", 0.0)
        self._other_play = config.get("other_play", False)
        self._meta_learning = config.get("meta_learning", False)

        self._obs_size = self._num_actions * (self._num_players - 1)

        self._pids = [f"agent_{pid}" for pid in range(self._num_players)]

        self.observation_spaces = {}
        self.action_spaces = {}
        self._truncated = {}

        for pid in self._pids:
            if (not self._meta_learning) or "agent_0" == pid:
                self.observation_spaces[pid] = Box(0, 1, shape=(self._obs_size,))
                self.action_spaces[pid] = Discrete(self._num_actions)
                self._truncated[pid] = False
        
        self._current_stage = 0
        self._num_episodes = 0

        self._rng = None

        # Action permutations for other-play
        self._forward_permutations = None
        self._backward_permutations = None

        # Fixed agent action for meta-learning
        self._fixed_agent_action = None

        # Variables used for visualization
        self._prev = []
        self._last_obs = None

    # TODO: Implement unit tests to make sure this works properly (sample tests are available in the 'junk/other_play_test.py' script)
    def _new_permutations(self):
        self._forward_permutations = {}
        self._backward_permutations = {}
        
        for policy_idx, pid in enumerate(self._pids):
            if 0 == policy_idx:
                forward = np.arange(self._num_actions)
            elif self._focal_point:
                forward = 1 + self._rng.permutation(self._num_actions - 1)
                forward = np.concatenate((np.zeros(1,dtype=np.int64), forward))
            else:
                forward = self._rng.permutation(self._num_actions)

            backward = np.zeros(self._num_actions, dtype=np.int64)
            for idx in range(self._num_actions):
                backward[forward[idx]] = idx

            self._forward_permutations[pid] = forward
            self._backward_permutations[pid] = backward
    
    def _permuted_obs(self, actions):
        obs = {}
        for pid in self._learning_pids():
            obs[pid] = np.zeros(self._obs_size)
            index = 0

            for other_pid in self._pids:
                if pid != other_pid:
                    action = self._backward_permutations[pid][actions[other_pid]]
                    obs[pid][index + action] = 1
                    index += self._num_actions
        
        return obs

    def _reset_fixed_action(self):
        self._fixed_agent_action = self._rng.integers(self._num_actions)

    def _learning_pids(self):
        if self._meta_learning:
            return self._pids[:1]
        else:
            return self._pids

    def _expand_actions(self, actions):
        for pid in self._pids[1:]:
            actions[pid] = self._fixed_agent_action

        return actions

    def _obs(self, actions=None):
        pids = self._learning_pids()

        obs = {}
        for pid in pids:
            obs[pid] = np.zeros(self._obs_size)

        if actions is not None:
            for pid in pids:
                index = 0

                for other_pid in self._pids:
                    if pid != other_pid:
                        obs[pid][index + actions[other_pid]] = 1
                        index += self._num_actions
        
        return obs

    def _step(self, actions):
        pids = self._learning_pids()

        # Generate reward noise if needed
        if self._noise > 0:
            noise = self._noise * self._rng.normal()
        else:
            noise = 0

        # Compute global reward
        if self._focal_point and all(a == 0 for a in actions.values()):
            reward = self._focal_payoff
        elif all(a == actions["agent_0"] for a in actions.values()):
            reward = 1 + noise
        else:
            reward = 0 + noise

        rewards = {pid:reward for pid in pids}

        # Determine if final stage reached
        self._current_stage += 1
        done = self._num_stages <= self._current_stage
        dones = {pid:done for pid in pids}

        # Save previous actions for rendering
        prev_actions = [actions[pid] for pid in self._pids]
        self._prev.append((prev_actions, reward))

        return rewards, dones

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        self._current_stage = 0
        self._prev = []

        if self._other_play:
            self._new_permutations()
        
        if self._meta_learning:
            self._reset_fixed_action()

        return self._obs()

    def step(self, actions):  # Wrapper to support action permutation
        actions = actions.copy()  # Need a copy we can modify

        if self._meta_learning:
            actions = self._expand_actions(actions)

        if self._other_play:
            for pid in self._pids:
                actions[pid] = self._forward_permutations[pid][actions[pid]]

            obs = self._permuted_obs(actions)
        else:
            obs = self._obs(actions)
        
        rewards, dones= self._step(actions)

        return obs, rewards, dones, self._truncated, None

    # TODO: Implement the "render" method from the Gym/PettingZoo API
    
    def visualize(self,
                  policies={},
                  policy_fn=None,
                  max_episodes=None, 
                  max_steps=None,
                  speed=1,
                  record_path=None,
                  headless=False,
                  cell_size=200,
                  max_width=1600,
                  **kwargs):
        # NOTE: There is a bug in VcXsrv with WSL that causes problems if we move the window
        import pyglet
        from pyglet.gl import Config

        # TODO: Implement recording functionality
        if record_path is not None:
            raise NotImplementedError("Recording is not yet supported for this environment")
        
        if headless:
            raise NotImplementedError("Headless visualization is not yet supported for this environment")

        # Enforce maximum window width
        if cell_size * self._num_stages > max_width:
            cell_size = max_width // self._num_stages

        # Create window
        width = cell_size * self._num_stages
        height = cell_size
        config = pyglet.gl.Config(sample_buffers=1, samples=4)  # NOTE: Needed for anti-aliasing
        window = pyglet.window.Window(width, height, 
                    config=config, caption="Coordination Game")

        # Initialize text overlay
        labels = []
        x_pos = cell_size // 2
        y_pos = cell_size // 10

        for stage in range(self._num_stages):
            labels.append(pyglet.text.Label(f"stage {stage + 1}",
                          font_name="Arial",
                          font_size=14,
                          x=x_pos, y=y_pos,
                          anchor_x="center", anchor_y="center",
                          color=(255,255,255,255), bold=True))
            x_pos += cell_size

        # Initialize geometry
        offset = cell_size // 5
        row_size = (cell_size - 2*offset) // self._num_actions
        column_size = (cell_size - 2*offset) // self._num_players

        fail_icon = pyglet.shapes.Rectangle(0, 0, column_size, row_size, color=(255,0,0))
        good_icon = pyglet.shapes.Rectangle(0, 0, column_size, row_size, color=(0,255,0))

        # Add drawing function
        @window.event
        def on_draw():

            # Clear window
            window.clear()
            
            # Render geometry
            glLoadIdentity()

            for stage, (actions, reward) in enumerate(self._prev):
                if 0 == reward:
                    icon = fail_icon
                else:
                    icon = good_icon
                
                glPushMatrix()
                glTranslatef(offset + stage * cell_size, offset, 0)

                for player, action in enumerate(actions):
                    glPushMatrix()
                    glTranslatef(column_size * player, row_size * action, 0)
                    icon.draw()
                    glPopMatrix()

                glPopMatrix()

            # Draw text overlay
            for label in labels:
                label.draw()

        # Initialize agents and environment
        agents = {} 
        for agent_id in self.action_space.keys():
            if policy_fn is not None:
                policy_id = policy_fn(agent_id)
            else:
                policy_id = agent_id
            
            agents[agent_id] = policies[policy_id].make_agent()

        self._num_episodes = 0
        self._last_obs = self.reset()

        # Schedule environment updates
        if max_steps is None:
            max_steps = self._num_stages
        else:
            max_steps = min(self._num_stages, max_steps)

        def update(dt):
            if self._current_stage >= max_steps:
                if max_episodes is not None and self._num_episodes >= max_episodes:
                    pyglet.app.exit()

                for agent_id in self.action_space.keys():
                    if policy_fn is not None:
                        policy_id = policy_fn(agent_id)
                    else:
                        policy_id = agent_id
            
                    agents[agent_id] = policies[policy_id].make_agent()

                self._num_episodes += 1
                self._last_obs = self.reset()
            else:
                actions = {}
                for agent_id, obs in self._last_obs.items():
                    actions[agent_id], _ = agents[agent_id].act(obs)
                
                self._last_obs, _, _, _ = self.step(actions)

        pyglet.clock.schedule_interval(update, .5 / speed)

        # Launch game loop - returns when window is closed or max-episodes reached
        pyglet.app.run()
