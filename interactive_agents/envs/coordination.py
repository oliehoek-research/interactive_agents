from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

class CoordinationGame(MultiagentEnv):
    """
    The N-player repeated coordination game.  Players observe all other player's
    previous actions as concatenated one-hot vectors.

    Supports 'focal point' actions that have slightly smaller payoffs, and
    suppots permuting action IDs used to implement Other-Play.
    """

    def __init__(self, config={}, spec_only=False):
        self._num_stages = config.get("stages", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)
        self._focal_point = config.get("focal_point", False)
        self._focal_payoff = config.get("focal_payoff", 0.9)
        self._noise = config.get("payoff_noise", 0.0)
        self._other_play = config.get("other_play", False)

        self._obs_size = self._num_actions * (self._num_players - 1)

        self.observation_spaces = {}
        self.action_spaces = {}

        for pid in range(self._num_players):
            self.observation_spaces[pid] = Box(0, 1, shape=(self._obs_size,))
            self.action_spaces[pid] = Discrete(self._num_actions)
        
        self._current_stage = 0
        self._num_episodes = 0

        # Action permutations for other-play
        self._forward_permutations = None
        self._backward_permutations = None

        # Variables used for visualization
        self._prev = []
        self._last_obs = None

    def _new_permutations(self):  # TODO: Implement unit tests to make sure this works properly (sample tests are available in the 'junk/other_play_test.py' script)
        self._forward_permutations = {}
        self._backward_permutations = {}
        
        for pid in range(self._num_players):
            if 0 == pid:
                forward = np.arange(self._num_actions)
            elif self._focal_point:
                forward = 1 + np.random.permutation(self._num_actions - 1)
                forward = np.concatenate([np.zeros(1,dtype=np.int64), forward])
            else:
                forward = np.random.permutation(self._num_actions)

            backward = np.zeros(self._num_actions, dtype=np.int64)
            for idx in range(self._num_actions):
                backward[forward[idx]] = idx

            self._forward_permutations[pid] = forward
            self._backward_permutations[pid] = backward
    
    def _permuted_obs(self, actions):
        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)
            index = 0

            for id, action in actions.items():
                if pid != id:
                    action = self._backward_permutations[pid][action]
                    obs[pid][index + action] = 1
                    index += self._num_actions
        
        return obs

    def _obs(self, actions=None):
        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)

        if actions is not None:
            for pid in range(self._num_players):
                index = 0

                for id, action in actions.items():
                    if pid != id:
                        obs[pid][index + action] = 1
                        index += self._num_actions
        
        return obs

    def _step(self, actions):

        # Generate reward noise if needed
        if self._noise > 0:
            noise = self._noise * np.random.normal()
        else:
            noise = 0

        # Compute global reward
        if self._focal_point and all(a == 0 for a in actions.values()):
            reward = self._focal_payoff
        elif all(a == actions[0] for a in actions.values()):
            reward = 1 + noise
        else:
            reward = 0 + noise

        rewards = {pid:reward for pid in range(self._num_players)}

        # Determine if final stage reached
        self._current_stage += 1
        done = self._num_stages <= self._current_stage
        dones = {pid:done for pid in range(self._num_players)}

        # Save previous actions for rendering
        prev_actions = [actions[pid] for pid in range(self._num_players)]  # TODO: Change IDs to strings
        self._prev.append((prev_actions, reward))

        return rewards, dones

    def reset(self):
        self._current_stage = 0
        self._prev = []

        if self._other_play:
            self._new_permutations()

        return self._obs()

    def step(self, actions):  # NOTE: This wrapper for the step function just handles the other-play permutations
        true_actions = actions.copy()  # NOTE: The action array will be used for learning, so don't modify it
        if self._other_play:
            for pid in range(self._num_players):
                true_actions[pid] = self._forward_permutations[pid][actions[pid]]

            obs = self._permuted_obs(true_actions)
        else:
            obs = self._obs(true_actions)
        
        rewards, dones = self._step(true_actions)

        return obs, rewards, dones, None

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
