"""Test of Pyglet visualization for Coordination game"""
from gym.spaces import Discrete, Box
import numpy as np
import pyglet
import time


class CoordinationGame:

    def __init__(self, config={}, render_config={}):  # NOTE: Simplify for testing by removing other-play permutations, focal points
        self._num_stages = config.get("stages", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)

        self._obs_size = self._num_actions * (self._num_players - 1)

        self.observation_space = {}
        self.action_space = {}

        for pid in range(self._num_players):
            self.observation_space[pid] = Box(0, 1, shape=(self._obs_size,))
            self.action_space[pid] = Discrete(self._num_actions)
        
        self._current_stage = 0
        self._prev_actions = None

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

    def reset(self):
        self._current_stage = 0
        return self._obs()

    def step(self, actions):

        # Encode actions into next observation
        obs = self._obs(actions)
        
        # Compute global reward
        if all(a == actions[0] for a in actions.values()):
            reward = 1
        else:
            reward = 0

        rewards = {pid:reward for pid in range(self._num_players)}

        # Determine if final stage reached
        self._current_stage += 1
        done = self._num_stages <= self._current_stage
        dones = {pid:done for pid in range(self._num_players)}

        return obs, rewards, dones, None

    def _open_window(self):
        # Open window
        width = self._cell_size * self._num_stages
        height = self._cell_size
        config = pyglet.gl.Config(sample_buffers=1, samples=4)
        self._window = pyglet.window.Window(width, height, 
                            config=config, caption="Coordination Game")

        @self._window.event
        def on_close():
            self._window.close()
            self._window = None  # Allow window to be garbage collected
            self._window_closed = True

        # Initialize text overlay
        self._labels = []
        x_pos = self._cell_size // 2
        y_pos = self._cell_size // 10

        for stage in range(self._num_stages):
            self._labels.append(pyglet.text.Label(f"stage {stage + 1}",
                          font_name="Arial",
                          font_size=36,
                          x=x_pos, y=y_pos,
                          anchor_x="center", anchor_y="center",
                          color=(1,1,1,1), bold=True))
            x_pos += self._cell_size

    def _draw(self):
        
        # Set window as current OpenGL context
        self._window.switch_to()

        # Clear window
        pyglet.gl.glClearColor(0, 0, 0, 1)
        self._window.clear()

        # Draw text overlay
        for label in self._labels:
            print("label draw")
            label.draw()

    def render(self):  # TODO: Drop the "render" method altogether and let the environment implement its own loop

        # Open window if needed
        if self._window is None and not self._window_closed:
            self._open_window()

        # Handle window events - allow window to close
        if not self._window_closed:
            self._window.dispatch_events()

        # Render frame to window - need to check that window hasn't closed
        if not self._window_closed:
            self._draw()

    @property
    def window_closed(self):
        return self._window_closed

    def visualize(self, 
                  policies={}, 
                  max_episodes=None, 
                  max_steps=None,
                  step_interval=2.0,
                  cell_size=200,
                  max_width=1600):
        
        # NOTE: Pyglet seems a bit flaky, does it have anything to do with WSL?

        # Enforce maximum window width
        if cell_size * self._num_stages > max_width:
            cell_size = max_width // self._num_stages

        # Create window
        width = cell_size * self._num_stages
        height = cell_size
        config = pyglet.gl.Config(sample_buffers=1, samples=4)  # Needed for anti-aliasing
        window = pyglet.window.Window(width, height, 
                    config=config, caption="Coordination Game")

        # Initialize geometry

        # Initialize text overlay
        labels = []
        x_pos = cell_size // 2
        y_pos = cell_size // 10

        for stage in range(self._num_stages):
            labels.append(pyglet.text.Label(f"stage {stage + 1}",
                          font_name="Arial",
                          font_size=16,
                          x=x_pos, y=y_pos,
                          anchor_x="center", anchor_y="center",
                          color=(255,255,255,255), bold=True))
            x_pos += cell_size

        # TODO: Schedule environment updates

        # Add drawing function
        @window.event
        def on_draw():

            # Clear window
            window.clear()
            
            # Render geometry


            # Draw text overlay
            for label in labels:
                label.draw()

        # Launch game loop - returns when window is closed or max-episodes reached
        pyglet.app.run()


if __name__ == "__main__":
    NUM_EPISODES = 100
    FPS = 0.5

    # Initialize environment
    env = CoordinationGame(
        config={
            "stages": 5,
            "actions": 8,
            "players": 2,
        })
    

    # Launch visualization
    env.visualize()  # NOTE: Is it possible that there is some environment variable we are missing here?

    # Run episodes
    '''
    sleep = 1.0 / FPS

    for episode in range(NUM_EPISODES):
        print(f"Episode {episode}")

        obs = env.reset()
        dones = {"__all__": False}

        env.render()
        time.sleep(sleep)

        while not all(dones.values()):
            actions = {}
            for id, space in env.action_space.items():
                actions[id] = space.sample()
            
            obs, reward, dones, _ = env.step(actions)

            env.render()

            if env.window_closed:
                exit()
            
            time.sleep(sleep)
    '''
