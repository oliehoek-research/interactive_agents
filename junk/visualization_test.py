"""Test of Pyglet visualization for Coordination game"""
from re import L
from gym.spaces import Discrete, Box
import numpy as np
import pyglet
from pyglet.gl import *


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
        self._prev = []

        self._num_episodes = 0
        self._last_obs = None

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
        self._prev = []

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

        # Save previous actions for rendering
        prev_actions = [actions[pid] for pid in range(self._num_players)]
        self._prev.append((prev_actions, reward))

        return obs, rewards, dones, None

    def visualize(self, 
                  policies={}, 
                  max_episodes=None, 
                  max_steps=None,
                  step_interval=1.5,
                  cell_size=200,
                  max_width=1600):
        # NOTE: There is a bug in VcXsrv with WSL that causes problems if we move the window

        # Enforce maximum window width
        if cell_size * self._num_stages > max_width:
            cell_size = max_width // self._num_stages

        # Create window
        width = cell_size * self._num_stages
        height = cell_size
        config = pyglet.gl.Config(sample_buffers=1, samples=4)  # Needed for anti-aliasing
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

        # Initialize agents
        agents = {}
        for pid in self.action_space.keys():
            if pid in policies:
                agents[pid] = policies[pid].make_agent()

        # Schedule environment updates
        if max_steps is None:
            max_steps = self._num_stages
        else:
            max_steps = min(self._num_stages, max_steps)

        def update(dt):
            if self._current_stage >= max_steps:
                self._num_episodes += 1

                if max_episodes is not None and self._num_episodes >= max_episodes:
                    pyglet.app.exit()

                self._last_obs = self.reset()
            else:
                actions = {}
                for pid, space in self.action_space.items():
                    if pid in agents:
                        actions[pid] = agents[pid].act(self._last_obs)
                    else:
                        actions[pid] = space.sample()
                
                self._last_obs, _, _, _ = self.step(actions)

        self._num_episodes = 0
        self._last_obs = self.reset()

        pyglet.clock.schedule_interval(update, step_interval)

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
    env.visualize()
