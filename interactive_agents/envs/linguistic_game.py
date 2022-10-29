from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

# NOTE: This enviroment is better suited to the AEC representation, but AEC is hard to integrate with RL - how hard would this actually be?
# NOTE: There is a simpler speaker-listener scenario, where only the listener actually takes actions
# NOTE: Behavioral cloning would be a good starting point to test the representational complexity of this task
class LinguisticCoordination(MultiagentEnv):
    """
    The 'linguistic' coordination game, in which a speaker privately observes
    a cue which determines the optimal joint action, and uses a 'cheap-talk'
    channel to communicate this before both agents act.
    """

    def __init__(self, config={}, spec_only=False):
        self._num_steps = config.get("stages", 5) * 2  # NOTE: "steps" here refers to individual MDP steps, each "round" consists of two steps
        self._num_actions = config.get("actions", 8)
        self._meta_learning = config.get("meta_learning", False)
        # NOTE: "Other-Play" is not implemented for this environment

        # NOTE: Speaker observes the true signal in one round, and its partner's action in the other, with different inputs for each
        self.observation_spaces = {
            "speaker": Box(0, 1, shape=(self._num_actions * 2 + 1,))
        }

        self.action_spaces = {
            "speaker": Discrete(self._num_actions)  # NOTE: Either a cheap-talk signal or an actual action
        }  # NOTE: Direct overlap between cheap-talk signals and real actions may actually make the task harder, as it would be difficult to recognize the differences between actions in different contexts
        
        if not self._meta_learning:
            self.observation_spaces["listener"] = Box(0, 1, shape=(self._num_actions + 1,))
            self.action_spaces["listener"] = Discrete(self._num_actions)

        self._current_step = 0
        self._current_type = 0  # NOTE: "type" here seems to refer to the joint action that will receive a reward in the current round

        self._fixed_language = None  # NOTE: Likely only used for meta-learning
        self._last_statement = None

    def reset(self):
        self._current_type = np.random.randint(0, self._num_actions)  # NOTE: We set a new type at the start of the episode
        self._current_step = 0

        obs = {}
        obs["speaker"] = np.zeros(self._num_actions * 2 + 1)  # NOTE: Everything is zero except for the "type" in the first stage of a round
        obs["speaker"][self._num_actions + self._current_type] = 1  # NOTE: This may not be consistent with later stages
        
        if not self._meta_learning:
            obs["listener"] = np.zeros(self._num_actions + 1)
        else:
            self._fixed_language = np.random.permutation(self._num_actions)  # NOTE: Check what the numpy "permutations" function returns

        return obs

    def step(self, actions):
        if self._current_step % 2 == 0:  # Last stage was a communication stage
            reward = 0  # NOTE: No reward recieved for taking an action in a communication stage

            obs = {}
            obs["speaker"] = np.zeros(self._num_actions * 2 + 1)
            obs["speaker"][self._num_actions + self._current_type] = 1 # NOTE: The type seems to always be visible, regardeless of the current stage
            obs["speaker"][-1] = 1  # NOTE: signal that this is the start of the "action" stage

            if not self._meta_learning:
                obs["listener"] = np.zeros(self._num_actions + 1)
                obs["listener"][actions["speaker"]] = 1
                obs["listener"][-1] = 1
            else:
                self._last_statement = actions["speaker"]  # NOTE: We only keep track of the cheap-talk signal internally when using meta-learning

        else:  # Last stage was a coordination stage
            if not self._meta_learning:
                listener_action = actions["listener"]
            else:
                listener_action = self._fixed_language[self._last_statement]  # NOTE: This is where meta-learning comes in
                self._last_statement = None

            reward = 1 if actions["speaker"] == listener_action else 0
            reward = reward if actions["speaker"] == self._current_type else 0

            self._current_type = np.random.randint(0, self._num_actions)

            obs = {}  # NOTE: Observation now includes both the new type, and the last action the partner took
            obs["speaker"] = np.zeros(self._num_actions * 2 + 1)
            obs["speaker"][listener_action] = 1 
            obs["speaker"][self._num_actions + self._current_type] = 1

            if not self._meta_learning:
                obs["listener"] = np.zeros(self._num_actions + 1)
                obs["listener"][actions["speaker"]] = 1

        rewards = {"speaker": reward}

        self._current_step += 1
        done = (self._current_step >= self._num_steps)
        dones = {"speaker": done}

        if not self._meta_learning:
            rewards["listener"] = reward
            dones["listener"] = done

        return obs, rewards, dones, None
