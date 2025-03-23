import gym
from stable_baselines3 import PPO
import numpy as np
from drone_control import TelloController
from gym import spaces
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILENAME = "drone_rl_full_eeg_agent"

class DroneControlEnv(gym.Env):
    def __init__(self, connect_drone=False):
        super(DroneControlEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)  # Update observation space
        self.current_state = None
        self.connect_drone = connect_drone
        if connect_drone:
            self.drone_controller = TelloController()
        self.drone_connected = False
        self.model = None
        self.logger = logging.getLogger(__name__)

    def connect_drone(self):
        if self.connect_drone and not self.drone_connected:
            self.drone_connected = self.drone_controller.connect()
            if self.drone_connected:
                self.logger.info("Drone connected successfully")
            else:
                self.logger.error("Failed to connect to drone")
        return self.drone_connected

    def reset(self):
        self.current_state = np.zeros(16)  # Update state size
        return self.current_state

    def step(self, action, human_feedback=None):
        if self.connect_drone and self.drone_connected:
            forward_backward_speed = int(action[0] * 100)
            left_right_speed = int(action[1] * 100)
            
            print(f"Drone action: FB={forward_backward_speed}, LR={left_right_speed}")
            
            self.drone_controller.set_forward_backward_speed(forward_backward_speed)
            self.drone_controller.set_left_right_speed(left_right_speed)
            self.drone_controller.send_rc_control()
        else:
            print(f"Simulated drone action: {action}")

        reward = 0
        if human_feedback is not None:
            reward = 1 if human_feedback else -1
        elif self.connect_drone and self.drone_connected and (forward_backward_speed != 0 or left_right_speed != 0):
            reward = 0.1

        done = False
        info = {}

        return self.current_state, reward, done, info

    def update_state(self, new_state):
        self.current_state = new_state
        print(f"Updated state: {self.current_state}")

    def load_or_create_model(self):
        if os.path.exists(f"{MODEL_FILENAME}.zip"):
            try:
                self.model = PPO.load(MODEL_FILENAME, env=self)
                self.logger.info("Loaded existing model")
            except Exception as e:
                self.logger.error(f"Error loading existing model: {e}. Creating a new model instead.")
                self.model = PPO("MlpPolicy", self, verbose=1)
                self.logger.info("Created new model")
        else:
            self.model = PPO("MlpPolicy", self, verbose=1)
            self.logger.info("Created new model")
        return self.model

    def train_step(self, eeg_data):
        state = np.array(eeg_data)
        self.update_state(state)
        action, _states = self.model.predict(state, deterministic=False)
        
        print(f"RL Agent state: {state}")
        print(f"RL Agent action: {action}")

        return action
