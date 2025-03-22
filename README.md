# tello_epoch_rl
This project aims to control a Tello drone using real-time EEG data streamed from an Emotiv EPOC+ headset. The EEG data, along with gyroscope data, is used to train a Reinforcement Learning (RL) agent that learns to map brain signals to tello drone control commands.



# 


Let us break down the reinforcement learning (RL) parts of this project, as if. We'll focus on the key concepts, the code, and the reasoning behind the design choices.

**The Big Picture: Reinforcement Learning**

Think of it as a robot. You give the robot commands (actions), and the robot does something. If the robot does something good, you give it a pat (reward). If it does something bad, you might say "No!" (negative reward). The robot learns over time which actions lead to rewards.

Reinforcement learning is similar, but instead of a robot, we have an *agent* (in this case, the drone), and instead of treats, we have numerical rewards. The agent's goal is to learn a *policy* – a strategy that tells it what action to take in any given situation to maximize its rewards.

**Key Components in the Code**

The main code related to Reinforcement Learning is in `realtimelearning_rlagent.py`.

1. **The Environment (`DroneControlEnv` class):**
    * Think of the environment as the world in which the agent operates. In our case, the environment is the simulated or real drone and its surroundings.
    * **`action_space`:** This defines the set of possible actions the agent can take. Here, it's a `spaces.Box` with two values:
        * The `Box` means the action values are continuous, meaning they can be any decimal number between a range.
        * `low=-1.0, high=1.0`: This means the action values can be between -1.0 and 1.0.
        * `shape=(2,)`: This means there are two action values.
        * `dtype=np.float32`: This tells us the action values are floating-point numbers, which are decimal numbers
        * In our case, the 2 values represents the forward/backward speed and the left/right speed of the drone. The values are normalized between -1.0 and 1.0 for the agent to train more easily.
    * **`observation_space`:** This defines the information the agent receives about the environment's current state. It's also a `spaces.Box`:
        * `low=-np.inf, high=np.inf`: The observation values can be any number from negative infinity to positive infinity.
        * `shape=(7,)`: This means there are 7 observation values.
        * In our case, the 7 values are the gyro X, gyro Y, and 5 PCA components of the EEG data. These values are used to represent the agent's understanding of the current state of the world.
    * **`reset()`:** This function resets the environment to a starting state.  It's like starting a new episode of training.
    * **`step(action)`:** This is the most important function. It takes an `action` from the agent, applies it to the environment (e.g., sends a command to the drone), and returns:
        * `observation`: The new state of the environment after the action.
        * `reward`: A numerical value indicating how good the action was.
        * `done`: A boolean value indicating whether the episode is over (e.g., the drone crashed or reached its goal).
        * `info`: Additional information (usually empty in this case).
        * **Rewarding the Agent**: We give a small positive reward (0.1) if the drone is moving forward or sideways. This encourages the drone to take actions that result in movement.
2. **The Agent (PPO - Proximal Policy Optimization):**
    * The agent is the brain that decides what actions to take. We're using a PPO (Proximal Policy Optimization) agent from the `stable_baselines3` library.
    * **Why PPO?** PPO is a popular RL algorithm that's known for being relatively stable and easy to tune. It's a good choice for continuous action spaces like ours.
    * **`PPO("MlpPolicy", self, verbose=1)`:** This line creates a PPO agent with a "MlpPolicy".
        * **MlpPolicy:** This means the agent's policy (the strategy it uses to choose actions) is represented by a Multi-Layer Perceptron, which is a type of neural network. Neural networks are good at learning complex patterns from data.
        * **`self`:** This tells the PPO agent to use our `DroneControlEnv` as the environment.
        * **`verbose=1`:** This means the agent will print out some information about its training progress.
3. **The Policy:**
    * The policy is how the agent maps a state (observation) to an action. In simpler terms, it's the agent's brain. The agent makes predictions from this brain and applies it to the environment.
    * The agent updates its policy to maximize reward using the PPO model.

**Why These Design Choices?**

* **Continuous Action Space:** We use a continuous action space (`spaces.Box`) because the drone's speed can be any value within a certain range, not just a few discrete options.
* **Neural Network Policy:** We use a neural network (`MlpPolicy`) to represent the policy because it can learn complex relationships between the EEG data and the optimal drone control actions. EEG data is complex, so a neural network is well-suited to this task.
* **Reward Function:** The reward function is designed to encourage the drone to move. A more sophisticated reward function could be used to encourage the drone to follow a specific path or avoid obstacles.
* **Proximal Policy Optimization (PPO):** It strikes a good balance between being relatively easy to implement and tune, while still being able to learn complex policies.

**How It All Works Together**

1. The `visualizer_epoch.py` script collects EEG data from the Emotiv headset.
2. The `emotiv_streamer_rl.py` script preprocesses the EEG data and extracts relevant features (band power, PCA components).
3. The preprocessed EEG data is fed into the `train_agent()` function in `realtimelearning_rlagent.py`.
4. The `train_agent()` function uses the PPO agent to predict an action based on the EEG data.
5. The `step()` function in the `DroneControlEnv` class applies the action to the drone (sends a command to the drone).
6. The `step()` function also calculates a reward based on the drone's movement.
7. The PPO agent uses the reward to update its policy (the neural network) so that it will be more likely to choose actions that lead to higher rewards in the future.
8. The process repeats, allowing the agent to learn over time how to control the drone using EEG data.

**Intuition and Analogy**

Think of it like learning to ride a bike:

* **State:** Your current balance, speed, and the angle of the bike.
* **Action:** Steering the handlebars, pedaling, applying the brakes.
* **Reward:** Staying upright and moving forward.
* **Policy:** Your brain's strategy for adjusting your actions to maintain balance and move forward.

Initially, you might wobble and fall a lot (negative rewards). But over time, your brain learns the right adjustments to make (updates the policy) to stay balanced and move forward (maximize rewards).

**The Human-in-the-Loop Part**

The added human-in-the-loop component allows you to act as a "teacher" for the RL agent. By providing feedback on the agent's actions (approving or disapproving), you can help the agent learn more quickly and safely. It's like having a driving instructor who tells you when you're doing something right or wrong.

Reinforcement learning can be a bit tricky to grasp at first, but with practice and experimentation, you'll get the hang of it. Let me know if you have any other questions.

<div style="text-align: center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/70686d52-9418-465d-b9e8-f0fb44e8b700/control_drone.py

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/35cf0bd3-50b8-4231-8533-3253a66cef85/realtimelearning_rlagent.py

[^3]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/7ddc0e3c-1f3d-4bc6-827d-b79d42ec97a0/visualizer_epoch.py

[^4]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/065ef27b-834c-45ed-a859-f79ade37e5be/realtime_visualizer_2D.py

[^5]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/f40564e8-b173-411f-bf64-55e8c2df5a9f/emotiv_streamer_rl.py

