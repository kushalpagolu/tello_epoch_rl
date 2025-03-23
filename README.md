# EEG-Based Drone Control with Reinforcement Learning

## Overview

This project aims to control a Tello drone using real-time EEG data streamed from an Emotiv EPOC+ headset. The EEG data, along with gyroscope data, is used to train a Reinforcement Learning (RL) agent that learns to map brain signals to tello drone control actions, such as moving forward, backward, left, and right.

## Project Structure

The project consists of several Python files:

-   `main.py`: Main entry point for running the application. It handles connecting to the Emotiv headset and Tello drone, processing data, and running the RL agent.
-   `stream_data_updated.py`: Handles communication with the Emotiv headset, decrypts the EEG data, and preprocesses it.
-   `learning_rlagent.py`: Defines the RL environment and agent, including the state space, action space, and reward function.
-   `visualizer_realtime.py`: Implements a real-time EEG data visualizer using Matplotlib.
-   `drone_control.py`: (Optional) Contains code to control the Tello drone.
-   `data/`: Directory to store the EEG and gyro data in Excel format.

## Prerequisites

Before running the project, make sure you have the following:

-   **Hardware:**
    -   Emotiv headset (EPOC+, Insight, or Flex)
    -   Tello drone (or a simulator)
    -   Computer with Python 3.6+
-   **Software:**
    -   Python 3.6+
    -   Required Python packages (see `requirements.txt`)
    -   Emotiv drivers and software

### Installation

1.  Clone the repository:

    ```
    git clone https://github.com/kushalpagolu/tello_epoch_rl
    cd tello_epoch_rl
    ```

2.  Install the required Python packages and or use virtual environment if you prefer that way:

    ```
    pip install -r requirements.txt
    ```

    If have any issues whiles using `requirements.txt` file, install with the commands below or install each library with "pip install *****" command:

    (Or)Create a virtual environment:

    Navigate to your project directory in the terminal and run:
    
    ```
    python3 -m venv venv
    ```
    This creates a new virtual environment named venv in your project directory.
    
    Activate the virtual environment:
    
    On macOS and Linux:
    
    ```
    source venv/bin/activate
    ```
    On Windows:
    
    ```
    .\venv\Scripts\activate
    ```
    Once the environment is activated, you'll see its name (venv) in parentheses at the beginning of your terminal prompt.
    
    Install the required packages:
    
    Make sure you are in the project directory and the virtual environment is activated. Then, run:
    
    ```
    pip install pycryptodome hid djitellopy numpy matplotlib pandas gym stable-baselines3 python-dotenv keyboard
    ```

    This will install all the necessary packages into your virtual environment.

    **_If you want to run the project in a your own system without virtual environment, then simply run the below command_**

    ```
    pip install pycryptodome hid djitellopy numpy matplotlib pandas gym stable-baselines3 python-dotenv keyboard
    ```

4.  Install the Emotiv drivers and software:

    -   Download and install the appropriate hid drivers for your code if pip did not install hid.
    -   Make sure your headset is properly connected and recognized by your computer.

5.  Install the Tello drone libraries:

    -   Refer to the Tello drone documentation for instructions on how to install the required libraries.
    -   Alternatively, you can use a Tello drone simulator.

### Usage

1.  Connect the Emotiv headset to your computer.
2.  Connect the Tello drone to your computer (or start the simulator).
3.  Run the `main.py` script without connecting drone to test streamer:

    ```
    python main.py   
    ```

4.  Use the `--connect-drone` flag to enable drone control when you connect both the drone and the Emotiv Epoch X headset. If you omit this flag, the script will run in streamer mode and simulate drone actions and print them on the console.

    ```
    python main.py --connect-drone    
    ```

### Code Explanation

Here's a breakdown of the key code files and their functionality:

#### `stream_data_updated.py`

This file handles communication with the Emotiv headset, decrypts the EEG data, and preprocesses it.

-   **`EmotivStreamer` class:**
    -   `__init__()`: Initializes the Emotiv streamer, including setting up the device ID, cipher key, and channel names.
    -   `connect()`: Connects to the Emotiv headset using the `hid` library.
    -   `disconnect()`: Disconnects from the Emotiv headset.
    -   `read_packet()`: Reads a packet of EEG data from the headset, decrypts it, and extracts the EEG and gyro data.
    -   `preprocess_eeg_data()`: Preprocesses the raw EEG data.

#### `learning_rlagent.py`

This file defines the RL environment and agent, including the state space, action space, and reward function.

-   **`DroneControlEnv` class:**
    -   `__init__()`: Initializes the RL environment, including defining the action space (forward/backward speed, left/right speed) and the observation space (EEG and gyro data).
    -   `reset()`: Resets the environment to its initial state.
    -   `step()`: Takes an action, applies it to the drone (or simulator), and returns the next state, reward, done flag, and info.
    -   `load_or_create_model()`: Loads a pre-trained RL model or creates a new one if none exists.
    -   `train_step()`: Trains the RL agent using the EEG data.

#### `main.py`

This is the main entry point for running the application.

-   **`main()` function:**
    -   Connects to the Emotiv headset and Tello drone.
    -   Initializes the RL environment and agent.
    -   Starts a loop to continuously read EEG data, preprocess it, and train the RL agent.
    -   Controls the drone based on the RL agent's actions.

## Data Preprocessing

The data preprocessing steps are performed in the `stream_data_updated.py` file:

1.  **Read Raw EEG Data:** The `read_packet()` function reads raw EEG data and gyroscope data from the Emotiv headset.
2.  **Extract Features:** The `preprocess_eeg_data()` function performs feature extraction.
    -   In the initial implementation, The EEG data from each channel would be passed to `calculate_band_power()`.
    -   `calculate_band_power()` is a function to calculate the band power of each EEG channel. It uses Welch's method to estimate the power spectral density (PSD) of the EEG data and then sums the PSD values within the frequency bands.
    -   The band power features (Delta, Theta, Alpha, Beta) for each channel are flattened into a single feature vector.
3.  **Combine with Gyro Data:** The EEG features are combined with the gyroscope data to create the final feature vector.

Please note that the current code uses the raw eeg values and does not estimate the band power, since we found that those methods were not applicable. If you want to re-implement the band power, you will need to estimate data using buffers.

## Testing

To test the project, follow these steps:

1.  Connect the Emotiv headset and Tello drone to your computer.
2.  Run the `main.py -- connect-drone` flag.
3.  Observe the drone's behavior. It should start to move based on your EEG signals.
4.  Try to control the drone by focusing on different mental tasks (e.g., thinking about moving forward, backward, left, or right).
5.  Monitor the real-time EEG data visualizer to see how your EEG signals change as you perform different mental tasks.
6.  Check that EEG and gyroscope data is being saved to the files under the data directory.
    -  If it is empty, terminate the running code via `````` to check that the last bits of code are saved as well.


# Let us break down the reinforcement learning (RL) parts of this project.

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


**How the Agent Interacts With You:**

The primary interaction occurs within the `DroneControlEnv` class, specifically in the `step` method and making use of the `get_human_feedback` method.

Here's a breakdown:

1. **Agent Proposes an Action:** The RL agent (the `PPO` model) uses the processed EEG data (`processed_data`) to predict an action. This happens in the `process_data` function within `main.py`.
2. **Action Displayed (Potentially):** The predicted action is printed to the console:  `print(f"RL Agent Suggested Action: {action}")`
3. **Human Feedback Request:** The code then pauses and asks for your approval of the action. This is done by presenting the prompt "Approve action? (Press 'y' for yes, 'n' for no, timeout=3s)" (or similar) in the console.
4. **Feedback Input:** You, the human, are expected to press either the 'y' key for "yes" (approve) or the 'n' key for "no" (reject) *within the time limit*.
5. **Action Execution (Conditional):**
    * If you approve (press 'y'), the action is sent to the drone (if `connect_drone` is True).
    * If you reject (press 'n'), a counter action is sent to revert.
6. **Reward Adjustment:** The agent receives a reward based on your feedback:
    * Positive feedback ('y') results in a positive reward (+1).
    * Negative feedback ('n') results in a negative reward (-1).
    * No feedback (timeout) results in a small reward (0.1).

**Possible Actions the Agent Might Ask You About:**

The actions are determined by the `action_space` defined in the `DroneControlEnv` and are continuous values within a specific range. The agent has 2 action features.

* **Forward/Backward Speed:** This controls the drone's movement forward or backward. The value will be a number between -1.0 and 1.0, which is then scaled to a speed value. A positive value means forward and negative means backwards.
* **Left/Right Speed:** This controls the drone's movement left or right. Similar to forward/backward speed, the value will be a number between -1.0 and 1.0. A positive value means right and negative means left.

**Example Scenario:**

1. The EEG data indicates that you are thinking about moving the drone forward and to the right.
2. The RL agent predicts the action `[0.5, 0.3]`.
3. The console displays: `"RL Agent Suggested Action: [0.5, 0.3]"` and `"Approve action? (Press 'y' for yes, 'n' for no, timeout=3s)"`.
4. If you press 'y', the drone will move forward at approximately 50% of its maximum speed and to the right at approximately 30% of its maximum speed. The agent receives a reward of +1.
5. If you press 'n', the drone will perform the inverse movement (-50% forward, -30% right, which is backwards and to the left). The agent receives a reward of -1.

**Important Considerations:**

* **Action Interpretation:** The raw action values (e.g., `[0.5, 0.3]`) might not be immediately intuitive. The agent is learning to map EEG patterns to these action values, and it's the *relative* values that matter.
* **Timeout:** If you don't provide feedback within the timeout period, the agent will proceed with the action (or a default behavior) and receive a neutral or small reward.
* **Exploration:** In the early stages of training, the agent will likely explore a wide range of actions, some of which might seem random or counterintuitive. This is a normal part of the learning process.
* 

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
7. MAX_SPEED Constant: I've defined MAX_SPEED = 50 (representing 50%) at the beginning of the file. This is the new default maximum speed. 
8. The DroneControlEnv takes a max_speed argument in its constructor (defaulting to MAX_SPEED). This value is stored as self.max_speed.
9. Speed Scaling: The step function scales the forward_backward_speed and left_right_speed by self.max_speed instead of 100. max_speed in train_drone_rl: The train_drone_rl function  also takes a max_speed argument. Command-line Configuration: The main script passes max_speed=25 or whatever value is given when the script is called.
8. The PPO agent uses the reward to update its policy (the neural network) so that it will be more likely to choose actions that lead to higher rewards in the future.
9. The process repeats, allowing the agent to learn over time how to control the drone using EEG data.

**How Learning Happens in Your Code:**

The 'train_step()' function takes EEG data, and uses the model to predict an action.
The 'step()' function applies the action to the drone, and returns the reward.
The 'model.learn(total_timesteps=1)' function takes the data from the step function, and updates the models weights.
This process is repeated many times, which causes the model to improve over time.

**Intuition and Analogy**

Think of it like learning to ride a bike:

* **State:** Your current balance, speed, and the angle of the bike.
* **Action:** Steering the handlebars, pedaling, applying the brakes.
* **Reward:** Staying upright and moving forward.
* **Policy:** Your brain's strategy for adjusting your actions to maintain balance and move forward.

Initially, you might wobble and fall a lot (negative rewards). But over time, your brain learns the right adjustments to make (updates the policy) to stay balanced and move forward (maximize rewards).

**The Human-in-the-Loop Part**

The added human-in-the-loop component allows you to act as a "teacher" for the RL agent. By providing feedback on the agent's actions (approving or disapproving), you can help the agent learn more quickly and safely. It's like having a driving instructor who tells you when you're doing something right or wrong.

Reinforcement learning can be a bit tricky to grasp at first, but with practice and experimentation, you'll get the hang of it.


## Future Improvements

-   Implement human-in-the-loop RL to improve the agent's performance.
-   Train the RL agent on a larger dataset to improve its generalization ability.
-   Add more sophisticated signal processing techniques to extract more meaningful features from the EEG data.
-   Explore different RL algorithms to find the one that works best for this task.

## Troubleshooting

If you encounter any issues while running the project, try the following:

-   Make sure all the required software packages are installed.
-   Double-check the Emotiv headset and Tello drone connections.
-   Consult the Emotiv and Tello documentation for troubleshooting tips.
-   Check the error logs for any error messages or warnings.
-   Increase tolerance and reconnection delays as stated earlier in the file.

## Contributions

Contributions to this project are welcome. Feel free to submit pull requests or open issues to report bugs or suggest new features.

## License

This project is licensed under the MIT License.



[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/70686d52-9418-465d-b9e8-f0fb44e8b700/control_drone.py

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/35cf0bd3-50b8-4231-8533-3253a66cef85/realtimelearning_rlagent.py

[^3]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/7ddc0e3c-1f3d-4bc6-827d-b79d42ec97a0/visualizer_epoch.py

[^4]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/065ef27b-834c-45ed-a859-f79ade37e5be/realtime_visualizer_2D.py

[^5]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39465668/f40564e8-b173-411f-bf64-55e8c2df5a9f/emotiv_streamer_rl.py

