import time
import numpy as np
import math
import pickle
import os

import capture_pocket_tanks as pt

# --- Environment Definition ---

class PocketTanksEnv:
    """
    A class to represent the Pocket Tanks game environment for a reinforcement learning agent.
    Each step in this environment constitutes a full turn: set angle, set power, and fire.
    """
    def __init__(self):
        self.hwnd = None
        self.rect = None
        self._setup_window()

        # Define the granularity of the action space
        self.angle_steps = 19  # 0-180 degrees in steps of 10
        self.power_steps = 11  # 0-100 in steps of 10
        
        # Action space is a combination of all possible angles and powers
        self.action_space = list(range(self.angle_steps * self.power_steps))

    def _setup_window(self):
        """Finds the game window and brings it to the front."""
        found = pt.find_window_by_title_substring("Pocket Tanks")
        if not found:
            raise Exception("Pocket Tanks window not found.")
        self.hwnd, _ = found
        pt.bring_window_to_front(self.hwnd)
        time.sleep(1)

    def get_state(self):
        """
        Captures the current game state.
        Returns a tuple of (angle, power, p1_score, p2_score, p1_pos, p2_pos).
        """
        return pt.get_state()

    def step(self, action):
        """
        Performs a full turn: sets angle, sets power, fires, and calculates reward.
        """
        pt.bring_window_to_front(self.hwnd)
        
        state_before = self.get_state()
        current_angle, current_power, p1_score_before, p2_score_before, p1_pos_before, p2_pos_before = state_before

        # --- Decode the combined action into angle and power ---
        angle_index = action // self.power_steps
        power_index = action % self.power_steps
        
        target_angle = angle_index * 20
        target_power = power_index * 10

        print(f"Executing Turn: Target Angle={target_angle}, Target Power={target_power}")
        print(current_angle,current_power)

        # --- Execute the sequence of actions ---
        # 1. Set Angle
        if current_angle is not None:
            diff = target_angle - current_angle
            print(diff)
            if diff > 0:
                pt.increase_angle(steps=diff)
            elif diff < 0:
                pt.decrease_angle(steps=-diff)
        
        time.sleep(0.1)

        # 2. Set Power
        if current_power is not None:
            diff = target_power - current_power
            if diff > 0:
                pt.increase_power(steps=diff)
            elif diff < 0:
                pt.decrease_power(steps=-diff)

        time.sleep(0.1)

        # 3. Fire and detect impact
        game_bbox = (0, 60, 1600, 900) # Define your game area bbox
        l,t,r,b = pt.get_client_screen_rect(self.hwnd)
        impact_coords = pt.predict_landing(p1_pos_before,target_angle,target_power)
        pt.fire()
        # print(impact_coords)
        # ix, iy = impact_coords
        # tx, ty = p2_pos_before
        # distance = math.sqrt((tx - ix)**2 + (ty - iy)**2)
        # print(f"Impact at {impact_coords}, Target at {p2_pos_before}. Distance: {distance:.2f}")
        reward = 0
        if impact_coords and p2_pos_before:
            ix, iy = impact_coords
            tx, ty = p2_pos_before
            distance = math.sqrt((tx - ix)**2 + (ty - iy)**2)
            print(f"Impact at {impact_coords}, Target at {p2_pos_before}. Distance: {distance:.2f}")

            # Reward is inversely proportional to distance. Max reward for a close hit.
            # You can tune these values.
            if distance < 50: # A good hit
                reward = 1000 - (distance * 10)
            else: # A miss, penalty increases with distance
                reward = -distance

        # Wait for opponent's turn to complete
        time.sleep(5)
        pt.fire()
        time.sleep(5)
        # --- Get new state and calculate reward ---
 




        done = False # In this setup, an episode could be a full game, but we run it step-by-step.
        
        return self.get_state(), reward, done

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        print("Resetting environment...")
        return self.get_state()

# --- Agent Definition ---

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def _discretize_state(self, state):
        """Converts a continuous state into a discrete one for the Q-table."""
        angle, power, _, _, p1_pos, p2_pos = state
        
        if angle is None or power is None or p1_pos is None or p2_pos is None:
            return None

        angle_bucket = int(angle / 10)
        power_bucket = int(power / 10)
        
        dx = p2_pos[0] - p1_pos[0]
        dy = p2_pos[1] - p1_pos[1]
        
        dx_bucket = int(dx / 50)
        dy_bucket = int(dy / 50)

        return (angle_bucket, power_bucket, dx_bucket, dy_bucket)

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        discrete_state = self._discretize_state(state)
        if discrete_state is None:
            return np.random.choice(self.action_space)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        
        q_values = self.q_table.get(discrete_state, {a: 0 for a in self.action_space})
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        """Updates the Q-table using the Bellman equation."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        if discrete_state is None or discrete_next_state is None:
            return

        old_value = self.q_table.get(discrete_state, {}).get(action, 0)
        
        next_max = 0
        if discrete_next_state in self.q_table:
            next_max = max(self.q_table[discrete_next_state].values())

        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = {a: 0 for a in self.action_space}
        self.q_table[discrete_state][action] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")

# --- Main Training Loop ---

def main():
    env = PocketTanksEnv()
    agent = QLearningAgent(action_space=env.action_space)
    agent.load_q_table()

    num_episodes = 1000
    # Each step is a full turn, so we need fewer steps per episode.
    max_steps_per_episode = 20 

    for episode in range(num_episodes):
        state = env.reset()
        
        for turn in range(max_steps_per_episode):
            # It's player 1's turn, choose and execute an action
            action = agent.choose_action(state)
            
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            
            print(f"Episode: {episode + 1}/{num_episodes}, Turn: {turn + 1}, Reward: {reward}")

            if done:
                break
        
        if (episode + 1) % 10 == 0:
            agent.save_q_table()

    print("Training finished.")

if __name__ == "__main__":
    main()