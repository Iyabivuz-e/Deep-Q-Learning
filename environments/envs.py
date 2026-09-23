import gymnasium as gym

class FrozenLakeEnv:
      
    def load_frozenlake_env(self, render: bool):  
        return gym.make(
            "FrozenLake-v1",
            render_mode="human" if render else None,
            map_name = "4x4",
            desc=None,
            is_slippery=False,
            reward_schedule=(1, 0, 0) # 1 when done, 0 when hit the hall, 0 when truncated
        )
