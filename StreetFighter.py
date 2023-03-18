import retro
import numpy as np
import cv2

import gym
from gym import Env
from gym.spaces import MultiDiscrete, Discrete, MultiBinary, Box


from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

#define global variable

# All characters name
kCharacters = [ 'Ryu', 'Ken', 'EHonda', 'ChunLi', 'Blanka', 'Zangief', 'Guile', 'Dhalsim', 'Balrog', 'Vega',  'Sagat' , 'MBison']

# define action space
kAction=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Neutral
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Right
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # Left
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Crouch
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # Crouch_Right
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # Crouch_Left

    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # Jump_Neutral
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # Jump_Right
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], # Jump_Left

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # Standing_Low_Punch
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # Standing_Medium_Punch
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # Standing_High_Punch

    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Standing_Low_Kick
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Standing_Medium_Kick
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # Standing_High_Kick 

    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], # Crouching_Low_Punch
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # Crouching_Medium_Punch
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], # Crouching_High_Punch

    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Crouching_Low_Kick
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Crouching_Medium_Kick
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Crouching_High_Kick 

    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], # Right_Throw
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # Left_Throw
]

#class for StreetFighter environment
class StreetFighter(Env):
    def __init__(self, state, no_of_player=1):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(100,80,1), dtype=np.uint8)
        self.previous_frame = np.zeros(self.observation_space.shape)
        self.action_space = Discrete(len(kAction))
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, state=state, players=no_of_player)
        self.no_of_player = no_of_player
                    
    def step(self, action):
                
        actions = None
        if self.no_of_player == 1:
            actions = kAction[action]
        else:
            actions = np.zeros(24)
            
            for i in range(len(kAction[0])):
                actions[i] = kAction[action[0]][i]
                actions[i+12] = kAction[action[1]][i]
        
        obs, reward, done, info = self.game.step(actions)
        obs = self.preprocess(obs)

        frame_delta = obs - self.previous_frame
        self.previous_frame = obs 
        
        # calc reward only when number of player is 1.
        if self.no_of_player == 1:
            reward = 0
            
            if done:

                if info['matches_won'] == 1:
                    reward += 100 + info['health'] * 3
                else: # info['enemy_matches_won'] == 1:
                    reward -= 100
                
            else:
                
                # calc combo count 
                if self.combo_count > 0:
                    self.combo_count -= 0.02

                if self.enemy_combo_count > 0:
                    self.enemy_combo_count -= 0.02      

                # calc damage of the frame
                enemy_damage = self.enemy_health - info['enemy_health']
                damage = self.health - info['health']

                if enemy_damage > 0 and damage > 0: 
                    # double hit
                    reward += enemy_damage - damage
                elif enemy_damage > 0:
                    reward += enemy_damage + self.combo_count * 8
                    self.combo_count += 1
                    self.enemy_combo_count = 0
                elif damage > 0: 
                    reward -= damage + self.enemy_combo_count * 8
                    self.enemy_combo_count += 1
                    self.combo_count = 0

            self.health = info['health']
            self.enemy_health = info['enemy_health']
        
        return frame_delta, reward, done, info

    def render(self, *args, **kwargs): 
        self.game.render()

    def reset(self):

        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs

        # Initial variables
        self.health = 176
        self.enemy_health = 176
        self.combo_count = 0
        self.enemy_combo_count = 0
        return obs

    def preprocess(self, observation): 
        # process the input image
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (100,80), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,80, 1))
        return state

    def close(self): 
        self.game.close()
        
# Remove reward for 2 player , to avoid error in DummyVecEnv when game return a reward array 
class RemoveReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reward(self, reward):
        return 0
    
def CreateEnv(state, no_of_player=1, log_folder=None, skip=1):
    env = StreetFighter(state,no_of_player)
    
    # Apply skip frame
    if skip > 1:
        env = MaxAndSkipEnv(env, skip)   
    # Apply monitor log
    if log_folder is not None:
        env = Monitor(env, log_folder)
    # Remove reward calc for more than 1 player to avoid error from gym
    if no_of_player > 1:
        env = RemoveReward(env)        

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env


# A debug function to check the model
def PeekGame( state, modelA = None, modelB = None, skip=5):

    no_of_player = 1
    
    if modelB is not None: # AI vs AI
        no_of_player = 2
    
    env = CreateEnv( state, no_of_player=no_of_player, log_folder = None, skip=skip )
    obs = env.reset()
    
    while True:

        actions = []
        
        if modelA is None: # random mode
            actions.append(env.action_space.sample())
        else:
            actionA, _ = modelA.predict(obs)
            
            
            if modelB is None:
                actions.append(actionA[0])
            else:
                actionB, _ = modelB.predict(obs)
                actions.append([actionA[0],actionB[0]])

        obs, rewards, done, info = env.step(actions)    

        env.render()
        if done == True:
            obs = env.close()
            break