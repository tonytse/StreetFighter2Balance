import retro
import numpy as np
import cv2

import gym
from gym import Env
from gym.spaces import MultiDiscrete, Discrete, MultiBinary, Box


from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

kCharacters = [ 'Ryu', 'Ken', 'EHonda', 'ChunLi', 'Blanka', 'Zangief', 'Guile', 'Dhalsim', 'Balrog', 'Vega',  'Sagat' , 'MBison']

# Mapping modal action_space to Sega Genesis action space
# Modal: 0: B, 1: A, 2: Up, 3: Down, 4: Left, 5: Right, 6: C, 7: Y, 8: X, 9:Z
#  Sega: 0: B, 1: A, 2: Mode, 3: Start, 4: Up, 5: Down, 6: Left, 7: Right, 8: C, 9: Y, 10: X, 12:Z
# Remove index 2,3
def StreetFighter_PocessAction( a ):
    actions =  np.zeros(12)
        
    actions[0]  = a[0]
    actions[1]  = a[1]
    actions[2]  = 0
    actions[3]  = 0
    actions[4]  = a[2]
    actions[5]  = a[3]
    actions[6]  = a[4]
    actions[7]  = a[5]
    actions[8]  = a[6]
    actions[9]  = a[7]
    actions[10] = a[8]
    actions[11] = a[9]

    return actions

#class for StreetFighter environment
class StreetFighter(Env):
    def __init__(self, state, no_of_player=1):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(100,80,1), dtype=np.uint8)
        self.previous_frame = np.zeros(self.observation_space.shape)
        self.action_space = MultiBinary(10)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, state=state, players=no_of_player)
        self.no_of_player = no_of_player
                    
    def step(self, action):
        #print( action )
        actions = None
        if self.no_of_player == 1:
            actions = StreetFighter_PocessAction(action)
        else:
            actions = action
            
        #print( actions )
        obs, reward, done, info = self.game.step(actions)
        obs = self.preprocess(obs)

        frame_delta = obs - self.previous_frame
        self.previous_frame = obs 
        
        if self.no_of_player == 1:
            reward = 0
            
            if done:

                if info['matches_won'] == 1:
                    reward += 100 + info['health'] * 3
                else: # info['enemy_matches_won'] == 1:
                    reward -= 100
                
            else:

                if self.combo_count > 0:
                    self.combo_count -= 0.02

                if self.enemy_combo_count > 0:
                    self.enemy_combo_count -= 0.02      

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
        
        
def CreateEnv(state, no_of_player=1, log_folder=None, skip=1):
    env = StreetFighter(state,no_of_player)
    if skip > 1:
        env = MaxAndSkipEnv(env, skip)
    if log_folder is not None:
        env = Monitor(env, log_folder)
        
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env



def PeekGame( state, modelA = None, modelB = None, skip=5):

    no_of_player = 1
    actions = None
    
    if modelB is not None: # AI vs AI
        no_of_player = 2
        actions = np.zeros((1,24))
    elif modelA is not None: #AI vs BuildIn AI
        actions = np.zeros((1,12))
    
    env = CreateEnv( state, no_of_player=no_of_player, log_folder = None, skip=skip )
    obs = env.reset()
    
    while True:

        if modelA is None: # random mode
            actions = [env.action_space.sample(),[]]
        else:
            actionA, _ = modelA.predict(obs)
            for i, a in enumerate(actionA[0]):
                actions[0][i] = a

            if modelB is not None:
                actionB, _ = modelB.predict(obs)
                for i, a in enumerate(actionB[0]):
                    actions[0][i+12] = a

        obs, rewards, done, info = env.step(actions)    

        env.render()
        if done == True:
            obs = env.close()
            break