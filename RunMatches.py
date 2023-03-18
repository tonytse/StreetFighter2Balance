#!/usr/bin/env python
# coding: utf-8

import StreetFighter as sf


import numpy as np
import math
import csv
import json
from sb3_contrib import TRPO


kRound = 100

def PerforMmatch( a, b ) :
    state = f'VS_{a}_{b}'
    #print( state )
    modelA = TRPO.load(f'model/{a}.zip')
    modelB = TRPO.load(f'model/{b}.zip')
        
    env = sf.CreateEnv( state, no_of_player=2, log_folder = None, skip=1 )
    obs = env.reset()
    
    winA = 0
    winB = 0
        
    for i in range(kRound):

        healthA = 176
        healthB = 176
        randomAction = 0
        sameHealhCount = 0
        
        while True:

            actions = []

            if sameHealhCount > 120:
                randomAction = 5
                sameHealhCount = 0               

            if randomAction > 0 :
                actionA = [env.action_space.sample()]
                actionB = [env.action_space.sample()]
                randomAction -= 1

            else:
                actionA, _ = modelA.predict(obs)
                actionB, _ = modelB.predict(obs)

            actions.append([actionA[0],actionB[0]])

            obs, rewards, done, info = env.step(actions)
            #env.render()
                
            currhealthA = info[0]['health']
            currhealthB = info[0]['enemy_health']
        
            if (healthA == currhealthA) and (healthB == currhealthB) :
                sameHealhCount += 1
            
            healthA = currhealthA
            healthB = currhealthB
            
            if done == True:
                obs = env.reset()
                break
                
        if info[0]['matches_won'] == 1:
            winA += 1
        elif info[0]['2p_matches_won'] == 1:
            winB += 1
        else:
            if healthA > healthB:
                winA += 1                           
            elif healthB > healthA:
                winB += 1
            else:
                winA += 0.5
                winB += 0.5
            
    return winA / kRound


n_char = len(sf.kCharacters)
match_tabel = np.zeros((n_char,n_char))


for i in range(n_char):
    for j in range(n_char):
        if j < i :
            continue
        if i == j:
            match_tabel[i][j] = -1
        else: 
            v = PerforMmatch( sf.kCharacters[i], sf.kCharacters[j] ) * 10 
            print( f'{sf.kCharacters[i]} vs {sf.kCharacters[j]} > {v}')
            match_tabel[i][j] = v 
            match_tabel[j][i] = 10-v


with open('Matchup_web/data.txt', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow('#-')
    
    o = [' ']
    o.extend(sf.kCharacters)
    tsv_writer.writerow(o)
        
    for i in range(n_char):
        l = match_tabel[i].tolist()
        o = [ sf.kCharacters[i]]
        for i in l:
            if i == -1 :
                o.append('-')
            else:
                if i % 1 == 0:
                    o.append( math.floor(i) )
                else:
                    o.append( round(i,1) )
        tsv_writer.writerow(o)
    
