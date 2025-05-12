# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:16:39 2022

@author: Anonymous User
"""



from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import sys
import time

class TestIt():
    def __init__(self,environment):
        self.env = environment
        self.score = 0
        self.debug = False

    def setDebug(self,status):
        self.debug = status

    def setEnv(self,environment):
        self.env = environment
        
    def reset(self):
        self.env.reset()
        self.score = 0

    def close(self):
        self.env.closeQE()

    def performAction(self,action,choice = -1):
        n_state, reward, terminated, truncated, info = self.env.step(action, choice)
        self.score += reward
        
        if self.debug:
            print(' ')
            print('New Action Attempt:')
            print('--> Action: {}'.format(action))
            print('--> St. Action: {}'.format(info["stAction"]))
            print('--> State: {}'.format(n_state))
            print('--> Reward: {}'.format(reward))
            print('--> Cum. Reward: {}'.format(self.score))
        
        return n_state,reward,self.score, terminated or truncated,info

        
    def simulate(self,episodes,policy = [],debug = False, forgivePenalty = True):
        totalScore = 0
        if policy:  
            print("Starting simulations on extraneously defined policy:")
        else:
            print("Starting random simulations:")
        for episode in range(1,episodes + 1):
            sys.stdout.write("\r\t%d%%" % ((episode/episodes)*100))
            sys.stdout.flush()
            self.env.reset()
            done = False
            self.score = 0
            pol = policy.copy()
            
            while not done:
                self.env.render()
                if not policy: 
                    # Executing Random Policy
                    #print("Random")
                    #print(policy)
                    action = self.env.action_space.sample()
                elif pol:
                    # Executing given policy
                    action = pol.pop(0)
                else: 
                    print("Error: Failed to end deterministic policy")
                    print("Requested policy: {}",format(policy))
                    print("Actions left policy: {}",format(pol))
                    print("Episode done: {}",format(done))
                    
                    # Executing given policy
                    
                    
                n_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if (reward != self.env.getInfeasiblePenalty()) or (not forgivePenalty):
                    self.score += reward
                
                if self.debug: 
                    print(' ')
                    print('New Action Attempt:')
                    print('--> Action: {}'.format(action))
                    print('--> St. Action: {}'.format(info))
                    print('--> State: {}'.format(n_state))
                    print('--> Reward: {}'.format(reward))
                    print('--> Cum. Reward: {}'.format(self.score))
    
                
                #Check if you reached a deadlock
                #print("Checking for deadlock:...")
                #canExit = False
                #for a in range(0,self.env.action_space.n - 1) :
                #    canExit = canExit or self.env.possible(a) 
                
                #if (not canExit and not done):
                #    print("**** Unspotted Deadlock! ***")
                    #sl, sa, pos, bitst = self.env.debug()
                    #print("**** ** SL: {}".format(sl))
                    #print("**** ** AL: {}".format(sa))
                    #print("**** ** Poss: {}".format(pos))
                    #print("**** ** State: {}".format(bitst))
                    #print("**** ** Last action:{} ****".format(action))
                #    break
                
                if (policy and (not pol) and (not done)):
                    print("**** Error: Failed to end deterministic policy ***")
            
            totalScore += self.score
            if debug:
                print('Episode:{} Score:{}'.format(episode,self.score))
        #print('Simulation: average reward: {}'.format(totalScore/episodes))
        print("")
        print("Simulations complete.")
        return(totalScore/episodes)


    def test_learning(self, learn_iter = 10_000, test_iter = 10000,logging= 1000, algo = "A2C"):
        
        st = time.process_time()
        print("Attempting {} model construction.".format(algo))
        
        if (algo == "A2C"):
            self.envm = Monitor(self.env,info_keywords=("is_success",))
            model = A2C("MlpPolicy", self.envm, verbose=1)
            print("Model Constructed. Learning starts...")
            model.learn(total_timesteps=learn_iter,log_interval = logging)
        elif (algo == "DQN"):
            self.envm = Monitor(self.env,info_keywords=("is_success",))
            model = DQN("MlpPolicy", self.envm, verbose=1)
            print("Model Constructed. Learning starts...")
            model.learn(total_timesteps=learn_iter,log_interval = logging)
        elif (algo == "PPO"):
            self.envm = Monitor(self.env,info_keywords=("is_success",))
            model = PPO("MlpPolicy", self.envm, verbose=1)
            print("Model Constructed. Learning starts...")
            model.learn(total_timesteps=learn_iter,log_interval = logging)
        else:
            print("Uknown learning algorithm")
            return
        vec_env = model.get_env()
        obs = vec_env.reset()
        params = model.get_parameters().get("policy.optimizer").get("param_groups")
        
        totalReward = 0
        totalIter = test_iter
        
        # Time
        et = time.process_time()
        res = et - st
        print('Learning Complete. Leargning CPU Execution time:', res, 'seconds')
        print("Starting testing..")
        for i in range(totalIter):
            sys.stdout.write("\r\t%d%%" % ((i/totalIter)*100))
            sys.stdout.flush()
            obs = vec_env.reset()
            episodeDone = False
            episodeReward = 0 
            while (not(episodeDone)):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episodeReward = episodeReward + reward[0]
                episodeDone = done[0]
            totalReward  = totalReward + episodeReward 
        print("")
        print("Learning simulations complete.")
        result = totalReward/totalIter
        
        
        return result, params
