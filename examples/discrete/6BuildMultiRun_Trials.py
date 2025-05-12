# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:22:43 2022

@author: Anonymous
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import scripts.GMEnv as sim
import scripts.Tester as test
import unittest


class TestSum(unittest.TestCase):
    simRandom = 0 
    simRandomForgive = 0 
    simCustom = 0 
    simOptimal = 0 
    
    simRandomIter = 100
    simCustomIter = 100
    simOptimalIter = 100
    testingIter = 1000
    
   
    trainingIter = 1000
    learningAlgorithm = "A2C"
    learningLoggingInterval = 2
    
    learningOptimal = 0
    learningParams = 0
    dtGologOptimal = 0.928199999999999
    
    def setUp(self):
        self.env = sim.GMEnv("./examples/discrete/6BuildMultiRun.pl")
        self.env.setDebug(False)
        self.env.setSeed(123)
        self.t = test.TestIt(self.env)
        self.t.debug = False
        self.inFeasiblePenalty = self.env.getInfeasiblePenalty()


    def tearDown(self):
        self.env.closeQE()
        
    def test_semiRandonSim(self):
        """
        A simulation based on a crude policy that assumes certain success of actions.

        Returns
        -------
        None.

        """
        result = self.t.simulate(self.simCustomIter,[0,2,0,2])
        TestSum.simCustom = result
            
    def test_randonSimForgive(self):
        """
        A random policy (actions are picked randomly)
 
        Returns
        -------
        None.
 
        """
        result = self.t.simulate(self.simRandomIter)
        TestSum.simRandom = result
    
    def test_randonSim(self):
        """
        A random policy (actions are picked randomly)
 
        Returns
        -------
        None.
 
        """
        result = self.t.simulate(self.simRandomIter,forgivePenalty=False)
        TestSum.simRandomForgive = result
        
    
    def test_learning(self):
        """
        Training and testing an RL agent

        Returns
        -------
        None.

        """
        result, params = self.t.test_learning(self.trainingIter,self.testingIter, logging = self.learningLoggingInterval, algo = self.learningAlgorithm)
        TestSum.learningOptimal = result
        TestSum.learningParams = params
        self.assertAlmostEqual(result,
                          self.dtGologOptimal, 
                          places = 0,
                          msg = "\n Learning failed: {} expected, {} observed".format(TestSum.dtGologOptimal,result))
            

    @classmethod
    def tearDownClass(cls):
        print('DT-Golog - calculated policy reward: {}'.format(cls.dtGologOptimal))
        #print('DT-Golog - simulated policy reward : {}'.format(cls.simOptimal))
        print('Partially optimal policy reward....: {}'.format(cls.simCustom))
        print('Random policy reward.............. : {}'.format(cls.simRandom))
        print('Random policy reward.(fg)......... : {}'.format(cls.simRandomForgive))
        print('Learned policy reward..............: {}'.format(cls.learningOptimal))
        print('--> Learning Parameters: \n {}'.format(cls.learningParams))


if __name__ == '__main__':
    unittest.main()



    # def test_simulation(self):
    #     result = self.t.simulate(1000,[0,2,0,2])
    #     TestSum.simOptimal = result
    #     self.assertAlmostEqual(result,
    #               TestSum.dtGologOptimal, 
    #               places = 1,
    #               msg = "\n Wrong simulaton result status: {} expected, {} observed".format(TestSum.dtGologOptimal,result))
  
    # def test_randomSim(self):
    #     result = self.t.simulate(1000)
    #     TestSum.simRandom = result

    # def test_learning(self):
    #     result, params = self.t.test_learning(5000,1000,logging = 100,algo = "A2C")
    #     TestSum.learningOptimal = result
    #     TestSum.learningParams = params
    #     self.assertAlmostEqual(result,
    #                       self.dtGologOptimal, 
    #                       places = 1,
    #                       msg = "\n Learning failed: {} expected, {} observed".format(TestSum.dtGologOptimal,result))
        




# from stable_baselines3 import A2C
# from stable_baselines3 import DQN

# env = sim.GMEnv("./Examples2/6BuildMultiRun.pl")
# model = A2C("MlpPolicy", env, verbose=1)
# model2 = DQN("MlpPolicy", env, verbose=1)
# params = model.get_parameters().get("policy.optimizer").get("param_groups")
# model.learn(total_timesteps=50,log_interval=10)

# model2.learn(total_timesteps=1000,log_interval=100)


    # @classmethod
    # def tearDownClass(cls):
    #     print('DT-Golog - calculated policy reward: {}'.format(cls.dtGologOptimal))
    #     print('DT-Golog - simulated policy reward.: {}'.format(cls.simOptimal))
    #     print('Random simulated policy reward.....: {}'.format(cls.simRandom))
    #     print('Learned policy reward..............: {}'.format(cls.learningOptimal))
    #     print('--> Learning Parameters: \n {}'.format(cls.learningParams))

    
    
    
    
    
    
    # def test_learning(self):
        #env = sim.GMEnv("./Examples/6BuildMultiRun.pl")
        #model = A2C("MlpPolicy", env, verbose=1)
    #     model = A2C("MlpPolicy", self.env, verbose=1)
    #     params = model.get_parameters().get("policy.optimizer").get("param_groups")
    #     model.learn(total_timesteps=1_000)

    #     vec_env = model.get_env()
    #     obs = vec_env.reset()
    #     totalReward = 0
    #     totalIter = 10
    #     for i in range(totalIter):
    #         obs = vec_env.reset()
    #         episodeDone = False
    #         episodeReward = 0
    #         print("Iteration:{}".format(i))
    #         while (not(episodeDone)):
    #             action, _state = model.predict(obs, deterministic=True)
    #             obs, reward, done, info = vec_env.step(action)
    #             episodeReward = episodeReward + reward[0]
    #             episodeDone = done[0]
    #         totalReward  = totalReward + episodeReward 
    #     result = totalReward/totalIter
    #     TestSum.learningOptimal = result
    #     TestSum.learningParams = params
        
    #     self.assertAlmostEqual(result,
    #                       TestSum.dtGologOptimal, 
    #                       places = 1,
    #                       msg = "\n Learning failed: {} expected, {} observed".format(TestSum.dtGologOptimal,result))
 





# import scripts.simulator as sim
# import scripts.tester as test
# import unittest

# env = sim.GoalModel("./Examples/6BuildMultiRun.pl")
# env.setDebug(True)
# env.setSeed(123)
# t = test.TestIt(env)
# t.debug = False
# t.reset()
# env.setSeed(321)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(3)

# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(3)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(3)
# env.reset()
# env.setSeed(15)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(3)
# a,b,c,d,e = t.performAction(3)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(2)
# a,b,c,d,e = t.performAction(3)
