# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:22:43 2022

@author: Anonymous
"""



import unittest

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import scripts.GMEnv as sim
import scripts.Tester as test


class TestSum(unittest.TestCase):
    
    def setUp(self):
        self.env = sim.GMEnv("./examples/discrete/1Order.pl")
        self.env.setDebug(False)
        self.env.setSeed(123)
        self.t = test.TestIt(self.env)
        self.t.debug = False

    def tearDown(self):
        self.env.closeQE()

    def takeStep(self,action,
                   stActionExp,
                   stateExp,
                   ALExp,
                   SLExp,
                   runExp,
                   rewardExp,
                   cumRewardExp,
                   doneExp,
                   achievedExp,
                   transState,
                   ID = -1,choice = -1):
    
        stateObs, rewardObs, cumRewardObs, doneObs, info = self.t.performAction(action,choice)
        
        self.assertEqual(stActionExp,
                         info["stAction"], 
                         msg = "\n (TestID: {}) - Wrong stochastic action: {} expected, {} observed".format(ID,stActionExp,info["stAction"]))
        self.assertEqual(stateExp,
                         stateObs, 
                         msg = "\n (TestID: {}) - Wrong state: {} expected, {} observed".format(ID,stateExp,stateObs))
        self.assertEqual(ALExp,
                         info["tH"], 
                         msg = "\n (TestID: {}) - Wrong AL: {} expected, {} observed".format(ID,ALExp,info["tH"]))
        self.assertEqual(SLExp,
                         info["eH"], 
                         msg = "\n (TestID: {}) - Wrong SL: {} expected, {} observed".format(ID,SLExp,info["eH"]))
        self.assertEqual(runExp,
                         info["Run"], 
                         msg = "\n (TestID: {}) - Wrong run: {} expected, {} observed".format(ID,runExp,info["Run"]))
        self.assertEqual(rewardExp,
                         rewardObs, 
                         msg = "\n (TestID: {}) - Wrong reward: {} expected, {} observed".format(ID,rewardExp,rewardObs))
        self.assertEqual(cumRewardExp,
                         cumRewardObs, 
                         msg = "\n (TestID: {}) - Wrong cummuative reward: {} expected, {} observed".format(ID,cumRewardExp,cumRewardObs))
        self.assertEqual(doneExp,
                         doneObs, 
                         msg = "\n (TestID: {}) - Wrong 'done' status: {} expected, {} observed".format(ID,doneExp,doneObs))
        self.assertEqual(achievedExp,
                         info["Achieved"], 
                         msg = "\n (TestID: {}) - Wrong 'achieved' status: {} expected, {} observed".format(ID,achievedExp,info["Achieved"]))
        self.assertEqual(transState,
                         info["TransState"], 
                         msg = "\n (TestID: {}) - Wrong 'TransState' status: {} expected, {} observed".format(ID,transState,info["TransState"]))
    
    def test_debug(self):
        
        self.takeStep(action = 1,
            stActionExp = 4, stateExp = 2, 
            ALExp = [[1],[]], SLExp = [[4],[]], runExp = 1, 
            rewardExp = 1.0*0.7 + 0.7*0.3, cumRewardExp = 1.0*0.7 + 0.7*0.3,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 1)

        self.takeStep(action = 1,
            stActionExp = -1, stateExp = 2, 
            ALExp = [[1],[]], SLExp = [[4],[]], runExp = 1, 
            rewardExp = self.env.inFeasiblePenalty, cumRewardExp = 1.0*0.7 + 0.7*0.3 + self.env.inFeasiblePenalty,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 2)

        self.takeStep(action = 0,
            stActionExp = -1, stateExp = 2, 
            ALExp = [[1],[]], SLExp = [[4],[]], runExp = 1, 
            rewardExp = self.env.inFeasiblePenalty, cumRewardExp = 1.0*0.7 + 0.7*0.3 + 2*self.env.inFeasiblePenalty,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 3)

        self.t.reset()
 
        self.takeStep(action = 0,
            stActionExp = 0, stateExp = 32, 
            ALExp = [[0],[]], SLExp = [[0],[]], runExp = 1, 
            rewardExp = 0.5*0.7 + 1.0*0.3, cumRewardExp = 0.5*0.7 + 1.0*0.3,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 4)   

        self.takeStep(action = 0,
            stActionExp = -1, stateExp = 32, 
            ALExp = [[0],[]], SLExp = [[0],[]], runExp = 1, 
            rewardExp = self.env.inFeasiblePenalty, cumRewardExp = 0.5*0.7 + 1.0*0.3 + self.env.inFeasiblePenalty,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 5) 

        self.t.reset()
        self.env.setSeed(9510)

        self.takeStep(action = 1,
            stActionExp = 5, stateExp = 1, 
            ALExp = [[1]], SLExp = [[5]], runExp = 0, 
            rewardExp = 1.0*0.7, cumRewardExp = 1.0*0.7,
            doneExp = True,  achievedExp = False,
            transState = "[]",
            ID = 6) 
    
        self.takeStep(action = 1,
            stActionExp = -1, stateExp = 1, 
            ALExp = [[1]], SLExp = [[5]], runExp = 0, 
            rewardExp = self.env.inFeasiblePenalty, cumRewardExp = 1.0*0.7 + self.env.inFeasiblePenalty,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 7)  

        self.takeStep(action = 0,
            stActionExp = -1, stateExp = 1, 
            ALExp = [[1]], SLExp = [[5]], runExp = 0,
            rewardExp = self.env.inFeasiblePenalty, cumRewardExp = 1.0*0.7 + 2*self.env.inFeasiblePenalty,
            doneExp = True, achievedExp = False,
            transState = "[]",
            ID = 8)
    
    
    # def test_random(self):

    #     # Test 1
    #     self.takeStep(action = 1,
    #         stActionExp = 3, stateExp = 4, 
    #         ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #         rewardExp = 1, cumRewardExp = 1,
    #         doneExp = True, achievedExp = True,
    #         ID = 1,choice = 3)
        
    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 4, 
    #         ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 1,
    #         doneExp = True, achievedExp = True,
    #         ID = 2,choice = 1)
        
    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 4, 
    #         ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 1,
    #         doneExp = True, achievedExp = True,
    #         ID = 3,choice = 4)
        
    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 4, 
    #         ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 1,
    #         doneExp = True, achievedExp = True,
    #         ID = 4,choice = 2)
        
    #     # Test 2
    #     self.t.reset()
    #     self.takeStep(action = 0,
    #         stActionExp = 2, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0.5*0.7, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 5,choice = 2)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 6,choice = 5)

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 7,choice = 0)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 8,choice = 2)


    #     # Test 3
    #     self.t.reset()
        
    #     self.takeStep(action = 0,
    #         stActionExp = 2, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0.5*0.7, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 9,choice = 2)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 10,choice = 2)
        
    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 11,choice = 2)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 8, 
    #         ALExp = [[0]], SLExp = [[2]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 12,choice = 1)

    #     # Test 4
    #     self.t.reset()
        
    #     self.takeStep(action = 0,
    #         stActionExp = 0, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0.5*0.7 + 0.3, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 13,choice = 0)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 14,choice = 4)

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 15,choice = 2)

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 16,choice = 1)


    #     # Test 5
    #     self.t.reset()
        
    #     self.takeStep(action = 0,
    #         stActionExp = 1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0.5*0.7 + 0.3*0.7, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = 13,choice = 1)

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = 14,choice = 1)

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = 15,choice = 2)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = 16,choice = 4)
    
    #     # Test 6
    #     self.t.reset()
        
    #     self.takeStep(action = 1,
    #         stActionExp = 5, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0.7, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 17,choice = 5)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 18,choice = 5)    

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 19,choice = 2)

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = 20,choice = 4)


    #     # Test 7
    #     self.t.reset()
        
    #     self.takeStep(action = 0,
    #         stActionExp = 0, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0.5*0.7 + 0.3, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 21,choice = 0)
        
    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 22,choice = 5)
        
    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 23,choice = 1)
        
    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 32, 
    #         ALExp = [[0]], SLExp = [[0]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3,
    #         doneExp = True, achievedExp = True,
    #         ID = 24,choice = 1)


    #     # Test 8
    #     self.t.reset()
       
    #     self.takeStep(action = 1,
    #        stActionExp = 4, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0.7 + 0.7*0.3, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = 25,choice = 4)

    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = 26,choice = 2)

    #     self.takeStep(action = 1,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = 27,choice = 5)

    #     i = 28

    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 2)
    #     i+=1
    
    
    #     # Test 9
    #     self.t.reset()
       
    #     self.takeStep(action = 1,
    #        stActionExp = 4, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0.7 + 0.7*0.3, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 4)
    #     i+=1

    #     self.takeStep(action = 1,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 4)
    #     i+=1

    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 1)
    #     i+=1

    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 2, 
    #        ALExp = [[1]], SLExp = [[4]], runExp = 0,
    #        rewardExp = 0, cumRewardExp = 0.7 + 0.7*0.3,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 0)
    #     i+=1

    
    #     # Test 10
    #     self.t.reset()
        
    #     self.takeStep(action = 1,
    #         stActionExp = 5, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0.7, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = i,choice = 5)
    #     i+=1

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = i,choice = 4)
    #     i+=1

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = i,choice = 3)
    #     i+=1

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 1, 
    #         ALExp = [[1]], SLExp = [[5]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.7,
    #         doneExp = True, achievedExp = False,
    #         ID = i,choice = 3)
        
    #     i = 38

    #     # Test 11
    #     self.t.reset()
        
    #     self.takeStep(action = 0,
    #         stActionExp = 1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0.5*0.7 + 0.3*0.7, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = i,choice = 1)
    #     i += i

    #     self.takeStep(action = 0,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = i,choice = 2)
    #     i += i

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = i,choice = 3)
    #     i += i

    #     self.takeStep(action = 1,
    #         stActionExp = -1, stateExp = 16, 
    #         ALExp = [[0]], SLExp = [[1]], runExp = 0, 
    #         rewardExp = 0, cumRewardExp = 0.5*0.7 + 0.3*0.7,
    #         doneExp = True, achievedExp = True,
    #         ID = i,choice = 3)
    #     i += i        

    #     i = 42
    
    #     # Test 12
    #     self.t.reset()
        
    #     self.takeStep(action = 1,
    #        stActionExp = 3, stateExp = 4, 
    #        ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #        rewardExp = 1, cumRewardExp = 1,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 3)
    #     i+=1
        
    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 4, 
    #        ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #        rewardExp = 0, cumRewardExp = 1,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 0)
    #     i+=1

    #     self.takeStep(action = 1,
    #        stActionExp = -1, stateExp = 4, 
    #        ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #        rewardExp = 0, cumRewardExp = 1,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 4)
    #     i+=1
        
    #     self.takeStep(action = 0,
    #        stActionExp = -1, stateExp = 4, 
    #        ALExp = [[1]], SLExp = [[3]], runExp = 0, 
    #        rewardExp = 0, cumRewardExp = 1,
    #        doneExp = True, achievedExp = True,
    #        ID = i,choice = 0)
    #     i+=1



if __name__ == '__main__':
    unittest.main()







#env = sim.GoalModel("./Examples/1Order.pl")
# env.setDebug(True)
# env.setDebug(False)
# env.setSeed(123)


# import scripts.simulator as sim
# env = sim.GoalModel("./Examples/1Order.pl")
# import scripts.tester as test
# t = test.TestIt(env)
# t.reset()
# t.debug = False
# t.simulate(100,[1])



# total_r = 0
# for i in range(1,1001):
#     env.reset()    
#     _, r, _, _ = env.step(1)
#     total_r += r

# print(total_r/1001)    

# t = test.TestIt(env)
# t.debug = False
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)
# t.reset()
# a,b,c,d,e = t.performAction(0)
# a,b,c,d,e = t.performAction(0)
# t.reset()
# env.setSeed(9510)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(1)
# a,b,c,d,e = t.performAction(0)


# env = sim.GoalModel("./Examples/1Order.pl")
# env.setDebug(False)
# env.setSeed(123)
# t = test.TestIt(env)
# t.debug = False
# t.simulate(100)

