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
    learningLoggingInterval = 500
    
    learningOptimal = 0
    learningParams = 0
    dtGologOptimal = 0.9235
    
    
    def setUp(self):
        self.env = sim.GMEnv("./examples/discrete/1Order.pl")
        self.env.setDebug(False)
        self.env.setSeed(123)
        self.t = test.TestIt(self.env)
        self.t.debug = False

    def tearDown(self):
        self.env.closeQE()
       
    def test_semiRandonSim(self):
        """
        A simulation based on a crude policy that assumes certain success of actions.
 
        Returns
        -------
        None.
 
        """
        result = self.t.simulate(self.simCustomIter,[1])
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


