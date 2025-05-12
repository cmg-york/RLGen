# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:39:07 2023

@author: Anonymous
"""

from pyswip import Prolog
from .QMI import QMI

class QueryEngine(QMI):
    
    
    def __init__(self,file):
        self.prolog = Prolog()
        # First load the interface file
        self.prolog.consult("./scripts/QE/DT-Golog-Iface.pl")
        # Then load the domain file
        self.prolog.consult(file)
        
    def setFile(self,file):
        """
        [Refer to QMI function documentation.]
        """
        # When setting a new file, we need to reload both files
        self.prolog.consult("./scripts/QE/DT-Golog-Iface.pl")
        self.prolog.consult(file)
        
    def possibleAt(self,t, eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "possibleAt([" + eH + "], " + str(t) + ")."
        if (list(self.prolog.query(query))):
            result = True
        else:
            result = False
        return result
        
    def getOutcomes(self,t,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getActionOutcomes(" + str(t) + ",[" + eH + "],SActs,Probs)."
        possStochActions = list(self.prolog.query(query))[0]['SActs']
        probs = list(self.prolog.query(query))[0]['Probs']
        return possStochActions, probs
    
    def getProbs(self,t,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getActionOutcomes(" + str(t) + ",[" + eH + "],SActs,Probs)."
        possStochActions = list(self.prolog.query(query))[0]['SActs']
        probs = list(self.prolog.query(query))[0]['Probs']
        return possStochActions, probs
        
    def reward(self,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getRewardRL([" + eH + "],R)."
        reward = list(self.prolog.query(query))[0]['R']
        return reward
    
    def getState(self,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getState([" + eH + "],State)."
        bitState = list(self.prolog.query(query))[0]['State']
        return bitState
    
    def getConState(self,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getCCState([" + eH + "],State)."
        ccState = list(self.prolog.query(query))[0]['State']
        return ccState

    def getRun(self,eH):
        """
        [Refer to QMI function documentation.]
        """
        query = "getRun([" + eH + "],C)."
        currentRun = list(self.prolog.query(query))[0]['C']
        return currentRun
            
    def done(self,eH):
        """
        [Refer to QMI function documentation.]
        """
        s = "done([" + eH + "])."
        if (list(self.prolog.query(s))):
            result = True
        else:
            result = False
        return result            
    
    def getDomainParams(self):
        """
        Returns various size parameters of the domain.

        Returns
        -------
        actionSize : integer
            The number of agent actions available.
        stateSize : integer
            The size of state space. A structure of such size must be allocated.
        bitState : integer[]
            A list of as many items as the domain predicates. Initialized to zero.
        obsType : {discrete, continuous}
            Returns a string on wether the state is to be discrete (a bit list) or continuous (a list of real values).
        """
        
        actionSize = list(self.prolog.query("actionSize(L)."))[0]['L']
        stateSize = list(self.prolog.query("stateSizeBits(S)."))[0]['S']
        numRuns = list(self.prolog.query("getNumRuns(R)."))[0]['R']
        bitState = list(self.prolog.query("getState([],S)."))[0]['S']
        obsType = list(self.prolog.query("getObsType(X)."))[0]['X']
        actualStateSize = 2**(stateSize*numRuns)
        return actionSize,actualStateSize,[bitState]*numRuns,obsType,numRuns
    
    def getStateShapeInfo(self):
        """
        Returns the shape info of a continuous state.

        Returns
        -------
        shapeInfo : A list containing three lists.
            The first is the list of terms designated for state representation, and the second and third the minimum and maximum values of each of these terms.

        """
        shapeInfo = list(self.prolog.query("getStateShapeInfo(T,Min,Max)."))[0]
        return shapeInfo
 
    def achieved(self,eH):
        """
        Returns if the goal is achieved after effect history eH.

        Parameters
        ----------
        eH : String
            A string of the form "i_1, i_2, ..." each integer representing a task of the goal model (after multi-run correction).

        Returns
        -------
        result : boolean
            True if the root goal is achieved at eH, false otherwise.

        """
        s = "achieved([" + eH + "])."
        if (list(self.prolog.query(s))):
            result = True
        else:
            result = False
        return result
    
    def getTransState(self, eH):
        """
        Retrives the cross-run state at history the latest run of eH

        Parameters
        ----------
        eH : String
            A string of the form "i_1, i_2, ..." each integer representing a task of the goal model (after multi-run correction).
            
        Returns
        -------
        The cross-run state in the form of a list of predicates.

        """
        #print("Getting trans state for {}".format(eH))
        s = "getTransState([" + eH + "],X)."
        ts = str(list(self.prolog.query(s))[0]['X'])
        return(ts.replace("'",""))

    def setTransState(self, tS):
        """
        Sets the cross-run state via asserting the inital state of the corresponding fluents. 

        Parameters
        ----------
        tS : String
            The cross-run state exactly as retrieved from the getTransState.

        Returns
        -------
        None.

        """
        self.prolog.retractall("init(_)")
        s = "init(" + tS + ")"
        #print("Asserting: {}".format(s))
        self.prolog.assertz(s)
    
    def getInfeasibleActionPenalty (self):
        """
        Retrieves the reward penalty for invoking an infeasible action.

        Returns
        -------
        penalty : Float
            The penalty incurred by the attempt of the infeasible action.

        """
        s = "getInfeasiblePenalty(P)."
        penalty = list(self.prolog.query(s))[0]['P']
        return penalty
    
    def close(self):
        self.prolog.retractall("init(_)")
        del self.prolog
