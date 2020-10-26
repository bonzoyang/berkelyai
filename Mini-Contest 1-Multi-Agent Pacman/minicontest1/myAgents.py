# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='ClosestDotAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

import numpy as np
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import random
import math

from game import Actions, Directions

class MyGraph:
    def __init__(self, adjm):
        self.adjm = adjm
        self._mask = [ [ bool(v) for v in adj] for adj in self.adjm ]
        self.mst = [ [ 0 for v in adj] for adj in self.adjm ]

    @staticmethod
    def edge_num(adjm):
        l = len(adjm)
        ex2 = 0 # for e*2
        for v in adjm:
            ex2 += l - v.count(0)
        return ex2//2
    
    @staticmethod
    def get_min_edge(adjm, i):
        m = float('inf')
        j = None
        for _, e in enumerate(adjm[i]):
            if e > 0 and e < m:
                m = e
                j = _
        return m, j

    @staticmethod
    def vertex_num(adjm):
        return len(adjm)
    
    def filter_adjm(self):
        return [ [ v if m else 0 for m, v in zip(msk, adj) ] for msk, adj in zip(self._mask, self.adjm) ]
    
    def reset_mask(self):
        self._mask = [ [ bool(v) for v in adj] for adj in self.adjm ]
        return self._mask
    
    def minimum_spanning_tree(self, nodes = None):
        expand = [0]
        adjm = self.filter_adjm()
        vm = self.vertex_num(adjm)
        self._parent = {}

        edges = nodes-1 if nodes is not None else vm -1
        while self.edge_num(self.mst) != edges:
            print(self.edge_num(self.mst), vm)
            m = float('inf')
            i = None
            for ex_ in expand:
                m_, i_ = self.get_min_edge(adjm, ex_)
                if m_ < m:
                    m = m_
                    i = i_
                    ex = ex_
                    
            else:
                expand.append(i)
                
                cnt = 0
                circle_check_fail = False
                son = ex
                while cnt < len(self._parent):
                    #print(f'now ex = {ex}, son = {son}, adjm = {adjm}, cnt = {cnt}, pnum = {len(self._parent)}')
                    father = self._parent[son]
                    print(f'father {father} son {son}')
                    

                    if father == i:
                        print(f'fucked up!!!!!!!!!!!!!! father:{father}')
                        circle_check_fail = True
                        break
                    elif father in self._parent:
                        son = father
                        father = self._parent[son]
                    else:
                        break
                    
                    cnt +=1
                                    
                if not circle_check_fail:
                    print(f"{ex}'s parent {i} is added")
                    self._parent[i] = ex
                    self.mst[ex][i] = m
                    self.mst[i][ex] = m
                    
                self._mask[ex][i] = False
                self._mask[i][ex] = False
                adjm = self.filter_adjm()


# class MyAgent(Agent):
#     """
#     Implementation of your agent.
#     """
#     init_done = False

    
#     @staticmethod
#     def manhattan(p1, p2):
#         # Manhattan distance
#         return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

#     @classmethod
#     def info(cls):
#         slist = [
#             f'cls.foods: {cls.foods}', 
#             f'cls.foods: {len(cls.foods.asList())} {cls.foods.asList()}', 
#             ]
        
#         s = '\n'.join(slist)
#         print('====================================')
#         print(s)
#         print('====================================')

#     @classmethod
#     def classLevelInit(cls, index, initState):
#         cls.init_done  = True
#         cls.foods = initState.getFood()
#         cls.foodsCount = cls.foods.count()
#         import math
#         cls.N = [1]* 15 + [2]*30 + [3]* 25 + [4] * 15 + [5]*10 + [6]*5

#         cls.walls = initState.getWalls()
#         cls.foodDict = {}
#         cls.foodReverseDict = {}
#         cls.agentNumber = initState.getNumPacmanAgents()
#         cls.agentPositions = initState.getPacmanPositions()
#         for i, f in enumerate(cls.foods.asList()):
#              cls.foodDict[i] = f
#              cls.foodReverseDict[f] = i

#     # def food_cost(self):
#     #     c = 0
#     #     for row in self.adjm:
#     #         c += sum(row)
#     #     return c//2
    
#     def getAction(self, state):
#         """
#         Returns the next action the agent will take
#         """

#         "*** YOUR CODE HERE ***"
#         if not self.__class__.init_done:
#             self.__class__.classLevelInit(self.index, initState = state)
        
#         self.__class__.agentPositions = state.getPacmanPositions()
#         if self.foods is None:
#             self.foods = self.__class__.foods



        
#         problem = FoodSearchProblem(state, self.index)
#         problemFoods = problem.startState[1]
#         #problemFoods = self.foods
#         #N = problemFoods.count() % 4+1 if problemFoods.count() >= self.agentNumber else 1
#         N = self.__class__.N[ int(problemFoods.count()/self.__class__.foodsCount * 100) -1 ] if problemFoods.count() >= self.agentNumber else 1
#         if problemFoods.count() < self.__class__.agent*2:

#         #print(f'N={N}, problemFoods.count():{problemFoods.count()}, self.__class__.foodsCount:{self.__class__.foodsCount}')
#         #print(f'agent:{self.index}, \nproblem.startState[1]:{problem.startState[1].count()}/{problem.startState[1].asList()}, \nself.__class__.foods:{self.__class__.foods.count()}/{self.__class__.foods.asList()}')

#         elif not self.traceAction:
#             i = 0

#             x, y = self.__class__.agentPositions[self.index]
            
            
#             #foods = self.foods
#             foods = problemFoods
#             thisIsFood = foods[x][y]

#             #nextState = (x, y), foods, thisIsFood
#             nextState = (x, y), self.__class__.foods, thisIsFood
#             while self.foods.count() != 0 and i < N:

#                 #print(f'agent:{self.index} round:{i}')
#                 problem.startState = nextState
#                 actions, nextState = aStarSearch(problem, nullHeuristic)
#                 #print(f'{self.index}, actions:{actions}, nextState:{nextState}, foods:nextState:{nextState[1].asList()}')
#                 # mask food, change problem state
#                 fx, fy = nextState[0]
#                 nextState[1][fx][fy] = False
#                 nextState = (nextState[0], nextState[1], False)

#                 self.__class__.foods[fx][fy] = False
#                 #problem.startState = nextState, problem.foods
#                 self.nextFoods.append(nextState[0])
#                 i += 1
#                 #print(f'nextState:{nextState}')
#                 self.traceAction = self.traceAction + actions

                
#             else:
#                 #print(f'i am:{self.index}, my target:{self.nextFoods}, actions:{self.traceAction}')
#                 pass




#         a = Directions.STOP
#         if self.traceAction:
#             a = self.traceAction.pop(0)
        
#         # print(f'self.traceAction:{self.traceAction}')
#         # print(f'a:{a}')
#         return a

    
#     def initialize(self):
#         """
#         Intialize anything you want to here. This function is called
#         when the agent is first created. If you don't need to use it, then
#         leave it blank
#         """

#         "*** YOUR CODE HERE"
#         self.traceAction=[]
#         self.nextFoods=[]
#         self.foods = None


        

        

#         # raise NotImplementedError()
#         pass

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    init_done = False
    N = [1]*15 + [2]*30 + [3]*25 + [4]*15 + [5]*10 + [6]*5
    agent_traceAction = {}
    agent_powerDown = {}
    neverPowerDown = False

    @classmethod
    def classLevelInit(cls, index, state):
        cls.init_done  = True
        cls.foods = state.getFood()
        cls.foodsCount = cls.foods.count()
        cls.foodsCoor = cls.foods.asList()
        cls.foodsCoorArray = np.array(cls.foodsCoor)
        cls.agentNumber = state.getNumPacmanAgents()
        cls.startPosition = state.getPacmanPositions()
        cls.threshold = (state.getWidth() + state.getHeight())

        if cls.agentNumber == 1 or cls.foodsCount/cls.agentNumber < 1.25:
            cls.neverPowerDown = True

        for i in range(cls.agentNumber):
            cls.agent_traceAction[i] = list()
            cls.agent_powerDown[i] = False
    
    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        if not self.__class__.init_done:
            self.__class__.classLevelInit(self.index, state = state)

        if not self.__class__.agent_traceAction[self.index] and not self.__class__.agent_powerDown[self.index]:
            fc = state.getFood().count()
            n = int(fc /self.__class__.foodsCount * 100) - 1 # max = 99
            N = self.__class__.N[ 0 if n < 0 else n ]               
            if N > fc and fc <= self.__class__.agentNumber*2:
                N = 1
            elif N > fc:
                N = fc

            #print(f'N:{N}')
            i = 0
            problem = AnyFoodSearchProblem(state, self.index)
            nextState = state.getPacmanPosition(self.index)
            # self.nextFoods = []

            while self.__class__.foods.count() != 0 and i < N:
                problem.startState = nextState
                problem.food = self.__class__.foods
                #actions, nextState = aStarSearch(problem, nullHeuristic)
                actions, nextState = breadthFirstSearch(problem)

                # self.nextFoods.append(nextState)
                if self.__class__.neverPowerDown: # 食物太稀疏或agent太少
                    if  self.__class__.foods.count() < self.__class__.agentNumber * 4  and False in [ self.__class__.agent_powerDown[k] for k in self.__class__.agent_powerDown  if k != self.index]:
                        # 自己以外如果有人還沒關機，自己就可以被關機    
                        if len(actions) > self.__class__.threshold*0.75 and len(actions) > sum([len(v) for v in self.__class__.agent_traceAction.values()]) :
                            self.__class__.agent_powerDown[self.index] = True
                            print(f'agent {self.index} stopped')
                            break

                if len(actions) <= self.__class__.threshold*0.5:
                    self.__class__.foods[ nextState[0] ][ nextState[1] ] = False
                self.__class__.agent_traceAction[self.index] = self.__class__.agent_traceAction[self.index] + actions
                i += 1
            else:
                pass

        a = Directions.STOP
        if self.__class__.agent_traceAction[self.index]:
            a = self.__class__.agent_traceAction[self.index].pop(0)

        return a

    
    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        self.traceAction=[]
        self.nextFoods=[]
        if self.index == 0:
            self.__class__.init_done = False
            self.__class__.neverPowerDown =False
            self.__class__.agent_traceAction = {}
            self.__class__.agent_powerDown = {}

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def manhattanHeuristic(state, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    x,y = state
    return min([abs(x - f[0]) + abs(y - f[1])  for f in problem.food.asList() ])

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    traverse = util.PriorityQueue()
    state_info = (problem.getStartState(), None, 0, None) # (state, action, cost, parent)
    #print(f'@aStarSearch state_info:{state_info}')
    state = state_info[0]
    come_from = {}    
    explored = []
    action = []

    while not problem.isGoalState(state):
        for s, a, c in problem.getSuccessors(state):
            g = c + state_info[2]
            h = heuristic(s, problem)
            f = g + h
            
            if s not in explored:
                come_from[(s, a, g, state)] = state_info
                traverse.push((s, a, g, state), f) 
        else:
            explored.append(state)
            state_info = traverse.pop()
            s, _, _, _ = state_info
            
            while s in explored:
                state_info = traverse.pop()
                s, _, _, _ = state_info
            state = state_info[0]
    else:
        # if hit goal back track
        s, a, g, parent = state_info
        
        while parent is not None:
            action.insert(0, a)
            state_info = come_from[state_info]
            _, a, _, parent = state_info
        
    return action, state

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    traverse = util.Queue()
    state_info = (problem.getStartState(), None, None, None) # (state, action, cost, parent)
    state = state_info[0]
    come_from = {}    
    explored = []
    action = []

    while not problem.isGoalState(state):
        for s, a, c in problem.getSuccessors(state):
            if s not in explored:
                come_from[(s, a, c, state)] = state_info
                traverse.push((s, a, c, state)) 
        else:
            explored.append(state)
            state_info = traverse.pop()
            s, _, _, _ = state_info
            
            while s in explored:
                state_info = traverse.pop()
                s, _, _, _ = state_info
            state = state_info[0]
            
    else:
        # if hit goal back track
        s, a, c, p = state_info
        
        while p is not None:
            action.insert(0, a)
            state_info = come_from[state_info]
            _, a, _, p = state_info
    
    return action, state

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
        actions = search.breadthFirstSearch(problem)
        return actions

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex), gameState.getFood()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]


# class FoodSearchProblem:
#     """
#     A search problem associated with finding the a path that collects all of the
#     food (dots) in a Pacman game.

#     A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
#       pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
#       foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
#     """
#     def __init__(self, startingGameState, agentIndex):
#         #self.foods = startingGameState.getFood()
#         x, y = startingGameState.getPacmanPosition(agentIndex)
#         foods = startingGameState.getFood()
#         self.startState = ((x, y), foods, foods[x][y])
#         self.walls = startingGameState.getWalls()
#         #self.startingGameState = startingGameState
#         self._expanded = 0 # DO NOT CHANGE
#         self.heuristicInfo = {} # A dictionary for the heuristic to store information

#     def getStartState(self):
#         return self.startState

#     # def isGoalState(self, state):
#     #     return state[1].count() == 0

#     def isGoalState(self, state):
#         "*** YOUR CODE HERE ***"
#         isGoal = state[2]
#         return isGoal

#     def getSuccessors(self, state):
#         "Returns successor states, the actions they require, and a cost of 1."
#         successors = []
#         self._expanded += 1 # DO NOT CHANGE
#         for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
#             x,y = state[0]
#             dx, dy = Actions.directionToVector(direction)
#             nextx, nexty = int(x + dx), int(y + dy)
#             if not self.walls[nextx][nexty]:
#                 nextFood = state[1].copy()
#                 thisIsFood = nextFood[nextx][nexty]
#                 nextFood[nextx][nexty] = False
#                 successors.append( ( ((nextx, nexty), nextFood, thisIsFood), direction, 1) )
#                 # don't change food map here, chage in your agent.getAction() 
#                 #successors.append( ( ((nextx, nexty), state[1]), direction, 1) )
#         return successors

#     def getCostOfActions(self, actions):
#         """Returns the cost of a particular sequence of actions.  If those actions
#         include an illegal move, return 999999"""
#         x,y= self.getStartState()[0]
#         cost = 0
#         for action in actions:
#             # figure out the next state and see whether it's legal
#             dx, dy = Actions.directionToVector(action)
#             x, y = int(x + dx), int(y + dy)
#             if self.walls[x][y]:
#                 return 999999
#             cost += 1
#         return cost


# class SpecificFoodSearchProblem(PositionSearchProblem):
#     """
#     A search problem for finding a path to a specific food.

#     This search problem is just like the PositionSearchProblem, but has a
#     different goal test, which you need to fill in below.  The state space and
#     successor function do not need to be changed.

#     The class definition above, SpecificFoodSearchProblem(PositionSearchProblem),
#     inherits the methods of the PositionSearchProblem.

#     You can use this search problem to help you fill in the findPathToClosestDot
#     method.
#     """

#     def __init__(self, gameState, agentIndex, goal):
#         "Stores information from the gameState.  You don't need to change this."
#         # Store the food for later reference
#         self.food = gameState.getFood()

#         # Store info for the PositionSearchProblem (no need to change this)
#         self.walls = gameState.getWalls()
#         self.startState = gameState.getPacmanPosition(agentIndex)
#         self.costFn = lambda x: 1
#         self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
#         self.goal = goal
        
#     def isGoalState(self, state):
#         """
#         The state is Pacman's position. Fill this in with a goal test that will
#         complete the problem definition.
#         """
#         #x,y = state

#         "*** YOUR CODE HERE ***"
#         return state == self.goal



