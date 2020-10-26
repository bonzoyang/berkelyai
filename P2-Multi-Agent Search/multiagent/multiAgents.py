# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newWalls = successorGameState.getWalls()
        maxDist = newWalls.height + newWalls.width
        newCapsules = successorGameState.getCapsules()
        def manhatton(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        g = [ int(manhatton(ghost.getPosition(), newPos) >1) for ghost in newGhostStates]
        g = sum(g)

        F = newFood.asList()
        f = 0
        if F:
            d = [ manhatton(food, newPos) for food in F]
            f = min(d)/maxDist
        

        c = 0
        if newCapsules:
            c = [ manhatton(capsule, newPos) for capsule in newCapsules ]
            c = min(c)/maxDist + (len(newCapsules)-1)
        
        score = g - 0.1*f -0.1*c - ( newFood.count()-1 if newFood.count() >0 else 0 ) - ( len(newCapsules)-1 if len(newCapsules) > 0 else 0)  + successorGameState.getScore()
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def value(self, state, agent, depth):
        if state.isWin() or state.isLose() or depth == 0:
            return (None, self.evaluationFunction(state))
        elif agent == 0: # pacman
            return self.maxValue(state, agent, depth)
        else: # ghost
            return self.minValue(state, agent, depth)

    def maxValue(self, state, agent, depth):
        n = state.getNumAgents()
        actions = state.getLegalActions(agent)
        sucStates = [ state.generateSuccessor(agent, action) for action in actions ]
        d = dict( [ (a, self.value(s, (agent+1)%n, depth)[1]) for a, s in zip(actions, sucStates) ] )
        d = sorted(d.items(), key=lambda x:x[1], reverse=True) # max
        return d[0]

    def minValue(self, state, agent, depth):
        n = state.getNumAgents()
        depth = depth - 1 if agent == n-1 else depth
        actions = state.getLegalActions(agent)
        sucStates = [ state.generateSuccessor(agent, action) for action in actions ]
        d = dict( [ (a, self.value(s, (agent+1)%n, depth)[1]) for a, s in zip(actions, sucStates) ] )
        d = sorted(d.items(), key=lambda x:x[1]) # min
        return d[0]

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, score = self.value(gameState, 0, self.depth)
        return action

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, state, agent, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return (None, self.evaluationFunction(state))
        elif agent == 0: # pacman
            return self.maxValue(state, agent, depth, alpha, beta)
        else: # ghost
            return self.minValue(state, agent, depth, alpha, beta)

    def maxValue(self, state, agent, depth, alpha, beta):
        n = state.getNumAgents()
        actions = state.getLegalActions(agent)
        # sucStates = [ state.generateSuccessor(agent, action) for action in actions ]
        # d = [ (a, self.value(s, (agent+1)%n, depth, alpha, beta )[1]) for a, s in zip(actions, sucStates) ]
        # d = sorted(d.items(), key=lambda x:x[1], reverse=True) # max
        # return d[0]
        v = float('-inf')
        a = None
        for action in actions:
            s = state.generateSuccessor(agent, action)
            v_ = self.value(s, (agent+1)%n, depth, alpha, beta)[1]

            # bookkeep best value and action
            if v_ > v:
                v = v_
                a = action
            
            # prune
            if v > beta:
                return (a, v)
            
            # if not prune, update alpha
            alpha = max(alpha, v)
        return (a, v)


    def minValue(self, state, agent, depth, alpha, beta):
        n = state.getNumAgents()
        depth = depth - 1 if agent == n-1 else depth
        actions = state.getLegalActions(agent)
        # sucStates = [ state.generateSuccessor(agent, action) for action in actions ]
        # d = dict( [ (a, self.value(s, (agent+1)%n, depth, alpha, beta)[1]) for a, s in zip(actions, sucStates) ] )
        # d = sorted(d.items(), key=lambda x:x[1]) # min
        # return d[0]
        v = float('inf')
        a = None
        for action in actions:
            s = state.generateSuccessor(agent, action)
            v_ = self.value(s, (agent+1)%n, depth, alpha, beta)[1]

            # bookkeep best value and action
            if v_ < v:
                v = v_
                a = action

            # prune
            if v < alpha:
                return (a, v)
            
            # if not prune, update beta
            beta = min(beta, v)
        return (a, v)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.value(gameState, 0, self.depth, float("-inf"), float("inf"))
        return action
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, state, agent, depth):
        if state.isWin() or state.isLose() or depth == 0:
            return (None, self.evaluationFunction(state))
        elif agent == 0: # pacman
            return self.maxValue(state, agent, depth)
        else: # ghost
            return self.expValue(state, agent, depth)

    def maxValue(self, state, agent, depth):
        n = state.getNumAgents()
        actions = state.getLegalActions(agent)
        sucStates = [ state.generateSuccessor(agent, action) for action in actions ]
        d = dict( [ (a, self.value(s, (agent+1)%n, depth)[1]) for a, s in zip(actions, sucStates) ] )
        d = sorted(d.items(), key=lambda x:x[1], reverse=True) # max
        return d[0]

    def expValue(self, state, agent, depth):
        n = state.getNumAgents()
        depth = depth - 1 if agent == n-1 else depth
        actions = state.getLegalActions(agent)
        sucStates = [ state.generateSuccessor(agent, action) for action in actions ]

        e = sum( [self.value(s, (agent+1)%n, depth)[1] for a, s in zip(actions, sucStates) ] ) / len(actions) # exp
        return (None, e)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.value(gameState, 0, self.depth)
        return action

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # using almost same function as ReflexAgent, except different g, and sum(newScaredTimes) -g instead of g
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newWalls = successorGameState.getWalls()
    maxDist = newWalls.height + newWalls.width
    newCapsules = successorGameState.getCapsules()
    def manhatton(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    # g = [ int(manhatton(ghost.getPosition(), newPos) >1) for ghost in newGhostStates]
    g = [ manhatton(ghost.getPosition(), newPos) for ghost in newGhostStates if manhatton(ghost.getPosition(), newPos) >1]
    g = sum(g)

    F = newFood.asList()
    f = 0
    if F:
        d = [ manhatton(food, newPos) for food in F]
        f = min(d)/maxDist
    

    c = 0
    if newCapsules:
        c = [ manhatton(capsule, newPos) for capsule in newCapsules ]
        c = min(c)/maxDist + (len(newCapsules)-1)
    
    score = sum(newScaredTimes) - g - 0.1*f -0.1*c - ( newFood.count()-1 if newFood.count() >0 else 0 ) - ( len(newCapsules)-1 if len(newCapsules) > 0 else 0)  + successorGameState.getScore()
    return score    
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
