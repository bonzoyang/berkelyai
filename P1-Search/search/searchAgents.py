# searchAgents.py
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################
class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState, costFn = lambda x: 1, goal=((None, None), (True, True, True, True)), start=None, warn=True, visualize=True):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        if start != None: self.startState = start
        self.cornerVisited = tuple([True if self.startingPosition == _ else False for _ in self.corners])
        self.startState = (self.startingPosition, self.cornerVisited)
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist= {}, [] # DO NOT CHANGE
        

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return self.startState

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        isGoal = state[1] == self.goal[1]

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state[0])
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            (x, y), cornersVisited = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            cornersVisited = tuple([ True if c == (nextx, nexty) else v for c, v in zip(self.corners, cornersVisited) ])
            if not self.walls[nextx][nexty]:
                nextState = ((nextx, nexty), cornersVisited)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        self._expanded += 1 # DO NOT CHANGE
        if state[0] not in self._visited:
            self._visited[state[0]] = True
            self._visitedlist.append(state[0])
            
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    import math
    def dist(p1, p2):
        return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    
    
    nextc, cornerVisited = state
    cornerCalculated = list(cornerVisited)
    d = 0
    while False in cornerCalculated:
        dmap = {}
        for i, c in enumerate(corners):
            dmap[dist(nextc, c) if not cornerCalculated[i] else float('inf')] = i 
        
        d += min(dmap)
        key = dmap[min(dmap)]
        nextc = corners[key]
        cornerCalculated[key] = True

    return  d# Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    
    # see how many foods are left
    return len(foodGrid.asList()) # Search nodes expanded: 12517


def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    import math
    def dist(p1, p2):
        return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    
    x, y = (0, 0)
    d = 0
    foods = foodGrid.asList()

    for fx, fy in foods:
        x += fx
        y += fy
    else:
        n = len(foods)
        if n ==0:
            d = 0
        else: 
            center = (x/n, y/n)
            d = dist(position, center)
    
    # get all food's center
    return d # Search nodes expanded: 12845

# for foodHeuristic(state, problem)
def dist(p1, p2):
    # Euclidean distance
    import math
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def manhattan(p1, p2):
    # Manhattan distance
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def nextto(p1, p2):
    # check if Manhattan distance is 1
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) == 1

def center(comp):
    # component's center of mass
    cx = 0
    cy = 0
    for x, y in comp:
        cx += x
        cy += y
    return cx/len(comp), cy/len(comp)

def closest(p, comp):
    # cloest Manhattan distance to start p
    comp = list(comp)
    d = [manhattan(p, _) for _ in comp]
    return min(d)

def get_connected_component(foods):
    # get connected component
    # build adj_matrix
    adj_matrix = {}
    for f1 in foods:
        adj_matrix[f1] = []
        for f2 in foods:
            if nextto(f1, f2):
                adj_matrix[f1].append(f2)
   
    # find component
    for i in range(len(adj_matrix)):
        for k in adj_matrix:
            comp = [ adj_matrix[_] for _ in adj_matrix[k] if _ != k ]
            temp = adj_matrix[k]
            for _ in comp:

                temp = temp + _
            comp = list(set(temp))
            adj_matrix[k] = list(filter(lambda x: x != k, comp))
    
    for k in adj_matrix:
        adj_matrix[k] = set([k] + adj_matrix[k])
    
    # remove repeated component
    component=[]
    for k in adj_matrix:
        if adj_matrix[k] not in component:
            component.append(adj_matrix[k])
            
    return component
         
def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    foods = foodGrid.asList()
    
    # connected component
    component = get_connected_component(foods)   
    # component center
    cc = [ center(_) for _ in component]
    
    start = position
    count = 0
    cost = 0
    componentMask = [False]*len(component)
    for r in range(len(cc)):
        
        d = [ dist(start, _ ) if not m else float('inf') for _,  m in zip(cc, componentMask)  ]
        # get min distance to center
        i =  d.index(min(d))
        nearest = cc[i]
        componentMask[i] = True
        
        if start[0] > nearest[0] and start[1] > nearest[1]: # toward SW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] and nearest[1] > _[1] ])
        elif start[0] < nearest[0] and start[1] > nearest[1]: # toward SE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] and nearest[1] > _[1]])
        elif start[0] < nearest[0] and start[1] < nearest[1]: # toward NE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] and nearest[1] < _[1]])
        elif  start[0] > nearest[0] and start[1] < nearest[1]: # toward NW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] and nearest[1] < _[1]])
            
        
        cost += (count + min(d))
        start = cc[i]

    # get all connected component，then genterate all component's center
    # next measure pacman's position to the nearest center, and plus 
    # food count in opposite direction relative to the quadrant of start point
    # then take privious center as next start position, then repeat the process
    return cost # Search nodes expanded: 7498

def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"    
    foods = foodGrid.asList()
    
    # connected component
    component = get_connected_component(foods)  
    # component center
    cc = [ center(_) for _ in component]
    
    start = position
    count = 0
    cost = 0
    componentMask = [False]*len(component)
    for r in range(len(cc)):
        
        d = [ manhattan(start, _ ) if not m else float('inf') for _,  m in zip(cc, componentMask)  ]
        # get min distance to center
        i =  d.index(min(d))
        nearest = cc[i]
        componentMask[i] = True
        
        if start[0] > nearest[0] and start[1] > nearest[1]: # toward SW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] and nearest[1] > _[1] ])
        elif start[0] < nearest[0] and start[1] > nearest[1]: # toward SE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] and nearest[1] > _[1]])
        elif start[0] < nearest[0] and start[1] < nearest[1]: # toward NE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] and nearest[1] < _[1]])
        elif  start[0] > nearest[0] and start[1] < nearest[1]: # toward NW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] and nearest[1] < _[1]])
            
        
        cost += (count + min(d))
        start = cc[i]

    # same as previous version, beside using manhattan as metric of distance to center
    return cost # Search nodes expanded: 6724

def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    d = 0
    foods = foodGrid.asList()
    
    # connecter component
    component = get_connected_component(foods)
            
    # component center
    cc = [ center(_) for _ in component]
    
    start = position
    count = 0
    cost = 0
    componentMask = [False for _ in component]
    for r in range(len(cc)):
        
        d = [ manhattan(start, _ ) if not m else float('inf') for _,  m in zip(cc, componentMask)  ]
        # get min distance to center
        i =  d.index(min(d))
        nearest = cc[i]
        componentMask[i] = True
        
        if start[0] > nearest[0] and start[1] > nearest[1]: # toward SW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] or nearest[1] > _[1] ])
        elif start[0] < nearest[0] and start[1] > nearest[1]: # toward SE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] or nearest[1] > _[1]])
        elif start[0] < nearest[0] and start[1] < nearest[1]: # toward NE
            count += len([ _ for _ in component[i] if nearest[0] < _[0] or nearest[1] < _[1]])
        elif  start[0] > nearest[0] and start[1] < nearest[1]: # toward NW
            count += len([ _ for _ in component[i] if nearest[0] > _[0] or nearest[1] < _[1]])

            
        cost += (count + min(d))
        start = cc[i]

    # same as previous version, beside food count in opposite quadrant of start point
    # changed to count all quadrants but start point's quadrant
    return cost # Search nodes expanded: 5182

def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    import math
    foods = foodGrid.asList()
    
    # connecter component
    component = get_connected_component(foods)  
    # component center
    start = position
    cc = [ center(_) for _ in component]
    cclosest = [closest(start, _) for _ in component] # every component closest

    # distribute quadrant by component center
    quarter = {1:[], 2:[], 3:[], 4:[]}
    for i, c in enumerate(cc):
        if c[0] > start[0] and c[1] >= start[1]:
            quarter[1].append(i)
        elif c[0] <= start[0] and c[1] > start[1]:
            quarter[2].append(i)
        elif c[0] < start[0] and c[1] <= start[1]:
            quarter[3].append(i) 
        else:
            quarter[4].append(i)
        
    def component_cost(start, cc, component):
        # get one quadrant's component cost, which will be used to decide 
        # start point of heuristic function latter
        
        componentMask = [False]*len(cc)
        count=float('inf')
        cost=float('inf')
        for r in range(len(cc)):
            if r == 0:
                count=0
                cost=0
            
            d = [ manhattan(start, _ ) if not m else float('inf') for _,  m in zip(cc, componentMask)  ]
            # get min distance to center
            i =  d.index(min(d))
            nearest = cc[i]
            componentMask[i] = True
            
            # same: use to check if there are food in the same quadrant of start point
            #       if same == 0, there are no points in the same quadrant of center, 
            #                     thus won't plus distance from start to center into cost
            #                     
            #       if same != 0, use distance from start to center as cost of quadrant of
            #                     start point
            same = 0
            if len(component[i]) == 1:
                same = 1
            elif  start == nearest:
                count += len([ _ for _ in component[i] if not _ == nearest ])
            elif start[0] > nearest[0] and start[1] >= nearest[1]: # toward SW
                same =  len([ _ for _ in component[i] if (_[0] > nearest[0] and _[1] >= nearest[1]) and _ != nearest])
                count += len([ _ for _ in component[i] if not (_[0] > nearest[0] and _[1] >= nearest[1]) and _ != nearest])
            elif start[0] <= nearest[0] and start[1] > nearest[1]: # toward SE
                same = len([ _ for _ in component[i] if (_[0] <= nearest[0] and _[1] > nearest[1]) and _ != nearest])
                count += len([ _ for _ in component[i] if not (_[0] <= nearest[0] and _[1] > nearest[1]) and _ != nearest ])
            elif start[0] < nearest[0] and start[1] <= nearest[1]: # toward NE
                same = len([ _ for _ in component[i] if (_[0] < nearest[0] and _[1] <= nearest[1]) and _ != nearest])
                count += len([ _ for _ in component[i] if not (_[0] < nearest[0] and _[1] <= nearest[1]) and _ != nearest ])
            elif  start[0] >= nearest[0] and start[1] < nearest[1]: # toward NW
                same = len([ _ for _ in component[i] if (_[0] >= nearest[0] and _[1] < nearest[1]) and _ != nearest])
                count += len([ _ for _ in component[i] if not (_[0] >= nearest[0] and _[1] < nearest[1]) and _ != nearest ])


            if same == 0:
                cost += count
            else:
                cost += (count + min(d))

            start = cc[i]

        cost = cost
        return cost
    
    cost1 = component_cost(start, cc=[cc[i] for i in quarter[1]], component=[component[i] for i in quarter[1]])
    cost2 = component_cost(start, cc=[cc[i] for i in quarter[2]], component=[component[i] for i in quarter[2]])
    cost3 = component_cost(start, cc=[cc[i] for i in quarter[3]], component=[component[i] for i in quarter[3]])
    cost4 = component_cost(start, cc=[cc[i] for i in quarter[4]], component=[component[i] for i in quarter[4]])
    
    
    # index +1 because quadrant's index start from 1
    # at first, move to quadrant with minimun cost    
    m = min([cost1, cost2, cost3, cost4])
    if m == float('inf'):
        q = None
    else:
        q = [cost1, cost2, cost3, cost4].index(m)+1
        
    count = 0
    cost = 0
    componentMask = [False]*len(component)
    for r in range(len(cc)):
        if r == 0 and q is not None:
            # go to cloest component of quadrant with minimun cost first
            d = [manhattan(start, cc[i])  for i in quarter[q]]
            _ = d.index(min(d))
            i = quarter[q][_]
           
        else:
            # then move to component with closest center
            d = [ manhattan(start, _ ) if not m else float('inf') for _,  m in zip(cc, componentMask)  ]
            i =  d.index(min(d))
        
        nearest = cc[i]
        componentMask[i] = True
        
        # same: use to check if there are food in the same quadrant of start point
        #       if same == 0, there are no points in the same quadrant of center, 
        #                     thus won't plus distance from start to nearest point of component
        #                     
        #       if same != 0, use distance from start to nearest point of component        
        same = 0
        if len(component[i]) == 1:
            same = 1
        elif  start == nearest:
            count += len([ _ for _ in component[i] if not _ == nearest ])
        elif start[0] > nearest[0] and start[1] >= nearest[1]: # toward SW
            same = len([ _ for _ in component[i] if (_[0] > nearest[0] and _[1] >= nearest[1]) and _ != nearest])
            count += len([ _ for _ in component[i] if not (_[0] > nearest[0] and _[1] >= nearest[1]) and manhattan(_, nearest) >= 1  and _ != nearest])
        elif start[0] <= nearest[0] and start[1] > nearest[1]: # toward SE
            same = len([ _ for _ in component[i] if (_[0] <= nearest[0] and _[1] > nearest[1]) and _ != nearest])
            count += len([ _ for _ in component[i] if not (_[0] <= nearest[0] and _[1] > nearest[1]) and manhattan(_, nearest) >= 1  and _ != nearest])
        elif start[0] < nearest[0] and start[1] <= nearest[1]: # toward NE
            same = len([ _ for _ in component[i] if (_[0] < nearest[0] and _[1] <= nearest[1]) and _ != nearest])
            count += len([ _ for _ in component[i] if not (_[0] < nearest[0] and _[1] <= nearest[1]) and manhattan(_, nearest) >= 1  and _ != nearest])
        elif  start[0] >= nearest[0] and start[1] < nearest[1]: # toward NW
            same = len([ _ for _ in component[i] if (_[0] >= nearest[0] and _[1] < nearest[1]) and _ != nearest])
            count += len([ _ for _ in component[i] if not (_[0] >= nearest[0] and _[1] < nearest[1]) and manhattan(_, nearest) >= 1  and _ != nearest])       
            
        if same == 0:
            cost += count
        else:
            cost += (count + cclosest[i])
            
        start = cc[i]
        cclosest = [closest(start, _) for _ in component]
    cost = cost
    return cost # Search nodes expanded: 5432




class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        actions = search.breadthFirstSearch(problem)
        return actions
        

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

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        isGoal = state in self.food.asList()
        return isGoal

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
