# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
from game import Actions

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        #
        oldFood = currentGameState.getFood()

        "*** YOUR CODE HERE ***"
        gx, gy = currentGameState.getGhostPosition(1)
        foodList = oldFood.asList()

        #Never go where a ghost can be!
        potGhostPosition = [(gx, gy)]
        potGhostPosition.append((gx+1, gy))
        potGhostPosition.append((gx-1, gy))
        potGhostPosition.append((gx, gy+1))
        potGhostPosition.append((gx, gy-1))

        

        if newPos in potGhostPosition:
          return 0
        elif newPos in foodList:
          return 1
        else:
          distances = []
          for pel in foodList:
            distances.append((((newPos[0] - pel[0]) ** 2 + (newPos[1] - pel[1]) **2 )** 0.5))
            distanceClosest = min(distances)
            #closestIndex = [index for index in range(len(distances)) if distances[index] == closestPel]
            return 1/(distanceClosest+1)


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
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        moveValues = []
        for move in legalMoves:
          moveValues.append(self.minimax(gameState.generateSuccessor(0, move), self.depth, min(1, gameState.getNumAgents() - 1)))
        #bestValue = max(moveValues)
        bestIndex  = [x for x in range(len(legalMoves)) if moveValues[x] == max(moveValues)]
        return legalMoves[bestIndex[0]]

    def minimax(self, gameState, depth, index):
      if depth == 0 or len(gameState.getLegalActions(index)) == 0:
        return self.evaluationFunction(gameState) #terminates
      if index == gameState.getNumAgents()-1: #sets up each ply
        nextDepth = depth-1
        nextIndex = 0
      else:
        nextDepth = depth
        nextIndex = index+1
      if index == 0:  #if pacman moves
        bestValue = float("-inf")
        for move in gameState.getLegalActions(index):
          bestValue = max(bestValue, self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex))
        return bestValue
      if index > 0: #ghosts!
        bestValue = float("inf")
        for move in gameState.getLegalActions(index):
          bestValue = min(bestValue, self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex))
        return bestValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalMoves = gameState.getLegalActions(0)
        moveValues = []
        alpha = float("-inf")
        beta  = float("inf")
        for move in legalMoves:
          value = self.minimax(gameState.generateSuccessor(0, move), self.depth, min(1, gameState.getNumAgents() - 1), alpha, beta)
          moveValues.append(value)
          alpha = max(alpha, value)
        #bestValue = max(moveValues)
        bestIndex  = [x for x in range(len(legalMoves)) if moveValues[x] == max(moveValues)]
        return legalMoves[bestIndex[0]]

    def minimax(self, gameState, depth, index, alpha, beta):
      if depth == 0 or len(gameState.getLegalActions(index)) == 0:
        return self.evaluationFunction(gameState) #terminates
      if index == gameState.getNumAgents()-1: #sets up each ply
        nextDepth = depth-1
        nextIndex = 0
      else:
        nextDepth = depth
        nextIndex = index+1
      if index == 0:  #if pacman moves
        v = float("-inf")
        for move in gameState.getLegalActions(index):
          v = max(v, self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex, alpha, beta))
          if v > beta:
            return v
          alpha = max(alpha, v)
        return v
      if index > 0: #ghosts!
        v = float("inf")
        for move in gameState.getLegalActions(index):
          v = min(v, self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex, alpha, beta))
          if v < alpha:
            return v
          beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        legalMoves = gameState.getLegalActions(0)
        moveValues = []
        for move in legalMoves:
          moveValues.append(self.minimax(gameState.generateSuccessor(0, move), self.depth, min(1, gameState.getNumAgents() - 1)))
        #bestValue = max(moveValues)
        bestIndex  = [x for x in range(len(legalMoves)) if moveValues[x] == max(moveValues)]
        return legalMoves[bestIndex[0]]

    def minimax(self, gameState, depth, index):
      if depth == 0 or len(gameState.getLegalActions(index)) == 0:
        return self.evaluationFunction(gameState) #terminates
      if index == gameState.getNumAgents()-1: #sets up each ply
        nextDepth = depth-1
        nextIndex = 0
      else:
        nextDepth = depth
        nextIndex = index+1
      if index == 0:  #if pacman moves
        bestValue = float("-inf")
        for move in gameState.getLegalActions(index):
          bestValue = max(bestValue, self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex))
        return bestValue
      if index > 0: #ghosts! #silly random ones!
        averageValue = float(0)
        for move in gameState.getLegalActions(index):
          averageValue += self.minimax(gameState.generateSuccessor(index, move), nextDepth, nextIndex)
        return averageValue/len(gameState.getLegalActions(index))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: currently it takes weighted of the following: distance to closest pellet and second closest and score
                    
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()


    distances = []
    distanceClosest = 0
    secondClosest = 0
    for pel in foodList:
      distances.append((((pos[0] - pel[0]) ** 2 + (pos[1] - pel[1]) **2 )** 0.5))
      #distances.append(mazeDistance(pos, pel, currentGameState))
    if len(distances) > 0:
      distanceClosest = distances.pop([x for x in range(len(distances)) if distances[x] == min(distances)][0])
    #closestIndex = [index for index in range(len(distances)) if distances[index] == closestPel]
    if len(distances) > 0:
      secondClosest = min(distances)

    #return ((10/(distanceClosest+1)) + (1/((sum(distances)/(len(distances)+1))+1)) + currentGameState.getScore())   
    return (1/(distanceClosest+1)) + (1/(secondClosest+1)) + currentGameState.getScore()

better = betterEvaluationFunction

