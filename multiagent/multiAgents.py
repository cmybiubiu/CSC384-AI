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

        "*** YOUR CODE HERE ***"
        # print (newPos)
        # print (newFood)
        # print (newGhostStates)
        # print (newScaredTimes)

        foodPos = newFood.asList()
        foodNum = len(foodPos)
        minDistance = 1e7

        for i in range(foodNum):
            distance = manhattanDistance(foodPos[i],newPos) + foodNum*100
            if distance < minDistance:
                minDistance = distance 

        if foodNum == 0:
            minDistance = 0

        output = -minDistance
        for i in range(len(newGhostStates)):
            ghostPos = successorGameState.getGhostPosition(i+1)
            if manhattanDistance(newPos,ghostPos)<=1 :
                return -1e6

        return output



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
        agent_num = gameState.getNumAgents()

        def DFMinMax(gameState, deepness, agent):

            # Another round and set pleyer to pacman
            if agent >= agent_num: 
                agent = 0
                deepness += 1

            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            #find best move 
            if (agent == 0): #pacman

                output = [" ", -1e5]
                pacActions = gameState.getLegalActions(agent)
            
                if not pacActions:
                    return self.evaluationFunction(gameState)
                
                for action in pacActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = DFMinMax(currState, deepness, agent+1)
                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue

                    if val > output[1]:
                        output = [action, val]
                return output

            else:  #ghost
                output = [" ", 1e5]
                ghostActions = gameState.getLegalActions(agent)
            
                if not ghostActions:
                    return self.evaluationFunction(gameState)
                
                for action in ghostActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = DFMinMax(currState, deepness, agent+1)
                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue
                    if val < output[1]:
                        output = [action, val]
                return output

        outputList = DFMinMax(gameState, 0, 0)
        return outputList[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlphaBeta(gameState, deepness, agent, alpha, beta):

        # agent be Min or Max
            if agent >= gameState.getNumAgents(): 
                agent = 0
                deepness += 1

        #find best move 
            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            if (agent == 0): #pacman

                output = [" ", -1e5]
                pacActions = gameState.getLegalActions(agent)
            
                if not pacActions:
                    return self.evaluationFunction(gameState)
                
                for action in pacActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = AlphaBeta(currState, deepness, agent+1, alpha, beta)

                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue

                    if val > output[1]:
                        output = [action, val]

                    alpha = max(alpha, val)
                    if alpha >= beta:
                        break

                return output

            else:#ghost
                output = [" ", 1e5]
                ghostActions = gameState.getLegalActions(agent)
            
                if not ghostActions:
                    return self.evaluationFunction(gameState)
                
                for action in ghostActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = AlphaBeta(currState, deepness, agent+1, alpha, beta)

                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue

                    if val < output[1]:
                        output = [action, val]

                    beta = min(beta, val)
                    if beta <= alpha:
                        break
                    

                return output

        outputList = AlphaBeta(gameState, 0, 0, -1e5, 1e5)

        return outputList[0]


        # agent_num = gameState.getNumAgents()
        # action_score = []

        # def Alpha_Beta(pos, turn_iter, alpha, beta):
        #     if pos.isLose() or pos.isWin() or self.depth * agent_num <= turn_iter:
        #         return self.evaluationFunction(pos)
        #     elif turn_iter % agent_num == 0:
        #         score = -1e7
        #         for act in pos.getLegalActions(0):
        #             successor = pos.generateSuccessor(0, act)
        #             score = max(score, Alpha_Beta(
        #                 successor, turn_iter + 1, alpha, beta))
        #             alpha = max(score, alpha)
        #             if turn_iter == 0:
        #                 action_score.append(score)
        #             if alpha >= beta:
        #                 break
        #         return score
        #     else:
        #         score = 1e7
        #         ghost_index = turn_iter % agent_num
        #         for act in pos.getLegalActions(ghost_index):
        #             successor = pos.generateSuccessor(ghost_index, act)
        #             score = min(score, Alpha_Beta(
        #                 successor, turn_iter + 1, alpha, beta))
        #             beta = min(beta, score)
        #             if alpha >= beta:
        #                 break
        #         return score

        # score = Alpha_Beta(gameState, 0, -1e7, 1e7)
        # opt_action = gameState.getLegalActions(
        #     0)[action_score.index(max(action_score))]
        # return opt_action




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
        "*** YOUR CODE HERE ***"
        def Expectimax(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1

            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            if (agent == 0):
                output = [" ", float(-1e7)]
                pacActions = gameState.getLegalActions(agent)
            
                if not pacActions:
                    return self.evaluationFunction(gameState)
                
                for action in pacActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = Expectimax(currState, deepness, agent+1)
                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue
                    if val > output[1]:
                        output = [action, val]
                return output

            else:
                output = [" ", 0]
                score = 0.0
                ghostActions = gameState.getLegalActions(agent)
            
                if not ghostActions:
                    return self.evaluationFunction(gameState)

                for action in ghostActions:
                    currState = gameState.generateSuccessor(agent, action)
                    currValue = Expectimax(currState, deepness, agent+1)
                    if type(currValue) is list:
                        val = currValue[1]
                    else:
                        val = currValue

                    output[0] = action
                    score += val 

                weight_sum_score = float(score) / len(gameState.getLegalActions(agent))
                return [output[0], weight_sum_score]
             
        outputList = Expectimax(gameState, 0, 0)
        return outputList[0]  

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodPos = currentGameState.getFood().asList() 
    foodDist = [] 
    currentPos = list(currentGameState.getPacmanPosition()) 
 
    for food in foodPos:
        f = manhattanDistance(food, currentPos)
        foodDist.append(-1*f)
        
    if not foodDist:
        foodDist.append(0)

    return max(foodDist) + currentGameState.getScore() 

# Abbreviation
better = betterEvaluationFunction

