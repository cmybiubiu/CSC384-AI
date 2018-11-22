# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    #Do not use full cycle checking for DFS
    open_data = util.Stack()
    open_data.push([[problem.getStartState()], []])
    while (not open_data.isEmpty()):
        [state, action] = open_data.pop()
        if (problem.isGoalState(state[-1])):
            return action
        for (s, a, c) in problem.getSuccessors(state[-1]):
            if not (s in state):
                open_data.push([state + [s],action + [a]])
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    open_data = util.Queue()
    startState = (problem.getStartState(), None, 0)
    open_data.push([startState])
    visited_list = {startState[0]: 0}

    while not open_data.isEmpty():
        node = open_data.pop()
        (end_state, _, _) = node[-1]
        actions = [action for (_, action, _) in node[1:]]
        if (problem.isGoalState(end_state)):
            return actions
        for succ in problem.getSuccessors(end_state):
            new_node = node + [succ]
            new_cost = sum([cost for (_, _, cost) in new_node])
            state = succ[0]
            if (not state in visited_list): #or (new_cost < visited_list[state]):
                open_data.push(new_node)
                visited_list[state] = new_cost
    return []
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    open_data = util.PriorityQueue()
    startState = (problem.getStartState(), None, 0)
    open_data.push([startState],0)
    visited_list = {problem.getStartState(): 0}

    while not open_data.isEmpty():
        node = open_data.pop()
        (end_state, _, _) = node[-1]
        actions = [action for (_, action, _) in node[1:]]
        cost = sum([c for (_, _, c) in node[1:]])

        if cost <= visited_list[end_state]:
            if (problem.isGoalState(end_state)):
                return actions
            for succ in problem.getSuccessors(end_state):
                new_node = node + [succ]
                new_cost = sum([c for (_, _, c) in new_node])
                state = succ[0]
                if (not state in visited_list) or new_cost < visited_list[state]:
                    open_data.update(new_node, new_cost )
                    visited_list[state] = new_cost

    return []
    #util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    open_data = util.PriorityQueue()
    startState = (problem.getStartState(), None, 0)
    open_data.push([startState],0)
    h_value = heuristic(problem.getStartState(), problem)
    visited_list = {problem.getStartState(): 0 + h_value}

    while not open_data.isEmpty():
        node = open_data.pop()
        (end_state, _, _) = node[-1]
        actions = [action for (_, action, _) in node[1:]]
        cost = sum([c for (_, _, c) in node[1:]])

        if cost <= visited_list[end_state]:
            if (problem.isGoalState(end_state)):
                return actions
            for succ in problem.getSuccessors(end_state):
                new_node = node + [succ]
                new_cost = sum([c for (_, _, c) in new_node])
                state = succ[0]
                if (not state in visited_list) or new_cost < visited_list[state]:
                    open_data.update(new_node, new_cost + heuristic(state, problem))
                    visited_list[state] = new_cost

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
