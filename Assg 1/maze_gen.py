import random
from pathlib import Path
import os

def printMaze(maze, height, width):
    """Displays the maze"""

    for y in range(height):
        for x in range(width):
            #Display the wall or empty space:
            print(maze[(x, y)], end='')
        print() 

def visit_iterative(hasVisited, maze, startX, startY, NORTH, SOUTH, EAST, WEST, WIDTH, HEIGHT, rng):
    #Blank cells mean walkable paths
    EMPTY = ' '

    #create stack
    stack = [(startX, startY)]
    hasVisited.add((startX, startY))
    maze[(startX, startY)] = EMPTY

    while stack:
        #Look at top of stack
        x, y = stack[-1]  

        unvisitedNeighbors = []

        if y > 1 and (x, y - 2) not in hasVisited:
            unvisitedNeighbors.append(NORTH)
        if y < HEIGHT - 2 and (x, y + 2) not in hasVisited:
            unvisitedNeighbors.append(SOUTH)
        if x > 1 and (x - 2, y) not in hasVisited:
            unvisitedNeighbors.append(WEST)
        if x < WIDTH - 2 and (x + 2, y) not in hasVisited:
            unvisitedNeighbors.append(EAST)

        if not unvisitedNeighbors:
            #Dead end reached so backtrack
            stack.pop()
            continue

        #Pick a random direction
        direction = rng.choice(unvisitedNeighbors)

        if direction == NORTH:
            nextX, nextY = x, y - 2
            pre_x, pre_y = x, y - 1
        elif direction == SOUTH:
            nextX, nextY = x, y + 2
            pre_x, pre_y = x, y + 1
        elif direction == WEST:
            nextX, nextY = x - 2, y
            pre_x, pre_y = x - 1, y
        elif direction == EAST:
            nextX, nextY = x + 2, y
            pre_x, pre_y = x + 1, y

        carve_chance = 0.70
        if rng.random() > carve_chance:
            #Keep the cell as a wall but mark as visited to avoid changing it later
            hasVisited.add((nextX, nextY))
            continue
        else:
            #Update cells to be empty thus walkable
            maze[(pre_x, pre_y)] = EMPTY
            maze[(nextX, nextY)] = EMPTY
            hasVisited.add((nextX, nextY))
            stack.append((nextX, nextY))

def create_maze(print = True, SEED = 1):
    """
    References: https://inventwithpython.com/recursion/chapter11.html
    """
    rng = random.Random(SEED)

    #Directions
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

    #Dimensions
    WIDTH = 101 # Width of the maze
    HEIGHT = 101 # Height of the maze

    #Keep track of visited cells
    hasVisited = set() 

    # Create the filled in maze data structure to start:
    maze = {}
    for x in range(WIDTH):
        for y in range(HEIGHT):
            #Every space is a wall at first
            maze[(x, y)] = '%' 
    
    # Carve out the paths in the maze data structure:
    start_x, start_y = rng.randrange(1, WIDTH, 2), rng.randrange(1, HEIGHT, 2)
    visit_iterative(hasVisited, maze, start_x, start_y, NORTH, SOUTH, EAST, WEST, WIDTH, HEIGHT, rng)

    # Visit all cells, restarting DFS if needed
    for y in range(1, HEIGHT, 2):
        for x in range(1, WIDTH, 2):
            if (x, y) not in hasVisited:
                visit_iterative(hasVisited, maze, start_x, start_y, NORTH, SOUTH, EAST, WEST, WIDTH, HEIGHT, rng)
                
    # Make sure random end goal is an already empty cell and not start cell
    end = rng.choice([cell for cell in hasVisited if maze[cell] == ' ' and cell != (start_x, start_y)])

    #Visuals to see start & end points in maze
    maze[(start_x, start_y)] = 'S'
    maze[end] = '!'

    if print:
        printMaze(maze, HEIGHT, WIDTH)

    return maze

def create_maze_path():
    root = Path(__file__).resolve().parent

    DATA = root / "generated_mazes"

    #Make sure training_data folder exists
    DATA.mkdir(parents= True, exist_ok= True)
    return DATA

def save_maze(maze, filename):
    """
    Function 
    """
    TEMP = create_maze_path()
    FILE = TEMP / filename
    with open(FILE, "w") as f:
        for y in range(101):
            for x in range(101):
                f.write(maze[(x, y)])
            f.write("\n")

def retrieve_maze(maze_name):
    """
    Function to retrieve a maze ranging from maze1 - maze50 as a list of strings.
    Params:
        maze_name : String name of maze you want to retrive. Should be in format 'maze1', 'maze2', ... 'maze50'
    """
    MAZE_FOL = create_maze_path()
    MAZE_TXT = maze_name + '.txt'
    MAZE_FILE = MAZE_FOL / MAZE_TXT
    with open(MAZE_FILE, "r") as f:
        maze = f.read().splitlines()
    return maze

def find_cell(maze, target):
    """
    Function to find either end or start point in maze
    Params:
        maze : list object that contains the specific maze you want to use
        target : string object (either 'S' for start or '!' for end) to find in maze
    """
    for i in range(0,len(maze)):
        for j in range(0,len(maze)):
            if maze[i][j] == target:
                return (i,j)