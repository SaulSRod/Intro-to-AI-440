"""
gen_test_json.py — Generate N random 101x101 mazes and save as mazes.json. Uses same algorithm as maze_generator.py.

Usage:
    python gen_test_json.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
import random
from constants import START_NODE, END_NODE, ROWS
from tqdm import tqdm
import argparse

# set random seed for reproducibility
random.seed(42)

def convert_to_list(maze, param):
    """Converts to maze"""
    grid = []

    for x in range(param):
        row = []
        for y in range(param):
            #Display the wall or empty space:
            row.append(maze[(x,y)])
        #print(f"Appended row:{row}")
        grid.append(row)
    
    #Make sure start and end are empty
    grid[0][0] = 0
    grid[param - 1][param - 1] = 0
    return grid

def printMaze(maze, param):
    """Displays the maze"""

    for x in range(param):
        for y in range(param):
            #Display the wall or empty space:
            print(maze[x][y], end='')
        print() 

def visit_iterative(hasVisited, maze, startX, startY, WIDTH, HEIGHT):
    #Blank cells mean walkable paths
    EMPTY = 0

    #create stack
    stack = [(startX, startY)]
    hasVisited.add((startX, startY))
    maze[(startX, startY)] = EMPTY

    while stack:
        #Look at top of stack
        x, y = stack[-1]  

        unvisitedNeighbors = []

        if y > 0 and (x, y - 1) not in hasVisited:
            unvisitedNeighbors.append((x, y-1))
        if y < HEIGHT - 1 and (x, y + 1) not in hasVisited:
            unvisitedNeighbors.append((x, y + 1))
        if x > 0 and (x - 1, y) not in hasVisited:
            unvisitedNeighbors.append((x - 1, y))
        if x < WIDTH - 1 and (x + 1, y) not in hasVisited:
            unvisitedNeighbors.append((x + 1, y))

        if not unvisitedNeighbors:
            #Dead end reached so backtrack
            stack.pop()
            continue

        #Pick a random direction
        next_x, next_y = random.choice(unvisitedNeighbors)

        carve_chance = 0.70
        if random.random() > carve_chance:
            #Keep the cell as a wall 
            hasVisited.add((next_x, next_y))
            continue
        else:
            #Update cell to be empty thus walkable
            maze[(next_x, next_y)] = EMPTY
            hasVisited.add((next_x, next_y))
            stack.append((next_x, next_y))

def create_maze() -> list:
    #TODO: Implement this function to generate and return a random maze as a 2D list of 0s and 1s.
    #Keep track of visited cells
    hasVisited = set() 
    EMPTY = 0
    WALLS = 1

    # Create the filled in maze data structure to start:
    maze = {}
    for x in range(ROWS):
        for y in range(ROWS):
            #Every space is a wall at first
            maze[(x, y)] = WALLS
    
    #Manually set points to be empty
    maze[(0,0)] = EMPTY
    maze[(100,100)] = EMPTY

    # Carve out the paths in the maze data structure:
    start_x, start_y = START_NODE
    visit_iterative(hasVisited, maze, start_x, start_y, ROWS, ROWS)

    #Convert to 2d list and return
    #printMaze(test, ROWS)
    return convert_to_list(maze, ROWS)
    
def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)
    
    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):  
        mazes.append(create_maze())

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)
    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")

if __name__ == "__main__":
    main()
