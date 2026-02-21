"""
q2.py — Repeated Forward A* (Forward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- L : load maze from text file (see readFile format) and run MAX-G
- ESC or close window : quit

Maze file format (readFile):
- Space-separated 0/1 values, 1 = blocked, 0 = free, one row per line.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path
YELLOW = start + agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
import time
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
UNBLOCKED = 0
BLOCKED = 1
#from custom_pq import CustomPQ_maxG, CustomPQ_minG


def heuristics(a: tuple, b: tuple) -> float:
    #Assign x and y cords
    x1, y1 = a
    x2, y2 = b
    #Calculate manhattan distance
    distance = abs(x1 - x2) + abs(y1 - y2)
    #print(distance)
    return distance

def knowledge_map(true_map):
    """
    Intializes a map of seen ceels for agent to keep track of blocked/unblocked, with each cell initially marked as unblocked.
    """
    map = [[UNBLOCKED for _ in range(0,ROWS)] for _ in range(0,ROWS)]
    return map

def get_neighbors(map, cell: tuple) -> list:
    """
    Get all neighbors of a given cell
    """
    NORTH, SOUTH, EAST, WEST = +1, -1, +1, -1

    c_x, c_y = cell
    possible = []

    if c_x + NORTH in range(0, len(map)) and c_y in range(0,len(map[c_x + NORTH])):
        if map[c_x + NORTH][c_y] == UNBLOCKED:
            possible.append((c_x + NORTH, c_y))

    if c_x + SOUTH in range(0, len(map)) and c_y in range(0,len(map[c_x + SOUTH])):
        if map[c_x + SOUTH][c_y] == UNBLOCKED:
            possible.append((c_x + SOUTH, c_y))

    if c_x in range(0, len(map)) and c_y + EAST in range(0,len(map[c_x])):
        if map[c_x][c_y + EAST] == UNBLOCKED:
            possible.append((c_x, c_y + EAST))

    if c_x in range(0, len(map)) and c_y + WEST in range(0,len(map[c_x])):
        if map[c_x][c_y + WEST] == UNBLOCKED:
            possible.append((c_x, c_y + WEST))

    #Debug to see why map ranges are out of bounds
    #print(f"map bounds: x -> {len(map)} y -> {len(map[0])}")
    return possible

def cell_is_blocked(map, cell):
    """
    Check to see if cell is a wall or open
    """
    x, y = cell
    if map[x][y] == BLOCKED:
        return True
    return False

def reconstructed_path(parent, start, goal):
    """
    Reconstruct the potential path found by a_start_search
    """
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        if cell == start:
            break
        cell = parent.get(cell)

    return list(reversed(path))

def a_star_search(start_cell: tuple, goal_cell: tuple, known_map: list, sign = -1):
    """
    Function that runs forward A* search using agent's knowledge of previously seen cells. Returns the optimal path possible found using the agent's knowledge map.
    Implements a binary heap (min heap) to optimize path searching.
    Params:
        start_cell : Starting cell in the form of a tuple (x,y)
        end : Ending cell in the map in the form of a tuple (x,y) that contains the goal '!'
        known_map : Map that contains the knowledge of cells previously seen by agent. Any cells not seen before are assumed open/empty cells
        sign : Int value that is either -1 or +1 Default will break ties based on largest g value of a cell.
            
            -1 → break ties based on largest g  
            +1 → break ties based on smallest g
    """
    #min heap to track cells we can reach
    open_list = []
    #Visited list to track which cells have already been visited
    closed_list = set()
    #sample entry for closed_list
    """
    Node : (f_cost, g_value, h_value, current cell)
        manhattan, cost to reach this cell, total cost, parent cell of node being added to closed_list
    """
    h_value = heuristics(start_cell, goal_cell)
    g_value = 0
    f_cost = h_value + g_value
    
    #initialize heap with start cell
    CONSTANT = len(known_map) * len(known_map[0]) + 1
    priority = CONSTANT * f_cost + (sign * g_value)
    heapq.heappush(open_list, (priority, g_value, h_value, start_cell))

    #keep track of g_value starting with start cell
    g_tracker = dict()
    g_tracker[start_cell] = g_value

    #Keep track of parents
    parent = dict()
    parent[start_cell] = None

    expansions = 0

    while open_list:
        #Pop lowest f_cost in heap
        state = heapq.heappop(open_list)
        current_cell = state[3]
        #print(f"current cell: {current_cell}")
        #print(state)

        #Check to see if we reached goal
        if state[3] == goal_cell:
            #print("Found potential goal, reconstructing path")
            #print(parent)
            return reconstructed_path(parent, start_cell, goal_cell), expansions

        #Update closed list as needed
        if current_cell in closed_list:
            continue

        #Add node to seen
        closed_list.add(current_cell)
        expansions += 1

        #get neighbors
        potential_neighbors = get_neighbors(known_map, state[3])

        for neighbor in potential_neighbors:
            #print each neighbor for debug
            #print(f"possible neighbor:{neighbor}")

            #Skip cell if it is known to be a wall
            if cell_is_blocked(known_map, neighbor):
                continue
            
            #Get the potential cost to reach this cell
            potential_g = g_tracker[current_cell] + 1
            
            #Push neighbor into open/frontier if they haven't been seen or found cheaper cost
            if neighbor not in g_tracker or potential_g < g_tracker[neighbor]:
                #Update g cost
                g_tracker[neighbor] = potential_g
                #Update parent of neighbor
                parent[neighbor] = current_cell

                #Calulate values of neighbor
                h_value = heuristics(neighbor, goal_cell)
                f_cost = h_value + g_tracker[neighbor]

                #Push neighbor into heap 
                priority = CONSTANT * f_cost + (sign * g_tracker[neighbor])
                heapq.heappush(open_list, (priority, g_tracker[neighbor], h_value, neighbor))
             
    return False, expansions

def update_known_map(true_map: list, agent_map: list, current_cell: tuple):
    """
    Update neighbors the cells current position
    """
    all_possible = get_neighbors(agent_map, current_cell)
    truly_possible = get_neighbors(true_map, current_cell)

    #Get a list of all blocked cells in 4 cardinal directions
    blocked_cells = [cell for cell in all_possible if cell not in truly_possible]
    #print(f"all -> {all_possible}")
    #print(f"true -> {truly_possible}")
    #print(f"blocked -> {blocked_cells}")

    #Update agent map
    for neighbor in blocked_cells:
        agent_map[neighbor[0]][neighbor[1]] == BLOCKED
    
    return agent_map

def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    mazes: List[List[List[int]]] = []
    for idx, grid in enumerate(data):
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = [[int(v) for v in row] for row in grid]
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)
        #print("found maze")
    return mazes

def repeated_forward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    tie_breaking: str = "max_g", # "min_g"
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # TODO: Implement Repeated Forward A* with min_g & max_g tie-braking strategies.
    # Use heapq for standard priority queue implementation and name your max_g heap class as `CustomPQ_maxG` 
    # and min_g heap class as `CustomPQ_minG`. Place them inside `custom_pq.py` file (see import statement in line 41).
    # and use it. 

    """
    Finalized function that repeatedly runs forward A* search until agent finds the best path possible, or returns an unsolvable maze.
    After every forward A* search, agent will update its knowledge of cells seen as needed and keep exploring.
    Params:
        begin_state : Starting cell of agent in the form of a tuple (x,y) that initially contains the value 'S'
        end : Ending cell in the map in the form of a tuple (x,y) that contains the goal '!'
        true_map : Map that contains the knowledge of all cells and what values they hold. It will label cells as walls, empty, start, goal as appropriate
        sign : Int value that is either -1 or +1 Default will break ties based on largest g value of a cell.
            
            -1 → break ties based on largest g  
            +1 → break ties based on smallest g
    """
    sign = -1
    if tie_breaking not in ["min_g", "max_g", "both"]:
        raise ValueError("Invalid tie_breaking method. Must be 'min_g' OR 'max_g' OR 'both' - case sensitive!!")
    if tie_breaking == 'min_g':
        sign = 1
    
    #debug tie breaking
    #print(f"You used {tie_breaking}. Sign used: {sign}")

    #Create knowledge map where every cell is assumed reachable by agent
    known_map = knowledge_map(actual_maze)
    current_state = start
    #Initialize the finalized path
    finalized_path = [current_state]
    
    #Let agent glance at nearby cells (4 cardinal directions) and update map with knowledge
    known_map = update_known_map(actual_maze, known_map, current_state)
    #print(potential_neighbors)
    
    total_expansions = 0
    replans = 0

    #Loop to find path or return maze as unsolvable
    while current_state != goal:
        #Initiate A* search using known knowledge
        potential_path, expansions = a_star_search(current_state, goal, known_map, sign)
        total_expansions += expansions
        
        if potential_path is False:
            #print("No path is possible. Maze cannot be solved after all cells are exhausted.")
            return False, -1, total_expansions, replans
        
        #Follow each cell in the found path
        for i in range(0,len(potential_path)-1):
            next_cell = potential_path[i+1]
            #print(next_cell)
            #Check to see if cell is blocked on true map and reevaluate path as needed
            if actual_maze[next_cell[0]][next_cell[1]] == BLOCKED:
                #print(f"Found a wall at {next_cell}. Restarting at {current_state} with updated map")
                known_map[next_cell[0]][next_cell[1]] = BLOCKED
                replans += 1
                break
            #Move agent if cell is open and walkable
            else:
                current_state = next_cell
                finalized_path.append(current_state)
                #Update neighbors of current agent cell 
                known_map = update_known_map(actual_maze, known_map, current_state)
                
    return True, finalized_path, total_expansions, replans


def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    # This function should display the maze used, the agent's knowledge, and the search process as the agent plans and executes.
    # As a reference, this function takes pygame Surface 'win' to draw on, the actual maze grid, the algorithm name for labeling, 
    # and optional parameters for controlling the visualization speed and saving a screenshot.
    # You are free to use other visualization libraries other than pygame. 
    # You can call repeated_forward_astar with visualize_callbacks that update the Pygame display as the agent plans and executes.
    # In the end it should store the visualization as a PNG file if save_path is provided, or default to "vis_{algo}.png".
    # print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    if save_path is None:
        save_path = f"vis_{algo}.png"

    # If 'win' is the display surface (it is), this works:
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q2: Repeated Forward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q2.json",
                        help="Path to output JSON results file")
    parser.add_argument("--tie_braking", type=str, choices=["max_g", "min_g", "both"], default="both",
                        help="Tie-breaking variant to run (default: both)")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q2-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        if args.tie_braking in ("max_g", "both"):
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="max_g"
            )
            t1 = time.perf_counter()

            entry["max_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        if args.tie_braking in ("min_g", "both"):
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="min_g"
            )
            t1 = time.perf_counter()

            entry["min_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        results.append(entry)

    if args.show_vis:
        # In case, PyGame is used for visualization, this code initializes a window and runs the visualization for the selected maze and algorithm.
        # Feel free to modify this code if you use a different visualization library or approach.
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Repeated Forward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()