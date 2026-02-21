"""
q5.py — Adaptive A* with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G Adaptive A* on the current maze
- 2 : run MIN-G Adaptive A* on the current maze
- ESC or close window : quit

Maze file format helper:
- readFile(fname) reads 0/1 space-separated tokens, 1=blocked, 0=free, one row per line.

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
from q2 import knowledge_map, update_known_map, reconstructed_path, heuristics, get_neighbors, cell_is_blocked, repeated_forward_astar, BLOCKED, UNBLOCKED
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
#from custom_pq import CustomPQ_maxG


# ---------------- FILE LOADER ----------------
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
    return mazes

def adaptive_a_start_search(start_cell, goal_cell, known_map, h_tracker, sign):
    """
    Function that runs adaptive a* search from given cell to goal cell using both the world knowledge and heuristics knowledge of cells the agent has seen.
    Params:
        start_cell : Starting cell in the form of a tuple (x,y)
        goal_cell : End cell in a tuple (x,y) that has '!'
        known_map : Map that has information on the cells the agent has seen while exploring. Any unseen cells are assumed to be empty/walkable
        h_tracker : Dict that has adaptive heuristic values for previously expanded states. If a cell has an entry the heuristic value is used instead of Manhattan
        sign :
            -1 will break ties on largest g  
             1 will break ties on smallest g
    """
    #min heap to track cells we can reach
    open_list = []
    #Visited list to track which cells have already been visited
    closed_list = dict()
    #sample entry for closed_list
    if start_cell in h_tracker:
        h_value = h_tracker[start_cell]
    else:
        h_value = heuristics(start_cell, goal_cell)
    g_value = 0
    f_cost = h_value + g_value
    
    #initialize heap with start cell
    CONSTANT = len(known_map) * len(known_map[0]) + 1
    priority = CONSTANT * f_cost + (sign * g_value)
    heapq.heappush(open_list, (priority, start_cell))

    #keep track of g_value starting with start cell
    g_tracker = dict()
    g_tracker[start_cell] = g_value

    #Keep track of parents
    parent = dict()
    parent[start_cell] = None

    #Track expansions
    expansions = 0

    while open_list:
        #Pop lowest f_cost in heap
        state = heapq.heappop(open_list)
        current_cell = state[1]
        #print(f"current cell: {current_cell}")
        #print(state)

        #Check to see if we reached goal
        if state[1] == goal_cell:
            #print("Found potential goal, reconstructing path")
            return reconstructed_path(parent, start_cell, goal_cell), expansions, closed_list, g_tracker[current_cell]

        #Update closed list as needed
        if current_cell in closed_list:
            continue

        #Add node to seen
        closed_list[current_cell] = g_tracker[current_cell]
        #print(f"Expanding {state}")
        expansions += 1

        #get neighbors
        potential_neighbors = get_neighbors(known_map, state[1])

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
                if neighbor in h_tracker:
                    h_value = h_tracker[neighbor]
                else:
                    h_value = heuristics(neighbor, goal_cell)
                    
                f_cost = h_value + g_tracker[neighbor]

                #Push neighbor into heap but give priority according to the sign
                CONSTANT = len(known_map) * len(known_map[0]) + 1
                priority = CONSTANT * f_cost + (sign * g_tracker[neighbor])
                heapq.heappush(open_list, (priority, neighbor))
             
    return False, expansions, closed_list, g_tracker[current_cell]

def adaptive_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # TODO: Implement Adaptive A* with max_g tie-braking strategy.
    # Use heapq for standard priority queue implementation and name your max_g heap class as `CustomPQ_maxG` and use it. 
    # break ties by g_max
    sign = -1

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

    #Create a heuristic value tracker
    heuristic_tracker = dict()

    #Loop to find path or return maze as unsolvable
    while current_state != goal:
        #Initiate A* search using known knowledge
        potential_path, current_exp_count, closed_list, g_cost_path = adaptive_a_start_search(current_state, goal, known_map, heuristic_tracker, sign)

        #Update g val of each cell in closed list
        for cell, g_val in closed_list.items():
            #Heuristic update
            h_new = g_cost_path - g_val
            #print(f"cell: {cell}, h_new: {h_new}, g val: {g_val}, path g cost:{g_cost_path}")
            heuristic_tracker[cell] = max(heuristic_tracker.get(cell, 0), h_new)

        #Update expansion counter
        total_expansions += current_exp_count
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
    
    #print(f"Total expansions performed: {expansion_count} with {sign}")
    return True, finalized_path, total_expansions, replans

def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    # This function should display the maze used, the agent's knowledge, and the search process as the agent plans and executes.
    # As a reference, this function takes pygame Surface 'win' to draw on, the actual maze grid, the algorithm name for labeling, 
    # and optional parameters for controlling the visualization speed and saving a screenshot.
    # You are free to use other visualization libraries other than pygame. 
    # You can call repeated_backward_astar with visualize_callbacks that update the Pygame display as the agent plans and executes.
    # In the end it should store the visualization as a PNG file if save_path is provided, or default to "vis_{algo}.png".
    # print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    if save_path is None:
        save_path = f"vis_{algo}.png"

    # If 'win' is the display surface (it is), this works:
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q5: Adaptive A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q5.json",
                        help="Path to output JSON results file")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q5-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        t0 = time.perf_counter()
        found, executed, expanded, replans = adaptive_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
        )
        t1 = time.perf_counter()

        entry["adaptive"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_forward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
        )
        t1 = time.perf_counter()

        entry["fwd"] = {
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
        pygame.display.set_caption("Adaptive A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "adaptive"
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
                        current_algo = "adaptive"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "adaptive"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "fwd"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()