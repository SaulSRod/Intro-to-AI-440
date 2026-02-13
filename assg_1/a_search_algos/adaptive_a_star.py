import heapq
from a_search_algos.a_search import (
    heuristics, reconstructed_path, get_neighbors, cell_is_blocked, knowledge_map, update_known_map
)
WALL = '%'
EMPTY = ' '
START = 'S'
GOAL = '!'

"""
    Function that runs adaptive a* search from given cell to goal cell using the world knowledge and using heuristics if the agent has them.
    Params: 
        start_cell : Starting cell in the form of a tuple (x,y)
        goal_cell : End cell in a tuple (x,y) that has '!'
        known_map : 2d list that contains the map knowledge of all cells the agent has seen at runtime
        h_tracker : set used to keep track of heuristic values. Path search will either use heuristic value if cell was seen before or calculate manhattan
        sign :
            -1 → break ties on largest g  
             1 → break ties on smallest g
    """

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
    closed_list = set()
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
        closed_list.add((current_cell, g_tracker[current_cell]))
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
             
    return False


def adaptive_a_star(begin_state, end, true_map, sign):
    """
    Function that repeatedly runs adaptive forward a* search until goal is reached or all possible cells exhausted. After each path found, agent will update the heuristic value of cells that were expanded (in the closed list).
    Params:
        begin_state : Starting cell in the form of a tuple (x,y) that is 'S'
        end : End cell in a tuple (x,y) that has '!'
        true_map : True map of maze with all information provided
        sign :
            -1 will break ties on largest g  
             1 will break ties on smallest g
    """
    #Create knowledge map where every cell is assumed reachable by agent
    known_map = knowledge_map(true_map)
    current_state = begin_state
    #Initialize the finalized path
    finalized_path = [current_state]
    
    #Let agent glance at nearby cells (4 cardinal directions) and update map with knowledge
    known_map = update_known_map(true_map, known_map, current_state)
    #print(potential_neighbors)
    expansion_count = 0

    #Create a heuristic value tracker
    heuristic_tracker = dict()

    #Loop to find path or return maze as unsolvable
    while current_state != end:
        #Initiate A* search using known knowledge
        potential_path, current_exp_count, closed_list, g_cost_path = adaptive_a_start_search(current_state, end, known_map, heuristic_tracker, sign)

        #Update g val of each cell in closed list
        for cell, g_val in closed_list:
            #Heuristic update
            h_new = g_cost_path - g_val
            #print(f"cell: {cell}, h_new: {h_new}, g val: {g_val}, path g cost:{g_cost_path}")
            heuristic_tracker[cell] = h_new

        #Update expansion counter
        expansion_count += current_exp_count
        if potential_path is False:
            return("No path is possible. Maze cannot be solved after all cells are exhausted.")
        
        #Follow each cell in the found path
        for i in range(0,len(potential_path)-1):
            next_cell = potential_path[i+1]
            #print(next_cell)
            #Check to see if cell is blocked on true map and reevaluate path as needed
            if true_map[next_cell[0]][next_cell[1]] == WALL:
                #print(f"Found a wall at {next_cell}. Restarting at {current_state} with updated map")
                known_map[next_cell[0]][next_cell[1]] = WALL
                break
            #Move agent if cell is open and walkable
            else:
                current_state = next_cell
                finalized_path.append(current_state)
                #Update neighbors of current agent cell 
                known_map = update_known_map(true_map, known_map, current_state)
    
    #print(f"Total expansions performed: {expansion_count} with {sign}")
    return finalized_path, expansion_count