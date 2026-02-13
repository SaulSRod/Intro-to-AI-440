import heapq

WALL = '%'
EMPTY = ' '
START = 'S'
GOAL = '!'

def heuristics(a: tuple, b: tuple) -> float:
    """
    Helper function to calculate heuristic (h) value of a cell using manhattan distance formula.
    Params:
        a : starting cell
        b : goal cell
    """
    #Assign x and y cords
    x1, y1 = a
    x2, y2 = b
    #Calculate manhattan distance
    distance = abs(x1 - x2) + abs(y1 - y2)
    #print(distance)
    return distance

def knowledge_map(true_map):
    """
    Creates a belief map for agent to keep track of what cells it has visited and if the cell is blocked/unblocked, with each cell initially marked as unblocked.
    Params:
        Index0 : ' ' if cell is empty, other wise '%' if wall. Cells not seen will always be assumed empty
    """
    map = [[' ' for _ in range(len(true_map))] for _ in range(len(true_map))]
    return map

def get_neighbors(map, cell: tuple) -> list:
    """
    Get all neighbors of a given cell
    """
    NORTH, SOUTH, EAST, WEST = +1, -1, +1, -1

    c_x, c_y = cell
    possible = []

    if c_x + NORTH in range(0, len(map)) and c_y in range(0,len(map[c_x + NORTH])):
        if map[c_x + NORTH][c_y] == EMPTY:
            possible.append((c_x + NORTH, c_y))

    if c_x + SOUTH in range(0, len(map)) and c_y in range(0,len(map[c_x + SOUTH])):
        if map[c_x + SOUTH][c_y] == EMPTY:
            possible.append((c_x + SOUTH, c_y))

    if c_x in range(0, len(map)) and c_y + EAST in range(0,len(map[c_x])):
        if map[c_x][c_y + EAST] == EMPTY:
            possible.append((c_x, c_y + EAST))

    if c_x in range(0, len(map)) and c_y + WEST in range(0,len(map[c_x])):
        if map[c_x][c_y + WEST] == EMPTY:
            possible.append((c_x, c_y + WEST))

    #Debug to see why map ranges are out of bounds
    #print(f"map bounds: x -> {len(map)} y -> {len(map[0])}")
    return possible

def cell_is_blocked(map, cell):
    """
    Check to see if cell is a wall or open
    """
    x, y = cell
    if map[x][y] == WALL:
        return True
    return False

def reconstructed_path(parent, start, goal):
    """
    Reconstruct the potential path found by a_start_search
    """
    path = []
    path.append(goal)
    cell = goal
    while parent[cell] != start:
        cell = parent[cell]
        path.append(cell)
        

    path.append(start)
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
        agent_map[neighbor[0]][neighbor[1]] == WALL
    
    return agent_map
    

def repeated_forward_a_star(begin_state: tuple, end: tuple, true_map: list, sign=-1):
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
    #Create knowledge map where every cell is assumed reachable by agent
    known_map = knowledge_map(true_map)
    current_state = begin_state
    #Initialize the finalized path
    finalized_path = [current_state]
    
    #Let agent glance at nearby cells (4 cardinal directions) and update map with knowledge
    known_map = update_known_map(true_map, known_map, current_state)
    #print(potential_neighbors)
    
    total_expansions = 0

    #Loop to find path or return maze as unsolvable
    while current_state != end:
        #Initiate A* search using known knowledge
        potential_path, expansions = a_star_search(current_state, end, known_map, sign)
        total_expansions += expansions
        
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
                
    return finalized_path, total_expansions

    """
    Function that repeatedly runs Backward A* search until goal is reached.
    Search direction: Goal -> Current State (in the agent's mind)
    Agent movement: Current State -> Goal (physically)
    """

def repeated_backward_a_star(begin_state: tuple, end: tuple, true_map: list, sign = -1):
    """
    Finalized function that repeatedly runs Backward A* search until until goal is reached.
    
    Search direction: Goal -> Current State (in the agent's mind)
    
    Agent movement: Current State -> Goal (physically)
    Params:
        begin_state : Starting cell of agent in the form of a tuple (x,y) that initially contains the value 'S'
        end : Ending cell in the map in the form of a tuple (x,y) that contains the goal '!'
        true_map : Map that contains the knowledge of all cells and what values they hold. It will label cells as walls, empty, start, goal as appropriate
        sign : Int value that is either -1 or +1 Default will break ties based on largest g value of a cell.
            
            -1 → break ties based on largest g  
            +1 → break ties based on smallest g
    """
    #Create knowledge map where every cell is assumed reachable by agent
    known_map = knowledge_map(true_map)
    current_state = begin_state
    #Initialize the finalized path
    finalized_path = [current_state]
    
    #Let agent glance at nearby cells (4 cardinal directions) and update map with knowledge
    known_map = update_known_map(true_map, known_map, current_state)
    
    total_expansions = 0

    #Loop to find path or return maze as unsolvable
    while current_state != end:
        # Initiate Backward A* search: From Goal TO Current State
        # Note: heuristics will be calculated distance(node, current_state) inside a_start_search
        potential_path, expansions = a_star_search(end, current_state, known_map, sign)
        total_expansions += expansions
        
        if potential_path is False:
            return("No path is possible. Maze cannot be solved.")
        
        # potential_path is [Goal, ..., Next, Current]
        # We need to traverse it from the end (Current) back to Goal.
        
        # The path returned is [StartNode, ..., GoalNode] of the *search*.
        # Search started at `end` (Goal of maze) and went to `current_state` (Start of search).
        # So path is [Goal, ..., NextStep, CurrentState].
        
        # We want to move from CurrentState -> NextStep.
        # Iterate backwards from the second to last element
        
        # indices: 0=Goal, 1=..., len-2=NextStep, len-1=CurrentState
        
        path_blocked = False
        for i in range(len(potential_path)-2, -1, -1):
            next_cell = potential_path[i]
            
            #Check to see if cell is blocked on true map
            if true_map[next_cell[0]][next_cell[1]] == WALL:
                known_map[next_cell[0]][next_cell[1]] = WALL
                path_blocked = True
                break
            else:
                current_state = next_cell
                finalized_path.append(current_state)
                known_map = update_known_map(true_map, known_map, current_state)
        
        if not path_blocked and current_state != end:
             # Should not happen if path was valid and we didn't break, unless we reached end
             pass
                
    return finalized_path, total_expansions

