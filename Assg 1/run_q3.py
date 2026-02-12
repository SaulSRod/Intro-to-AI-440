import maze_gen as mg
import a_search as search
import time
import os

def run_experiments():
    # Ensure mazes exist or generate them
    # The assignment likely asks to generate 50 gridworlds
    # We will generate them and save them as maze1..maze50 if they don't exist
    
    num_mazes = 50
    results = []

    print(f"Running experiments on {num_mazes} mazes...")
    print(f"{'Maze':<10} | {'Fwd Exp':<10} | {'Bwd Exp':<10} | {'Fwd Time':<10} | {'Bwd Time':<10} | {'Status'}")
    print("-" * 75)

    for i in range(1, num_mazes + 1):
        maze_name = f"maze{i}"
        
        # Check if maze exists, if not create it
        try:
            maze_list = mg.retrieve_maze(maze_name)
            # CAUTION: retrieve_maze returns a list of strings
            # a_search expects a 2D list/array or handles list of strings? 
            # let's check a_search.py knowledge_map: 
            # map = [[' ' for _ in range(len(true_map))] for _ in range(len(true_map))]
            # valid if true_map is list of strings too.
        except FileNotFoundError:
            print(f"Generating {maze_name}...")
            maze_dict = mg.create_maze(print=False, SEED=i)
            mg.save_maze(maze_dict, f"{maze_name}.txt")
            # retrieve again to get the list format expected by a_search if it relies on that
            maze_list = mg.retrieve_maze(maze_name)

        # Convert list of strings to list of lists if needed, or dict?
        # a_search.py functions mostly assume list of lists or similar for indexing [x][y]
        # BUT strings are not mutable. a_search definitely mutates `known_map`.
        # `true_map` is only read. `true_map[x][y]` works for list of strings.
        # So passing list of strings as `true_map` is fine.
        
        start = mg.find_cell(maze_list, search.START)
        goal = mg.find_cell(maze_list, search.GOAL)
        
        # Run Repeated Forward A*
        t0 = time.time()
        # sign=1 means break ties towards smaller g (default/favored usually? instruction says larger g)
        # Instruction: "break ties among cells with the same f-value in favor of cells with larger g-values"
        # Larger g-value = Closer to goal (usually).
        # In the heap, we pop smallest. 
        # priority = C * f - g  => larger g gives smaller priority => popped first.
        # So we want `sign * g` to be negative for larger g. 
        # So sign should be -1.
        # Let's verify a_search.py logic:
        # priority = CONSTANT * f_cost + (sign * g_value)
        # if sign is -1: larger g => smaller priority => popped earlier. CORRECT.
        
        fwd_path, fwd_exp = search.repeated_a_star(start, goal, maze_list, -1)
        t1 = time.time()
        fwd_time = t1 - t0

        # Run Repeated Backward A*
        t0 = time.time()
        bwd_path, bwd_exp = search.repeated_backward_a_star(start, goal, maze_list, -1)
        t1 = time.time()
        bwd_time = t1 - t0
        
        status = "OK"
        if isinstance(fwd_path, str) or isinstance(bwd_path, str):
            status = "Fail"
        elif len(fwd_path) != len(bwd_path):
             # Path lengths might differ if one finds optimal and other doesn't? 
             # Repeated A* on static grid should find optimal if A* is optimal?
             # Actually Repeated A* with unobserved obstacles is not guaranteed optimal path *traversal* cost,
             # but the final path on the map... wait.
             # The implementations return the path taken.
             # They might differ.
             if abs(len(fwd_path) - len(bwd_path)) > 0:
                 status = "DiffLen"

        results.append({
            "Maze": maze_name,
            "Fwd_Exp": fwd_exp,
            "Bwd_Exp": bwd_exp,
            "Fwd_Time": fwd_time,
            "Bwd_Time": bwd_time
        })

        print(f"{maze_name:<10} | {fwd_exp:<10} | {bwd_exp:<10} | {fwd_time:<10.4f} | {bwd_time:<10.4f} | {status}")

    # Calculate averages
    avg_fwd = sum(r["Fwd_Exp"] for r in results) / num_mazes
    avg_bwd = sum(r["Bwd_Exp"] for r in results) / num_mazes
    
    print("-" * 75)
    print(f"{'Average':<10} | {avg_fwd:<10.1f} | {avg_bwd:<10.1f} | {'-':<10} | {'-':<10} |")

if __name__ == "__main__":
    run_experiments()
