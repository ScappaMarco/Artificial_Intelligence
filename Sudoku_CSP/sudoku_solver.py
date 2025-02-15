import constraint

def solver(sudoku_problem):
    p = constraint.Problem(constraint.RecursiveBacktrackingSolver())
    
    possible_val = [i for i in range(1, 9 + 1)] # domain for each non-prefilled cell
    # define variables and domains
    for i in range(9):
        for j in range(9):
            if sudoku_problem[i][j] == 0: # non-prefilled variable
                p.addVariable((i,j),possible_val)
            else:
                p.addVariable((i,j),[sudoku_problem[i][j]]) # prefilled-variable

    #row constraints
    for i in range(9):
        row = [(i, j) for j in range(9)]
        p.addConstraint(constraint.AllDifferentConstraint(), row)

    #column constraints
    for i in range(9):
        column = [(j, i) for j in range(9)]
        p.addConstraint(constraint.AllDifferentConstraint(), column)
    
    # box constraints
    boxes = [((0,2),(0,2)), ((0,2),(3,5)), ((0,2),(6,8)), ((3,5),(0,2)), ((3,5),(3,5)), ((3,5),(6,8)), ((6,8),(0,2)), ((6,8),(3,5)), ((6,8),(6,8))]
    for box in boxes:
        box_cells = [(i,j) for i in range(box[0][0], box[0][1]+1) for j in range(box[1][0], box[1][1]+1)]
        p.addConstraint(constraint.AllDifferentConstraint(), box_cells)

    return p.getSolution()

def convert_to_list(s):
    t = {i: {} for i in range(9)}
    for k, v in s.items():
        t[k[0]][k[1]] = v

    result_table = []
    for k, l in t.items():
        result_table.append([l[i] for i in range(9)])
    return result_table

def sanity_check(solution):
    correct = tuple(i for i in range(1, 9 + 1))
    
    for row in solution:
        if tuple(sorted(row)) != correct:
            print(tuple(sorted(row)))
            return False

    for j in range(9):
        column = [solution[i][j] for i in range(9)]
        if tuple(sorted(column)) != correct:
            print(tuple(sorted(column)))
            return False

    boxes = [((0,2),(0,2)), ((0,2),(3,5)), ((0,2),(6,8)), ((3,5),(0,2)), ((3,5),(3,5)), ((3,5),(6,8)), ((6,8),(0,2)), ((6,8),(3,5)), ((6,8),(6,8))]
    for box in boxes:
        box_values = [solution[i][j] for i in range(box[0][0], box[0][1]+1) for j in range(box[1][0], box[1][1]+1)]
        if tuple(sorted(box_values)) != correct:
            return False

    return True

sudoku_problem = [
    [1, 0, 7, 2, 4, 6, 0, 9, 3],
    [2, 3, 0, 7, 8, 9, 5, 0, 1],
    [9, 4, 0, 0, 1, 3, 7, 0, 2],
    [3, 6, 9, 0, 7, 2, 1, 5, 0],
    [5, 7, 1, 0, 0, 8, 2, 3, 0],
    [8, 2, 4, 3, 0, 1, 9, 7, 6],
    [0, 1, 5, 6, 0, 0, 3, 8, 9],
    [0, 8, 3, 1, 9, 0, 4, 2, 7],
    [7, 0, 2, 8, 3, 0, 0, 1, 5]
]

solution = solver(sudoku_problem)
readable_solution = convert_to_list(solution)
for i in readable_solution:
    print(i)

sanity_check(readable_solution)