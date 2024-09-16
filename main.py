from sudoku import Sudoku

path = "csv/sudoku_sencillo.csv"


sudoku_obj = Sudoku(table=path)

if sudoku_obj.brute_force():
    print("solution!")
else:
    print("no solution")
