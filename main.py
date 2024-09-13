from sudoku import Sudoku
import utils


path = "csv/sudoku_sencillo.csv"

sudoku_obj = Sudoku(table=path)

print(sudoku_obj._basic_solution())