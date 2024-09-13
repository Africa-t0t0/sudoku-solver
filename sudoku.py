import numpy as np
import pandas as pd
import utils


class Sudoku(object):
    _table_df = None
    _full_table_dimensions = None
    _sub_table_dimensions = None
    _expected_ls = None

    def __init__(self, table: list | str):
        self._table_df = self.__handle_table_type(table=table)
        self._full_table_dimensions = len(table)
        self._expected_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    @staticmethod
    def __handle_table_type(table: list | str):
        if isinstance(table, str):
            table_df = utils.read_table_from_csv(path=table)
        elif isinstance(table, list):
            table_df = utils.convert_list_to_dataframe(ls=table)
        return table_df

    @staticmethod
    def print_tables(tables_ls) -> None:
        for idx, table in enumerate(tables_ls):
            print(f"table {idx}")
            print(table)

    def _divide_full_table(self) -> list:
        """
        returns a list of lists containing all the sub tables
        :return:
        """
        full_table_df = self._table_df
        sub_sectors_ls = list()
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                sub_section_df = full_table_df.iloc[i: i+3, j: j+3]
                sub_section_df.columns = [0, 1, 2]
                sub_sectors_ls.append(sub_section_df.reset_index(drop=True))
        return sub_sectors_ls

    @staticmethod
    def _merge_sub_tables(sub_table_ls: list) -> pd.DataFrame:

        sub_matrices_np_ls = [np.array(sub_table) for sub_table in sub_table_ls]

        # Crear una matriz vacía de 9x9
        final_matrix = np.zeros((9, 9), dtype=int)

        # Iterar sobre las submatrices
        for sub_idx, sub_matrix in enumerate(sub_matrices_np_ls):
            row_start = (sub_idx // 3) * 3
            col_start = (sub_idx % 3) * 3
            final_matrix[row_start:row_start + 3, col_start:col_start + 3] = sub_matrix

        # Mostrar la matriz final
        print(final_matrix)

    def _check_full_line(self, position_x: int, position_y: int) -> (bool, bool):
        full_table_df = self._table_df
        expected_ls = self._expected_ls
        valid_x = False
        valid_y = False
        horizontal_ls = full_table_df.iloc[position_x, :].to_list()
        vertical_ls = full_table_df.iloc[:, position_y].to_list()
        if set(horizontal_ls) == set(expected_ls):
            valid_x = True
        if set(vertical_ls) == set(expected_ls):
            valid_y = True
        return valid_x, valid_y

    def _check_sub_table(self, sub_table_df: pd.DataFrame) -> bool:
        """
        check if current square (3x3) is valid (no numbers are repeated).
        :param sub_table_df: 3x3 array
        :return: is valid or not
        """
        expected_ls = self._expected_ls
        effective_ls = list()
        large = 3
        for i in range(0, large):
            for j in range(0, large):
                effective_ls.append(sub_table_df.loc[i, j])
        if set(expected_ls) == set(effective_ls):
            is_valid = True
        else:
            is_valid = False
        return is_valid

    @staticmethod
    def _check_current_iteration_sub_table(sub_table_df: pd.DataFrame) -> bool:
        large = len(sub_table_df)
        current_ls = list()
        for i in range(0, large):
            for j in range(0, large):
                value = sub_table_df.loc[i, j]
                if value != 0:
                    if value not in current_ls:
                        current_ls.append(sub_table_df.loc[i, j])
                    else:
                        return False
        return True

    def _check_table_is_valid(self) -> list:
        """
        This method checks if the full table is valid.
        :return:
        """
        sub_tables_ls = self._divide_full_table()
        invalid_sub_tables_ls = list()
        for sub_table in sub_tables_ls:
            is_valid = self._check_sub_table(sub_table_df=sub_table)
            if not is_valid:
                invalid_sub_tables_ls.append(sub_table)
        if len(invalid_sub_tables_ls) > 0:
            print("invalid sub sectors!")
            self.print_tables(tables_ls=invalid_sub_tables_ls)
        else:
            print("sub sectors are ok!")

    @staticmethod
    def _find_empty_cells(table_df: pd.DataFrame) -> list:
        large = len(table_df)
        empty_positions_ls = list()
        for i in range(0, large):
            for j in range(0, large):
                if table_df.loc[i, j] == 0:
                    empty_positions_ls.append((i, j))
        return empty_positions_ls

    def _fill_empty_cells(self, empty_cells_ls: list, available_solutions_ls: list, sub_table_df: pd.DataFrame):

        solutions_dd = dict()
        empty_cells_idx = 0
        solutions_idx = 0
        backtracking = False

        while empty_cells_idx < len(empty_cells_ls):
            while solutions_idx < len(available_solutions_ls):
                if backtracking:
                    current_available_solutions_ls = available_solutions_ls.copy()
                    for i in range(0, empty_cells_idx - 1):
                        position_x, position_y = empty_cells_ls[i]
                        previous_solution = solutions_dd[(position_x, position_y)]
                        current_available_solutions_ls.pop(previous_solution)
                current_available_solutions_ls = available_solutions_ls.copy()
                effective_solutions_ls = [value for value in solutions_dd.values()]
                complement_ls = utils.list_operator(ls=[current_available_solutions_ls, effective_solutions_ls],
                                                    operation="complement")
                current_available_solutions_ls = complement_ls
                solution = current_available_solutions_ls[solutions_idx]
                position_x, position_y = empty_cells_ls[empty_cells_idx]
                aux_sub_table_df = sub_table_df.copy()
                aux_sub_table_df.loc[position_x, position_y] = solution

                if self._check_current_iteration_sub_table(sub_table_df=sub_table_df):
                    sub_table_df.loc[position_x, position_y] = solution
                    found_solution = True
                    solutions_dd[(position_x, position_y)] = solution
                    empty_cells_idx += 1
                    solutions_idx = 0
                    backtracking = False
                    break
                else:
                    found_solution = False
                    solutions_idx += 1
                    continue

            if not found_solution:
                if empty_cells_idx == 0:
                    raise Exception("invalid sudoku!")
                else:
                    backtracking = True
                    empty_cells_idx -= 1
                    break
        return sub_table_df

    def _solve_sub_table(self, sub_table_df: pd.DataFrame) -> pd.DataFrame:
        expected_ls = self._expected_ls
        large = len(sub_table_df)
        valid = False
        empty_cells_ls = self._find_empty_cells(table_df=sub_table_df)
        cleaned_sub_table_df = sub_table_df.astype(int)
        # ravel gives a "set" of items from the dataframe.
        effective_set = set(map(int, cleaned_sub_table_df.values.ravel()))
        effective_ls = list(effective_set)
        available_solutions_ls = utils.list_operator(ls=[expected_ls, effective_ls], operation="complement")

        solved_sub_table_df = self._fill_empty_cells(empty_cells_ls=empty_cells_ls,
                                                     available_solutions_ls=available_solutions_ls,
                                                     sub_table_df=sub_table_df)

        return solved_sub_table_df

    def _basic_solution(self) -> pd.DataFrame:
        """
        Given an incomplete sudoku, we solve it by using brute force.
        :return:
        """
        sub_table_ls = self._divide_full_table()
        solution_sub_tables_ls = list()
        for sub_table in sub_table_ls:
            solved_sub_table_df = self._solve_sub_table(sub_table)
            solution_sub_tables_ls.append(solved_sub_table_df)
        self._merge_sub_tables(solution_sub_tables_ls)

