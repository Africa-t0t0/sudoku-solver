import os
import random

import numpy as np
import pandas as pd

from sudoku_solver import utils


class Sudoku(object):
    _table_df = None
    _full_table_dimensions = None
    _sub_table_dimensions = None
    _expected_ls = None

    def __init__(self, table: list | str | pd.DataFrame):
        self._table_df = self.__handle_table_type(table=table)
        self._full_table_dimensions = len(table)
        self._expected_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def _set_table_df(self, table_df: pd.DataFrame) -> None:
        self._table_df = table_df

    @staticmethod
    def __handle_table_type(table: list | str) -> pd.DataFrame:
        if isinstance(table, str):
            table_df = utils.read_table_from_csv(path=table)
        elif isinstance(table, list):
            table_df = utils.convert_list_to_dataframe(ls=table)
        elif isinstance(table, pd.DataFrame):
            table_df = table
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

        final_matrix_df = np.zeros((9, 9), dtype=int)

        for sub_idx, sub_table_df in enumerate(sub_matrices_np_ls):
            row_start = (sub_idx // 3) * 3
            col_start = (sub_idx % 3) * 3
            final_matrix_df[row_start:row_start + 3, col_start:col_start + 3] = sub_table_df

        return final_matrix_df

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

    def _check_tables_rows_columns(self, full_table_df: pd.DataFrame) -> dict:
        invalid_dd = dict()
        large = len(full_table_df)
        for i in range(0, large):
            for j in range(0, large):
                valid_x, valid_y = self._check_full_line(position_x=i, position_y=j)
                if not valid_x or not valid_y:
                    invalid_dd[(i, j)] = (valid_x, valid_y)
        return invalid_dd

    @staticmethod
    def _clean_available_solutions(available_solutions_ls: list,
                                   empty_cells_ls: list,
                                   solutions_dd: dict) -> list:
        current_available_solutions_ls = available_solutions_ls.copy()
        large_solutions = len(solutions_dd.keys())
        for i in range(0, large_solutions):
            position_x, position_y = empty_cells_ls[i]
            previous_solution = solutions_dd[(position_x, position_y)]
            current_available_solutions_ls.pop(previous_solution)
        return current_available_solutions_ls

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

    def _find_empty_cells_brute(self):
        for i in range(9):
            for j in range(9):
                if self._table_df.iloc[i, j] == 0:
                    return i, j
        return None

    def print_table_realtime(self) -> None:
        os.system('clear' if os.name == 'posix' else 'cls')  # Limpiar la pantalla
        print(self._table_df.to_string(index=False, header=False), flush=True)

    @staticmethod
    def _clean_sub_table_df(sub_table_df: pd.DataFrame, prev_solutions_dd: dict) -> pd.DataFrame:
        large = len(sub_table_df)
        for i in range(0, large):
            for j in range(0, large):
                if (i, j) in prev_solutions_dd.keys():
                    sub_table_df.loc[i, j] = 0
        return sub_table_df

    def _fill_empty_cells(self,
                          empty_cells_ls: list,
                          available_solutions_ls: list,
                          sub_table_df: pd.DataFrame,
                          prev_solutions_dd: dict = None) -> (pd.DataFrame, dict):
        # TODO: this method is pending
        solutions_dd = dict()
        empty_cells_idx = 0
        solutions_idx = 0
        backtracking = False
        large_empty_cells = len(empty_cells_ls)
        large_available_solutions = len(available_solutions_ls)
        if prev_solutions_dd:
            sub_table_df = self._clean_sub_table_df(sub_table_df=sub_table_df, prev_solutions_dd=prev_solutions_dd)

        while empty_cells_idx < large_empty_cells:
            while solutions_idx < large_available_solutions:
                if backtracking:
                    current_available_solutions_ls = self._clean_available_solutions(
                        available_solutions_ls=available_solutions_ls,
                        empty_cells_ls=empty_cells_ls,
                        solutions_dd=solutions_dd
                    )
                else:
                    current_available_solutions_ls = available_solutions_ls.copy()
                effective_solutions_ls = [value for value in solutions_dd.values()]
                complement_ls = utils.list_operator(ls=[current_available_solutions_ls, effective_solutions_ls],
                                                    operation="complement")
                current_available_solutions_ls = complement_ls
                solution = current_available_solutions_ls[solutions_idx]
                position_x, position_y = empty_cells_ls[empty_cells_idx]
                aux_sub_table_df = sub_table_df.copy()
                if prev_solutions_dd:

                    if prev_solutions_dd[(position_x, position_y)] == solution:
                        filtered_ls = [item for item in current_available_solutions_ls if item != solution]
                        solution = random.choice(filtered_ls)

                aux_sub_table_df.loc[position_x, position_y] = solution

                if self._check_current_iteration_sub_table(sub_table_df=aux_sub_table_df):
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
        return sub_table_df, solutions_dd

    def _solve_sub_table(self, sub_table_df: pd.DataFrame, prev_solutions_dd: dict = None) -> (pd.DataFrame, dict):
        expected_ls = self._expected_ls
        if prev_solutions_dd:
            empty_cells_ls = [value for value in prev_solutions_dd.keys()]
            available_solutions_ls = [value for value in prev_solutions_dd.values()]
            sub_table_df = self._clean_sub_table_df(sub_table_df=sub_table_df, prev_solutions_dd=prev_solutions_dd)
        else:
            empty_cells_ls = self._find_empty_cells(table_df=sub_table_df)
            cleaned_sub_table_df = sub_table_df.astype(int)
            # ravel gives a "set" of items from the dataframe.
            effective_set = set(map(int, cleaned_sub_table_df.values.ravel()))
            effective_ls = list(effective_set)
            available_solutions_ls = utils.list_operator(ls=[expected_ls, effective_ls], operation="complement")

        solved_sub_table_df, solutions_dd = self._fill_empty_cells(empty_cells_ls=empty_cells_ls,
                                                                   available_solutions_ls=available_solutions_ls,
                                                                   sub_table_df=sub_table_df,
                                                                   prev_solutions_dd=prev_solutions_dd)

        return solved_sub_table_df, solutions_dd

    def _brute_force_valid(self, position_x: int, position_y: int, num: int) -> bool:
        if num in self._table_df.iloc[position_x, :].values:
            return False
        if num in self._table_df.iloc[:, position_y].values:
            return False

        sub_row, sub_col = 3 * (position_x // 3), 3 * (position_y // 3)
        subgrid = self._table_df.iloc[sub_row:sub_row + 3, sub_col:sub_col + 3].values
        if num in subgrid:
            return False
        return True

    def _basic_solution(self) -> bool:
        """
        Given an incomplete sudoku, we solve it by using brute force.
        :return:
        """
        empty_cell = self._find_empty_cells_brute()
        if not empty_cell:
            return True
        position_x, position_y = empty_cell
        for num in range(1, 10):
            if self._brute_force_valid(position_x=position_x, position_y=position_y, num=num):
                self._table_df.iloc[position_x, position_y] = num
                self.print_table_realtime()
                if self._basic_solution():
                    return True
                self._table_df.iloc[position_x, position_y] = 0
                self.print_table_realtime()
        return False

    def brute_force(self) -> bool:
        solution = self._basic_solution()
        return solution
