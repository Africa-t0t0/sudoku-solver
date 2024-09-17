import pandas as pd


def read_table_from_csv(path: str) -> pd.DataFrame:
    if not path:
        raise Exception("no path provided")

    table_df = pd.read_csv(path, header=None)

    if table_df.empty:
        raise Exception("table is empty, please check file")

    return table_df


def convert_list_to_dataframe(ls: list) -> pd.DataFrame:
    df = pd.DataFrame(ls)
    if df.empty:
        raise Exception("dataframe is empty!")

    return df


def list_operator(ls: list, operation: str) -> list:
    if operation == "inter":
        result_ls = [value for value in ls[0] if value in ls[1]]
    elif operation == "complement":
        result_ls = [value for value in ls[0] if value not in ls[1]]
    return result_ls
