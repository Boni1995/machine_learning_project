import pandas as pd
from data import Replace


class Functions:

    @staticmethod
    def min_max_split(data, column):
        # Iteration to split all the columns with ranges, into min and max each
        for col in column:

            # Delete "[]" from each column
            data[col] = data[col].str[1:-1]

            # Split columns into min and max
            data[[f'{col}-min', f'{col}-max']] = data[col].str.split(', ', expand=True)

            # Give float type
            data[f'{col}-min'] = data[f'{col}-min'].astype(float)
            data[f'{col}-max'] = data[f'{col}-max'].astype(float)

            # Drop the original columns
            data = data.drop(columns=[col])

        data = data.applymap(Replace())

        return data
    
    @staticmethod
    def one_hot_encoding(data, column):
        for col, possible_values in column.items():
            for value in possible_values:
                data[f'{col}-{value}'] = data[col].apply(lambda x: 1 if value in x else 0)
    
            data = data.drop(columns={col}) # Drop the original columns
        return data