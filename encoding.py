import pandas as pd

class Encoding:

    def _init_(self):
        self.categories = {}

    def one_hot_encoding(self, data, fit=False):
        if fit:
            self.categories = {col: data[col].unique() for col in data.columns if data[col].dtype == 'object'}

        transformed_df = pd.DataFrame()

        for col in data.columns:
            if col in self.categories:
                for category in self.categories[col]:
                    transformed_df[f"{col}_{category}"] = (data[col] == category).astype(int)
            else:
                transformed_df[col] = data[col]

        return transformed_df

    def target_encoding(self, target, fit=False):
        if fit:
            self.target_mapping = {category: index for index, category in enumerate(target.unique())}

        encoded_series = target.map(self.target_mapping)
        
        return encoded_series, self.target_mapping
