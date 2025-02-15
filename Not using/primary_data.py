import numpy as np
import pandas as pd
from data import Replace
from columns_function import Functions


# Import primary_data dataset
primary_data = pd.read_csv("C:\\Users\\franc\\Documents\\GitHub\\machine_learning_project\\Datasets\\primary_data.csv", sep=';')

# Copy the original dataset to avoid modifying the original one, and drop the first 2 columns
reference_data = primary_data.drop(columns = ['family', 'name'])

# Correct "Cap-surface"
reference_data = reference_data.rename(columns = {'Cap-surface':'cap-surface', 'Spore-print-color':'spore-print-color'})

# Check columns' info
print(reference_data.info())

#print(reference_data.head(5))
#secondary_data = pd.read_csv("C:\\Users\\franc\\Documents\\GitHub\\machine_learning_project\\Datasets\\secondary_data.csv", sep=';')


'''
As there are attributes with empty records, I will replace them, to avoid future manipulation problems.

This empty records are taken as 'Nan', so it outputs an error when trying to iterate.

This is why I'll replace them with a string value ('None').
'''

# Use applymap to replace '-' in any column with empty records
reference_data = reference_data.applymap(Replace())

# Check again the not-null
print(reference_data.info())


'''
class column:
    
    - Renamed to "poisonous"
    
    - give binary value: p=1 and e=0
'''
# rename column
reference_data = reference_data.rename(columns = {'class':'poisonous'})

# Define p=1 and e=0
reference_data['poisonous'] = np.where(reference_data['poisonous'] == 'p', 1, 0)


'''
Numerical attributes (cap-diameter, stem-height, stem-width): Split them into 2 columns each (min and max).
'''
num_columns = ['cap-diameter', 'stem-height', 'stem-width'] # Columns which have ranges of numbers that I'll split into min and max

# Split all the columns into min and max each
reference_data = Functions.min_max_split(reference_data, num_columns)

# Those max values with "-" will be replaced with the min value
reference_data['cap-diameter-max'] = reference_data.apply(lambda row: row['cap-diameter-min'] if row['cap-diameter-max'] == '-' else row['cap-diameter-max'], axis=1)
reference_data['stem-height-max'] = reference_data.apply(lambda row: row['stem-height-min'] if row['stem-height-max'] == '-' else row['stem-height-max'], axis=1)
reference_data['stem-width-max'] = reference_data.apply(lambda row: row['stem-width-min'] if row['stem-width-max'] == '-' else row['stem-width-max'], axis=1)


'''
cap-shape column:
- Set all possible values
- Iterate over each row to check wether it has or not the given shapes
- Through one-hot encoding I put 1 for the shapes that each row contains and 0 if not
'''

# List of attributes and possible values
attribute_value = {
    'cap-shape': ['b', 'c', 'x', 'f', 's', 'p', 'o'],
    'cap-surface': ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e'],
    'cap-color': ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'],
    'gill-attachment': ['a', 'x', 'd', 'e', 's', 'p', 'f', '?'],
    'gill-spacing': ['c', 'd', 'f'],
    'gill-color': ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    'stem-root': ['b', 's', 'c', 'u', 'e', 'z', 'r'],
    'stem-surface': ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e', 'f'],
    'stem-color': ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    'veil-type': ['p', 'u'],
    'veil-color': ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    'ring-type': ['c', 'e', 'r', 'g', 'l', 'p', 's', 'z', 'y', 'm', 'f', '?'],
    'spore-print-color': ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'],
    'habitat': ['g', 'l', 'm', 'p', 'h', 'u', 'w', 'd'],
    'season': ['s', 'u', 'a', 'w']
}

# One-hot encoding
reference_data = Functions.one_hot_encoding(reference_data, attribute_value)

'''
does-bruise-or-bleed column: give binary value (t=1 and f=0)
'''
# Define t=1 and f=0
reference_data['does-bruise-or-bleed'] = np.where(reference_data['does-bruise-or-bleed'] == '[t]', 1, 0)

'''
has-ring column: give binary value (t=1 and f=0)
'''
# Define t=1 and f=0
reference_data['has-ring'] = np.where(reference_data['has-ring'] == '[t]', 1, 0)


# Final check of the dataset modified
print(reference_data.head(5))

# Save it as .xlsx for future usage
reference_data.to_excel('final_ref_data.xlsx', index=False)