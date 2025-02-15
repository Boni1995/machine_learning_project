import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import referential dataset
ref_data = pd.read_excel("C:\\Users\\franc\\Documents\\GitHub\\machine_learning_project\\final_ref_data.xlsx")


# Define column type for future manipulation
col_types = []

for col in ref_data.columns:
    if ref_data[col].max() > 1:
        col_types.append([col, 'continuous'])
    else:
        col_types.append([col, 'binary'])

col_types = pd.DataFrame(col_types, columns=['attribute', 'type'])

#print(col_types)

# Check correlation with poisonous
corr = ref_data.corr()

sorted_df = corr['poisonous'].sort_values(ascending=False)
df_corr_poison = sorted_df.reset_index()

df_corr_poison.columns = ['attribute', 'correlation']
df_corr_poison = df_corr_poison.dropna(subset=['correlation'])

df_corr_poison['abs_corr'] = np.abs(df_corr_poison['correlation'])

df_corr_poison = df_corr_poison.sort_values(by='abs_corr', ascending=False)
df_corr_poison = df_corr_poison.reset_index()
df_corr_poison = df_corr_poison.drop(columns=['index'])

# Merge with type of attribute
df_corr_poison = df_corr_poison.merge(col_types, how = 'left')

# Exclude poisonous row, as it is the attribute to predict
df_corr_poison = df_corr_poison[df_corr_poison['attribute'] != 'poisonous']

# Save it as .xlsx for future usage
df_corr_poison.to_excel('corr_poisonous.xlsx', index=False)

'''
For the measurements, I define min and max for the tree by calculating the mean min and mean max of poisonous mushrooms.

For this, I check the distribution of each attribute (column), and in case of any significant outlier, I drop it just for determining the thresholds.
'''

# Filter just poisonous mushrooms
data_measure = ref_data[ref_data['poisonous'] == 1]

# Leave just the columns with measures
data_measure = data_measure[['cap-diameter-min', 'cap-diameter-max', 'stem-height-min', 'stem-height-max', 'stem-width-min', 'stem-width-max']]

# Check the distribution of each column, and with graphpad check any significant outlier (uses Grubb's test)
data_measure = data_measure[
    (data_measure['cap-diameter-max'] < 50) &
    (data_measure['stem-height-min'] < 15) &
    (data_measure['stem-height-max'] < 35) &
    (data_measure['stem-width-max'] < 40)
]

print(data_measure)


# Use boxplot to see if significant outliers were removed
#fig, ax = plt.subplots()

#ax.boxplot(data_measure)

#ax.set_xticklabels(['cap-diameter-min', 'cap-diameter-max', 'stem-height-min', 'stem-height-max', 'stem-width-min', 'stem-width-max'])

#plt.show()

# Plot histograms from matplotlib to check if now columns have a normal distribution so I can use the mean as threshold in the tree predictor
#attributes = ['cap-diameter-min', 'cap-diameter-max', 'stem-height-min', 'stem-height-max', 'stem-width-min', 'stem-width-max']

#for col in attributes:

#    fig, ax = plt.subplots()

#    ax.hist(data_measure[col], bins=8)
#    ax.set_title(col)

#    plt.show()


# cap-diameter-min: asymmetric -> median
# cap-diameter-max: asymmetric -> median
# stem-height-min: symmetric -> mean
# stem-height-max: asymmetric -> median
# stem-width-min: asymmetric and multimodal -> median
# stem-width-max: asymmetric -> median

print(
    f"cap-diameter-min: {data_measure['cap-diameter-min'].median()}\n"
    f"cap-diameter-max: {data_measure['cap-diameter-max'].median()}\n"
    f"stem-height-min: {round(data_measure['stem-height-min'].mean())}\n"
    f"stem-height-max: {data_measure['stem-height-max'].median()}\n"
    f"stem-width-min: {data_measure['stem-width-min'].median()}\n"
    f"stem-width-max: {data_measure['stem-width-max'].median()}"    
    )