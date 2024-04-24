import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import json

# Read in the file
with open('./results/exact_count_data.json', 'r') as f:
    data = json.load(f)

# Convert the list of dicts to a DataFrame
df = pd.DataFrame(data)
df = df[['num_row', 'num_val', 'dim', 'num_col', 'correct_averages']]
#print(df.to_string())

# Create a new column labeled "num_row_num_val"
df['num_row_num_val'] = df['num_row'].astype(str) + ':' + df['num_val'].astype(str)

# Create a color and marker map for the 'dim' column
dim_styles = {dim_value: (color, marker) for dim_value, color, marker in zip(df['dim'].unique(), sns.color_palette(n_colors=df['dim'].nunique()), ['v', '^', 'o'])}
df['color'] = df['dim'].map(lambda x: dim_styles[x][0])
df['marker'] = df['dim'].map(lambda x: dim_styles[x][1])

# Create a size map for the 'num_col' column
size_map = {20: 10, 40: 23, 80: 36, 160: 50}
df['size'] = df['num_col'].map(size_map)
#print(df[['color', 'marker', 'size','dim']].to_string())

# Specify the order of x-axis values
order = ['10:8', '10:4', '10:2', '20:8', '20:4', '20:2', '40:8', '40:4', '40:2']
df['num_row_num_val'] = pd.Categorical(df['num_row_num_val'], categories=order, ordered=True)

# Sort the DataFrame by 'num_row_num_val'
df = df.sort_values('num_row_num_val')

# Create the scatter plot
plt.figure(figsize=(8, 4))
#for (color, marker, size), dim in zip(df[['color', 'marker', 'size']].drop_duplicates().values, sorted(df['dim'].unique())):
#print(df[['color', 'marker', 'size']].drop_duplicates().values)
#for (color, marker, size) in zip(df[['color', 'marker', 'size']].drop_duplicates().values):
for thing in zip(df[['color', 'marker', 'size']].drop_duplicates().values):
    color = thing[0][0]
    marker = thing[0][1]
    size = thing[0][2]
    #print(color, marker, size)
    #df_dim = df[df['dim'] == dim]
    df_dim = df[(df['color'] == color) & (df['marker'] == marker) & (df['size'] == size)]
    #print(df_dim.to_string())
    plt.scatter(df_dim['num_row_num_val'], df_dim['correct_averages'], color=color, marker=marker, alpha=0.8, s=size)

plt.xlabel('Aggregate True Count : Number of Other Column Values', fontsize=12)
plt.ylabel('Precision', fontsize=12)

# Set the range of the y-axis
plt.ylim([-0.05, 1.05])  # Replace with your desired range

# Add horizontal grid lines
plt.grid(axis='y')

# Create legend
legend_elements = [mlines.Line2D([0], [0], color=dim_styles[dim][0], marker=dim_styles[dim][1], linestyle='None', markersize=5, label=f"Attack dims {dim}") for dim in dim_styles]
legend1 = plt.legend(handles=legend_elements, fontsize='small', loc='lower left', bbox_to_anchor=(0, 0.75))
plt.legend([mlines.Line2D([0], [0], color='black', marker='o', markersize=size/10, linestyle='None') for num_col, size in size_map.items()], ['Columns: {}'.format(num_col) for num_col in size_map.keys()], title='', loc='lower left', bbox_to_anchor=(0, 0.5), fontsize='small')
plt.gca().add_artist(legend1)

# Save the plot
plt.savefig('./results/exact_count_prec_agg.png', bbox_inches='tight', dpi=300)	

plt.close()