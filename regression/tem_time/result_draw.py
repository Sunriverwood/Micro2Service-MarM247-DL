import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the uploaded Excel file
file_path = 'predictions_results.xlsx'
data = pd.read_excel(file_path)
groups = data.groupby(['True Temperature', 'True Time'])

# Plot
plt.figure(figsize=(10, 15))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']  # Marker options
for (temp, time), group in groups:
    marker = markers[(hash((temp, time)) % len(markers))]  # Cycle through markers
    plt.scatter(group['Predicted Temperature'], group['Predicted Time'],
                label=f'True Temp: {temp}, True Time: {time}', alpha=0.7, marker=marker)

# Add plot details
plt.xlabel('Predicted Temperature')
plt.ylabel('Predicted Time')
plt.title('Temperature-Time Clusters with Unique Markers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#
# # Re-import necessary libraries and reload the data after the reset
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Reload the file
# file_path = 'predictions_results.xlsx'
# data = pd.read_excel(file_path)
#
# # Group data by True Temperature and True Time to adjust vertical spacing and plot
# groups = data.groupby(['True Temperature', 'True Time'])
#
# # Create a new figure
# plt.figure(figsize=(10, 15))
#
# # Define unique colors and markers for each group
# colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']
#
# # Plot each group with adjusted vertical spacing
# y_offset = 0  # Vertical offset
# y_spacing = 3000  # Adjust this value to control group separation
# for i, ((temp, time), group) in enumerate(groups):
#     y_offset = i * y_spacing  # Add spacing between groups
#     color = colors[i % len(colors)]
#     marker = markers[i % len(markers)]
#
#     # Scatter plot with offset
#     plt.scatter(group['Predicted Temperature'], group['Predicted Time'] + y_offset,
#                 label=f'True Temp: {temp}, True Time: {time}', alpha=0.7, marker=marker, color=color)
#
#     # Highlight the area of each group (convex hull approximation)
#     hull_points = np.column_stack((group['Predicted Temperature'], group['Predicted Time'] + y_offset))
#     hull = plt.Polygon(hull_points[np.argsort(hull_points[:, 0])], alpha=0.2, color=color, edgecolor=None)
#     plt.gca().add_patch(hull)
#
# # Add plot details
# plt.xlabel('Predicted Temperature')
# plt.ylabel('Predicted Time (with Offset)')
# plt.title('Temperature-Time Clusters with Group Areas')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
