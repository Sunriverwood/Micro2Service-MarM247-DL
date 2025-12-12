import pandas as pd
import re

# Read the content of the uploaded file
file_path = 'train3.txt'
with open(file_path, 'r') as file:
    content = file.readlines()

# Extract epoch, train_loss, train_r2, val_loss, val_r2
data = []
pattern = r"\[epoch (\d+)\] train_loss: ([\d.]+), train_r2: ([\d.-]+), val_loss: ([\d.]+), val_r2: ([\d.-]+)"

for line in content:
    match = re.search(pattern, line)
    if match:
        epoch, train_loss, train_r2, val_loss, val_r2 = match.groups()
        data.append({
            'epoch': int(epoch),
            'Train Loss': float(train_loss),
            'Train R2': float(train_r2),
            'Val Loss': float(val_loss),
            'Val R2': float(val_r2)
        })

# Create a DataFrame and save it to an Excel file
df = pd.DataFrame(data)

# 保存为 Excel
output_file = 'train3.xlsx'
df.to_excel(output_file, index=False)

print(f"数据已保存到 {output_file}")
