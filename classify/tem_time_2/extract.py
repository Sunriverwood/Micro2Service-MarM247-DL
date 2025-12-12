import pandas as pd
import re


def extract_data(file_path):
    epochs = []
    train_losses = []
    train_accuracies=[]
    val_losses = []
    val_accuracies = []
    times = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # 提取epoch、train_loss和val_accuracy
            epoch_match = re.search(r'\[epoch (\d+)\]', line)
            train_loss_match = re.search(r'train_loss:\s*([0-9.]+)', line)
            train_accuracy_match = re.search(r'train_accurate:\s*([0-9.]+)', line)
            val_loss_match = re.search(r'val_loss:\s*([0-9.]+)', line)
            val_accuracy_match = re.search(r'val_accurate:\s*([0-9.]+)', line)
            time_match = re.search(r'^\d+\.\d+', line)

            if epoch_match and train_loss_match and val_accuracy_match and train_accuracy_match and val_loss_match:
                epochs.append(int(epoch_match.group(1)))
                train_losses.append(float(train_loss_match.group(1)))
                train_accuracies.append(float(train_accuracy_match.group(1)))
                val_losses.append(float(val_loss_match.group(1)))
                val_accuracies.append(float(val_accuracy_match.group(1)))

            if time_match:
                times.append(float(time_match.group(0)))

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies, times


def save_to_excel(epochs, train_losses, train_accuracies, val_losses, val_accuracies, times, output_file):
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Train Accuracy':train_accuracies,
        'Validation Loss':val_losses,
        'Validation Accuracy': val_accuracies,
        'Time': times
    })
    df.to_excel(output_file, index=False)


def main():
    input_file = '1000CC.txt'  # 请根据实际文件名调整
    output_file = '1000CC.xlsx'

    epochs, train_losses, train_accuracies, val_losses, val_accuracies, times = extract_data(input_file)
    save_to_excel(epochs, train_losses, train_accuracies, val_losses, val_accuracies, times, output_file)

    print("Data extracted and saved to Excel successfully!")


if __name__ == '__main__':
    main()
