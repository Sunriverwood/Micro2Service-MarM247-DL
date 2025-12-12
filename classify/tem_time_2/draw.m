% 读取 Excel 文件
data = readtable('800.xlsx'); % 替换 'your_file.xlsx' 为你的文件名

% 获取每一列的数据
epochs = data.Epoch;
train_loss = data.TrainLoss;
train_accuracy = data.TrainAccuracy;
validation_loss = data.ValidationLoss;
validation_accuracy = data.ValidationAccuracy;

% 计算相邻五个数据点的平均值
window_size = 5;

train_loss_avg = movmean(train_loss, window_size);
train_accuracy_avg = movmean(train_accuracy, window_size);
validation_loss_avg = movmean(validation_loss, window_size);
validation_accuracy_avg = movmean(validation_accuracy, window_size);

% 创建图形
figure;
set(gcf, 'DefaultAxesFontSize', 28, 'DefaultTextFontSize', 28);

% 绘制 Train Loss 和 Validation Loss 曲线
subplot(1, 2, 1); % 两个子图，选择第一个
plot(epochs, train_loss_avg, '-b', 'DisplayName', 'Train Loss');
hold on;
plot(epochs, validation_loss_avg, '-r', 'DisplayName', 'Validation Loss');
xlabel('Epoch');
ylabel('Loss');
title('Train Loss and Validation Loss');
legend;

% 绘制 Train Accuracy 和 Validation Accuracy 曲线
subplot(1, 2, 2); % 选择第二个子图
plot(epochs, train_accuracy_avg, '-b', 'DisplayName', 'Train Accuracy');
hold on;
plot(epochs, validation_accuracy_avg, '-r', 'DisplayName', 'Validation Accuracy');
xlabel('Epoch');
ylabel('Accuracy');
title('Train Accuracy and Validation Accuracy');
legend('Location', 'southeast'); % 设置图例位置在右下角

% 显示图形
hold off;
