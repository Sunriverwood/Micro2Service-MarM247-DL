% 读取 Excel 文件
data = readtable('800对比.xlsx'); % 替换 'your_file.xlsx' 为你的文件名

% 获取每一列的数据
epochs = data.Epoch;
validation_loss_1 = data.ValidationLoss1;
validation_accuracy_1 = data.ValidationAccuracy1;
validation_loss_2 = data.ValidationLoss2;
validation_accuracy_2 = data.ValidationAccuracy2;

% 计算相邻五个数据点的平均值
window_size = 5;

validation_loss_1_avg = movmean(validation_loss_1, window_size);
validation_accuracy_1_avg = movmean(validation_accuracy_1, window_size);
validation_loss_2_avg = movmean(validation_loss_2, window_size);
validation_accuracy_2_avg = movmean(validation_accuracy_2, window_size);

% 创建图形
figure;
set(gcf, 'DefaultAxesFontSize', 28, 'DefaultTextFontSize', 28);

% 绘制两个 Validation Loss 的变化曲线
subplot(1, 2, 1); % 两个子图，选择第一个
plot(epochs, validation_loss_1_avg, '-b', 'DisplayName', 'Without C');
hold on;
plot(epochs, validation_loss_2_avg, '-r', 'DisplayName', 'Include C');
xlabel('Epoch');
ylabel('Loss');
title('Validation Loss');
legend;

% 绘制两个 Validation Accuracy 的变化曲线
subplot(1, 2, 2); % 选择第二个子图
plot(epochs, validation_accuracy_1_avg, '-b', 'DisplayName', 'Without C');
hold on;
plot(epochs, validation_accuracy_2_avg, '-r', 'DisplayName', 'Include C');
xlabel('Epoch');
ylabel('Accuracy');
title('Validation Accuracy');
legend('Location', 'southeast'); % 设置图例位置在右下角

% 显示图形
hold off;
