% 读取数据
data = readtable('train.xlsx'); % 请根据实际文件名调整
epoch = data.Epoch; 
train_loss = data.TrainLoss; 
val_accuracy = data.ValidationAccuracy;

% 拟合训练损失
% 将epoch转为列向量
X_train_loss = epoch(:);
y_train_loss = train_loss(:);

% 创建SVM模型
svmModel_loss = fitrsvm(X_train_loss, y_train_loss, 'KernelFunction', 'gaussian', 'Standardize', true);

% 拟合验证准确率
X_train_accuracy = epoch(:);
y_train_accuracy = val_accuracy(:);

% 创建SVM模型
svmModel_accuracy = fitrsvm(X_train_accuracy, y_train_accuracy, 'KernelFunction', 'gaussian', 'Standardize', true);

% 生成拟合范围
x_fit = linspace(min(epoch), max(epoch), 100)';

% 预测
y_fit_loss = predict(svmModel_loss, x_fit);
y_fit_accuracy = predict(svmModel_accuracy, x_fit);

% 设置全局字体大小
font_size = 28;

% 绘制结果
figure;

subplot(1, 2, 1);
scatter(epoch, train_loss, 'b', 'filled');
hold on;
plot(x_fit, y_fit_loss, 'r', 'LineWidth', 2);
title('Epoch vs Train Loss', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('Train Loss', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);

% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);

subplot(1, 2, 2);
scatter(epoch, val_accuracy, 'g', 'filled');
hold on;
plot(x_fit, y_fit_accuracy, 'k', 'LineWidth', 2);
title('Epoch vs Validation Accuracy', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('Validation Accuracy', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);

% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);

hold off;
