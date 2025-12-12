% 读取数据
data = readtable('output_tem_time.xlsx'); % 请根据实际文件名调整
epoch = data.Epoch; 
train_loss = data.TrainLoss; 
val_accuracy = data.ValidationAccuracy;

% 定义非线性模型函数
modelFuncLoss = @(b, x) b(1) + b(2) * exp(-b(3) * x);
modelFuncAccuracy = @(b, x) b(1) + b(2) * exp(b(3) * x);

% 拟合训练损失
initialGuessLoss = [1, 1, 1]; % 初始猜测参数
fitLoss = nlinfit(epoch, train_loss, modelFuncLoss, initialGuessLoss);

% 拟合验证准确率
initialGuessAccuracy = [0.5, 0.5, -0.1]; % 初始猜测参数
fitAccuracy = nlinfit(epoch, val_accuracy, modelFuncAccuracy, initialGuessAccuracy);

set(0, 'DefaultTextFontSize', 24); % 设置文本字体大小
set(0, 'DefaultAxesFontSize', 24); % 设置坐标轴字体大小

% 绘制结果
x_fit = linspace(min(epoch), max(epoch), 100); % 拟合范围

% 计算拟合值
y_fit_loss = fitLoss(1) + fitLoss(2) * exp(-fitLoss(3) * x_fit);
y_fit_accuracy = fitAccuracy(1) + fitAccuracy(2) * exp(fitAccuracy(3) * x_fit);

figure;
subplot(1, 2, 1);
scatter(epoch, train_loss, 'b', 'filled');
hold on;
plot(x_fit, y_fit_loss, 'r', 'LineWidth', 2);
title('Epoch vs Train Loss');
xlabel('Epoch');
ylabel('Train Loss');
legend('Data', 'Fitted Curve');

subplot(1, 2, 2);
scatter(epoch, val_accuracy, 'g', 'filled');
hold on;
plot(x_fit, y_fit_accuracy, 'k', 'LineWidth', 2);
title('Epoch vs Validation Accuracy');
xlabel('Epoch');
ylabel('Validation Accuracy');
legend('Data', 'Fitted Curve');

hold off;
