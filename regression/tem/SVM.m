% 读取数据
data = readtable('3000-0C.xlsx'); % 请根据实际文件名调整
epoch = data.epoch; 
train_loss = data.TrainLoss; 
val_loss = data.ValLoss;
trainr2=data.TrainR2;
valr2=data.ValR2;

% 拟合训练损失
% 将epoch转为列向量
X_train_loss = epoch(:);
y_train_loss = train_loss(:);

% 创建SVM模型
svmModel_trainloss = fitrsvm(X_train_loss, y_train_loss, 'KernelFunction', 'gaussian', 'Standardize', true);

% 拟合验证准确率
X_train_accuracy = epoch(:);
y_train_accuracy = val_loss(:);

% 创建SVM模型
svmModel_valloss = fitrsvm(X_train_accuracy, y_train_accuracy, 'KernelFunction', 'gaussian', 'Standardize', true);


X_train_trainr2=epoch(:);
y_train_trainr2=trainr2(:);
svmModel_trainr2 = fitrsvm(X_train_trainr2, y_train_trainr2, 'KernelFunction', 'gaussian', 'Standardize', true);

X_train_valr2=epoch(:);
y_train_valr2=trainr2(:);
svmModel_valr2 = fitrsvm(X_train_valr2, y_train_valr2, 'KernelFunction', 'gaussian', 'Standardize', true);

% 生成拟合范围
x_fit = linspace(min(epoch), max(epoch), 100)';

% 预测
y_fit_trainloss = predict(svmModel_trainloss, x_fit);
y_fit_valloss = predict(svmModel_valloss, x_fit);
y_fit_trainr2=predict(svmModel_trainr2,x_fit);
y_fit_valr2=predict(svmModel_valr2,x_fit);

% 设置全局字体大小
font_size = 28;

% 绘制结果
figure;

subplot(1, 4, 1);
scatter(epoch, train_loss, 'b', 'filled');
hold on;
plot(x_fit, y_fit_trainloss, 'r', 'LineWidth', 2);
title('Epoch vs Train Loss', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('Train Loss', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);

% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);

subplot(1, 4, 2);
scatter(epoch, val_loss, 'blue', 'filled');
hold on;
plot(x_fit, y_fit_valloss, 'r', 'LineWidth', 2);
title('Epoch vs Validation Loss', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('Validation Loss', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);

% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);

subplot(1, 4, 3);
scatter(epoch, trainr2, 'b', 'filled');
hold on;
plot(x_fit, y_fit_trainr2, 'r', 'LineWidth', 2);
title('Epoch vs Train R^{2}', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('trainr2', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);
ylim([0, 1]);


% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);
subplot(1, 4, 4);
scatter(epoch, valr2, 'b', 'filled');
hold on;
plot(x_fit, y_fit_valr2, 'r', 'LineWidth', 2);
title('Epoch vs Validation R^{2}', 'FontSize', font_size);
xlabel('Epoch', 'FontSize', font_size);
ylabel('valr2', 'FontSize', font_size);
legend('Data', 'SVM Fitted Curve', 'FontSize', font_size);
ylim([0, 1]);

% 设置刻度标签字体大小
set(gca, 'FontSize', font_size);

hold off;
