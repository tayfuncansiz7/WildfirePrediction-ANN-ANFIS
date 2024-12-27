
data = readtable("veriseti.csv");

X = data{:, 1:3}; % bizde 1:3 çünkü verisetinde ilk 3 sütun girdi
y = data{:, 4}; % bizde 4 çünkü verisetinde 4. sütun çýktý

rng(42);
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

net = feedforwardnet([10, 5]);

net = train(net, X_train', y_train');

predictions = net(X_test');

rmse = sqrt(mean((predictions - y_test').^2));
fprintf('RMSE: %f\n', rmse);

 errors = abs(y_test' - predictions);
    
    absolute_percentage_errors = errors ./ abs(y_test');
    
    absolute_percentage_errors(isnan(absolute_percentage_errors)) = 0;
    absolute_percentage_errors(isinf(absolute_percentage_errors)) = 0;
  
    mape_value = mean(absolute_percentage_errors) * 100;
  
fprintf('MAPE: %f%%\n', mape_value);

sse = sum((predictions - y_test').^2);
sst = sum((y_test' - mean(y_test')).^2);
rsquared = 1 - sse / sst;
fprintf('R²: %f\n', rsquared);
