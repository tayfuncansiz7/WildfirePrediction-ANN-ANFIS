%%
% ANFIS'i çalýþtýrmak için ilk olarak bu kod bloðunu çalýþtýrýn.
% fuzzy tool'a giriþ yapýn.
% ANFIS'e girin.
% Training Data Load kýsmýndan trainData_clean'i dahil edin.
% Testing Data Load kýsmýndan testData_clean'i dahil edin.
% Generate FIS ile FIS oluþturun.
% Epoch deðerini 100'e ayarlayýp Train'i baþlatýn.
% Ýþlem bittikten sonra Kurallarý Workspace'e anfisOutput adýyla kaydedin.

Data = readmatrix('Algeria.csv');

predict = cvpartition(size(Data, 1), "HoldOut", 0.3);
trainData = Data(training(predict), :);
testData = Data(test(predict), :);

Classes = Data(:, end);
DataOutput = Data(:,1:3);  
index1 = any(isnan(trainData) | isinf(trainData), 2); 
trainData_clean = trainData(~index1, :);   
index2 = any(isnan(testData) | isinf(testData), 2); 
testData_clean = testData(~index2, :);

%%
% Ýþlem bittikten sonra Kurallarý Workspace'e anfisOutput adýyla
% kaydettikten sonra bu kod bloðunu çalýþtýrýn.
anfisOut = evalfis(DataOutput, anfisOutput);

%%
% Metrik hesaplamalarýný görmek için bu kod bloðunu çalýþtýrýn.
rmse_clean = (anfisOut - Classes).^2;
rmse_clean(isnan(rmse_clean)) = 0;

rmse = sqrt(mean(rmse_clean));
fprintf('Root Mean Squared Error (RMSE): %f\n', rmse);

Errors = abs(Classes - anfisOut);

clean = (Errors ./ Classes);
clean(isinf(clean)) = 0;
clean(isnan(clean)) = 0;
mape = mean(abs(clean)) * 100;
fprintf('MAPE: %f%%\n', mape);

meanGercekDeger = mean(Classes);
meanGercekDeger(isnan(meanGercekDeger)) = 0;

ss_total_clean = ((Classes - meanGercekDeger).^2);
ss_total_clean(isnan(ss_total_clean)) = 0;
ss_total = sum(ss_total_clean);

errors_clean = sum(Errors.^2);
errors_clean(isnan(errors_clean)) = 0;
ss_residual = sum(errors_clean);

r_squared = 1 - (ss_residual / ss_total);
fprintf('R^2: %f\n', r_squared);


