% ANFIS de�erleri i�in ROC Analizi
tahmin = anfisOut; % Tahmin de�erleri girildi (ANFIS Tahmin ��kt�s�)
gercek_sinif = data.Classes; % Kar��la�t�r�lacak de�erler (Verisetindeki Classes S�tunu, yani Outputlar)

% ANN de�erleri i�in ROC Analizi
% tahmin = predictions; % Tahmin de�erleri girildi (ANFIS Tahmin ��kt�s�)
% gercek_sinif = y_test'; % Kar��la�t�r�lacak de�erler (ANN'deki Output De�erleri)

[X, Y, ~, AUC] = perfcurve(gercek_sinif, tahmin, true);

% ROC e�risini �izdirme
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Analizi');
grid on;

% AUC de�erini ekrana yazd�rma
fprintf('AUC De�eri:�%.4f\n',AUC);