% ANFIS deðerleri için ROC Analizi
tahmin = anfisOut; % Tahmin deðerleri girildi (ANFIS Tahmin Çýktýsý)
gercek_sinif = data.Classes; % Karþýlaþtýrýlacak deðerler (Verisetindeki Classes Sütunu, yani Outputlar)

% ANN deðerleri için ROC Analizi
% tahmin = predictions; % Tahmin deðerleri girildi (ANFIS Tahmin Çýktýsý)
% gercek_sinif = y_test'; % Karþýlaþtýrýlacak deðerler (ANN'deki Output Deðerleri)

[X, Y, ~, AUC] = perfcurve(gercek_sinif, tahmin, true);

% ROC eðrisini çizdirme
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Analizi');
grid on;

% AUC deðerini ekrana yazdýrma
fprintf('AUC Deðeri: %.4f\n',AUC);