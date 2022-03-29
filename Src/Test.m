%%
clear;                                                                      % 清除所有变量
close all;                                                                  % 清图
clc;                                                                        % 清屏
load('.\TestData\Fs_Data2.mat');
load('.\TestData\Fs_Data4.mat');


%%
BaseData = Fs_Data4;
sampleFeature = BaseData(:, 2:end);
sampleLabel = BaseData(:, 1);
rateOfTrain = 0.8;
rateOfTest = 1 - rateOfTrain;
[trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(sampleFeature, sampleLabel, rateOfTrain, rateOfTest);
[preLabel] = predictOfSvm(trainFeature, trainLabel, testFeature);          % SVM分类
numOfCorrect = length(find(preLabel == testLabel));                        % 正确分类数目
precision = numOfCorrect / length(preLabel);                               % 分类准确率
fprintf('测试样本数：%d 分类正确数：%d 分类准确率:%f\n',length(testLabel), numOfCorrect, precision);


%%
BaseData = Fs_Data4;
sampleFeature = BaseData(:, 2:end);
sampleLabel = BaseData(:, 1);
rateOfTrain = 0.8;
rateOfTest = 1 - rateOfTrain;
[trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(sampleFeature, sampleLabel, rateOfTrain, rateOfTest);
[preLabel] = predictOfCtree(trainFeature, trainLabel, testFeature);        % 决策树分类
numOfCorrect = length(find(preLabel == testLabel));                        % 正确分类数目
precision = numOfCorrect / length(preLabel);                               % 分类准确率
fprintf('测试样本数：%d 分类正确数：%d 分类准确率:%f\n',length(testLabel), numOfCorrect, precision);

