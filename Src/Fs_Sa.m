% -------------------------------------------------------------------------
% 模拟退火（SA）+ 求解特征选择问题（FS）+ 毒性分类
% @作者：冰中呆
% @邮箱：1209805090@qq.com
% @时间：2022.03.28
% -------------------------------------------------------------------------
%% 清空
clear;                                                                      % 清除所有变量
close all;                                                                  % 清图
clc;                                                                        % 清屏
%% 参数配置
addpath(genpath('.\'));                                                     % 将当前文件夹下的所有文件夹都包括进调用函数的目录
rng(0);                                                                     % 随机种子

populationSize = 2;                                                         % 种群规模
maxGeneration = 5000;                                                      % 最大进化代数

% dataSetName = 'BaseData.mat';                                             % 数据集
dataSetName = 'FsData5.mat';                                                % 数据集
classificationModel = @predictOfCtree;                                      % 决策树分类模型
% classificationModel = @predictOfSvm;                                      % SVM分类模型

[model] = initModelOfFs(dataSetName, classificationModel);                  % 问题定义
numOfDecVariables = model.numOfDecVariables;                                % 决策变量维度

Temperature = 100 * numOfDecVariables;                                      % 初始温度
MarkovChain = 5;                                                         	% 马可夫链长度
AttenuationFactor = 0.98;                                                   % 温度衰减系数
mutationRate = 0.01;                                                        % 变异概率


%% 初始化
population = initialPopulation(populationSize, model);                      % 初始化种群
popFitness = getFitness(population, model);                                 % 计算种群适应度

bestIndividualSet = zeros(maxGeneration, numOfDecVariables);                % 每代最优个体集合
bestFitnessSet = zeros(maxGeneration, 1);                                   % 每代最高适应度集合
avgFitnessSet = zeros(maxGeneration, 1);                                    % 每代平均适应度集合

%% 进化
for i = 1 : maxGeneration
    [bestIndividual, bestFitness, avgFitness] = getBestIndividualAndFitness(population, popFitness);
    bestIndividualSet(i, :) = bestIndividual;                               % 第i代最优个体
    bestFitnessSet(i) = bestFitness;                                        % 第i代最高适应度
    avgFitnessSet(i) = avgFitness;                                          % 第i代种群平均适应度
    fprintf('第%i代种群的最优值：%.3f\n', i, bestFitness);
    if mod(i, 50) == 0
        showEvolCurve(1, i, bestFitnessSet, avgFitnessSet);                 % 显示进化曲线
    end
    
    newPopulation = population;
    newPopFitness = popFitness;
	for j = 1 : MarkovChain
        tempPopulation = mutationOperationOf01(newPopulation, mutationRate);        % 随机扰动
        tempPopFitness = getFitness(tempPopulation, model);                         % 计算种群适应度
        [newPopulation, newPopFitness] = metropolisRule(newPopulation, newPopFitness, tempPopulation, tempPopFitness, Temperature);
	end
    
    % Mode==0,无精英选择,(模拟退火无需精英策略)
    [population, popFitness] = eliteStrategy(population, popFitness, newPopulation, newPopFitness, 1);
    Temperature = Temperature * AttenuationFactor;                          % 温度不断下降
end


%% 结果出图
pngName = sprintf('%s_%s_N%d_Gen%d.png', 'SA', 'FsData5', populationSize, maxGeneration);
dataSetPath = '..\Result\';
saveas(gcf, [dataSetPath  pngName]);
fprintf('分类准确率提升：%.3f%%\n', (bestFitness - model.basePrecision) * 100);