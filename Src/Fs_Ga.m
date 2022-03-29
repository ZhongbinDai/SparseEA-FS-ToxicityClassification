% -------------------------------------------------------------------------
% 遗传算法（GA）+ 求解特征选择问题（FS）+ 毒性分类
% @作者：冰中呆
% @邮箱：1209805090@qq.com
% @时间：2022.03.26
% -------------------------------------------------------------------------
%% 清空
clear;                                                                      % 清除所有变量
close all;                                                                  % 清图
clc;                                                                        % 清屏
%% 参数配置
addpath(genpath('.\'));                                                     % 将当前文件夹下的所有文件夹都包括进调用函数的目录
rng(0);                                                                     % 随机种子

populationSize = 30;                                                        % 种群规模
maxGeneration = 1000;                                                       % 最大进化代数
crossoverRate = 0.6;                                                        % 交叉概率
mutationRate = 0.03;                                                        % 变异概率

% dataSetName = 'FsData2.mat';                                                % 数据集
dataSetName = 'FsData5.mat';                                                % 数据集
classificationModel = @predictOfCtree;                                      % 决策树分类模型
% classificationModel = @predictOfSvm;                                      % SVM分类模型

[model] = initModelOfFs(dataSetName, classificationModel);                  % 问题定义

%% 初始化
population = initialPopulation(populationSize, model);                      % 初始化种群
popFitness = getFitness(population, model);                                 % 计算种群适应度
numOfDecVariables = size(population, 2);                                    % 决策变量维度

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
    
    
    newPopulation = selectionOperationOfTournament(population, popFitness);	% 选择操作
    newPopulation = crossoverOperation(newPopulation, crossoverRate);	    % 交叉操作
    newPopulation = mutationOperationOf01(newPopulation, mutationRate);     % 变异操作

    newPopFitness = getFitness(newPopulation, model);                       % 子代种群适应度
    [population, popFitness] = eliteStrategy(population, popFitness, newPopulation, newPopFitness, 2); % 精英策略
end

%% 结果出图
pngName = sprintf('%s_%s_N%d_Gen%d.png', 'GA', 'FsData5', populationSize, maxGeneration);
dataSetPath = '..\Result\';
saveas(gcf, [dataSetPath  pngName]);
fprintf('分类准确率提升：%.3f%%\n', (bestFitness - model.basePrecision) * 100);




