% -------------------------------------------------------------------------
% �Ŵ��㷨��GA��+ �������ѡ�����⣨FS��+ ���Է���
% @���ߣ����д�
% @���䣺1209805090@qq.com
% @ʱ�䣺2022.03.26
% -------------------------------------------------------------------------
%% ���
clear;                                                                      % ������б���
close all;                                                                  % ��ͼ
clc;                                                                        % ����
%% ��������
addpath(genpath('.\'));                                                     % ����ǰ�ļ����µ������ļ��ж����������ú�����Ŀ¼
% rng(0);                                                                     % �������

populationSize = 30;                                                        % ��Ⱥ��ģ
maxGeneration = 1000;                                                       % ����������
crossoverRate = 0.6;                                                        % �������
mutationRate = 0.01;                                                        % �������

% dataSetName = 'FsData2.mat';                                                % ���ݼ�
dataSetName = 'FsData5.mat';                                                % ���ݼ�
classificationModel = @predictOfCtree;                                      % ����������ģ��
% classificationModel = @predictOfSvm;                                      % SVM����ģ��

[model] = initModelOfFs(dataSetName, classificationModel);                  % ���ⶨ��

%% ��ʼ��
population = initialPopulation(populationSize, model);                      % ��ʼ����Ⱥ
popFitness = getFitness(population, model);                                 % ������Ⱥ��Ӧ��
numOfDecVariables = size(population, 2);                                    % ���߱���ά��

bestIndividualSet = zeros(maxGeneration, numOfDecVariables);                % ÿ�����Ÿ��弯��
bestFitnessSet = zeros(maxGeneration, 1);                                   % ÿ�������Ӧ�ȼ���
avgFitnessSet = zeros(maxGeneration, 1);                                    % ÿ��ƽ����Ӧ�ȼ���


%% ����
for i = 1 : maxGeneration
    
    [bestIndividual, bestFitness, avgFitness] = getBestIndividualAndFitness(population, popFitness);
    bestIndividualSet(i, :) = bestIndividual;                               % ��i�����Ÿ���
    bestFitnessSet(i) = bestFitness;                                        % ��i�������Ӧ��
    avgFitnessSet(i) = avgFitness;                                          % ��i����Ⱥƽ����Ӧ��
    fprintf('��%i����Ⱥ������ֵ��%.3f\n', i, bestFitness);
    if mod(i, 50) == 0
        showEvolCurve(1, i, bestFitnessSet, avgFitnessSet);                 % ��ʾ��������
    end
    
    
    newPopulation = selectionOperationOfTournament(population, popFitness);	% ѡ�����
    newPopulation = crossoverOperation(newPopulation, crossoverRate);	    % �������
    newPopulation = mutationOperationOf01(newPopulation, mutationRate);     % �������

    newPopFitness = getFitness(newPopulation, model);                       % �Ӵ���Ⱥ��Ӧ��
    [population, popFitness] = eliteStrategy(population, popFitness, newPopulation, newPopFitness, 2); % ��Ӣ����
end

%% �����ͼ
pngName = sprintf('%s_%s_N%d_Gen%d.png', 'GA', 'FsData5', populationSize, maxGeneration);
dataSetPath = '..\Result\';
saveas(gcf, [dataSetPath  pngName]);
fprintf('����׼ȷ��������%.3f%%\n', (bestFitness - model.basePrecision) * 100);