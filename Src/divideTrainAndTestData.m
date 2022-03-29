function [trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(sampleFeature, sampleLabel, rateOfTrain, rateOfTest)
    numOfSample = length(sampleLabel);                                     % 样本数量
    randIndex = randperm(numOfSample);
    numOfTrainSet = floor(numOfSample * rateOfTrain);                      % 训练集数量
    numOfTestSet = floor(numOfSample * rateOfTest);                        % 训练集数量
    
    if numOfTrainSet + numOfTestSet > numOfSample
        numOfTestSet = numOfSample - numOfTrainSet;
    end
    
    trainFeature = sampleFeature(randIndex(1:numOfTrainSet), :);           % 训练集特征
    trainLabel = sampleLabel(randIndex(1:numOfTrainSet));                  % 训练集标签

    testFeature = sampleFeature(randIndex(numOfTrainSet + 1: numOfTrainSet + numOfTestSet), :);     % 测试集特征
    testLabel = sampleLabel(randIndex(numOfTrainSet + 1: numOfTrainSet + numOfTestSet));            % 测试集标签
end

