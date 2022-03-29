function [trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(sampleFeature, sampleLabel, rateOfTrain, rateOfTest)
    numOfSample = length(sampleLabel);                                     % ��������
    randIndex = randperm(numOfSample);
    numOfTrainSet = floor(numOfSample * rateOfTrain);                      % ѵ��������
    numOfTestSet = floor(numOfSample * rateOfTest);                        % ѵ��������
    
    if numOfTrainSet + numOfTestSet > numOfSample
        numOfTestSet = numOfSample - numOfTrainSet;
    end
    
    trainFeature = sampleFeature(randIndex(1:numOfTrainSet), :);           % ѵ��������
    trainLabel = sampleLabel(randIndex(1:numOfTrainSet));                  % ѵ������ǩ

    testFeature = sampleFeature(randIndex(numOfTrainSet + 1: numOfTrainSet + numOfTestSet), :);     % ���Լ�����
    testLabel = sampleLabel(randIndex(numOfTrainSet + 1: numOfTrainSet + numOfTestSet));            % ���Լ���ǩ
end

