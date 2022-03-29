function [model] = initModelOfFs(dataSetName, classificationModel)

    dataSetPath = '..\TestData\';                                            % �������ݼ�����·��
    variateName = dataSetName(1: find(dataSetName == '.') - 1);             % ��ȡ�ļ���(������׺��)
    load([dataSetPath dataSetName]);                                        % ��������
    dataSet = eval(variateName);                                            % ����������ݸ�ֵ��dataSet
    if isa(dataSet, 'table') 
        dataSet = table2array(dataSet);
    end
    
    global historyIndividualRecord;                                         % ȫ�ֱ���,��ʷ�����,��1��Ϊ����ID��
    historyIndividualRecord = zeros(1, 2);
    
    model.dataSet = dataSet;
    model.sampleLabel = dataSet(:, 1);                                      % ������ǩ
    model.sampleFeature = dataSet(:, 2: end);                               % ��������
    [N, D] = size(model.sampleFeature);
    model.numOfDecVariables = D;                                            % ������
    model.numOfSample = N;                                                  % ��������
    
    rateOfTrain = 0.8;
    rateOfTest = 1 - rateOfTrain;
    [trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(model.sampleFeature, model.sampleLabel, rateOfTrain, rateOfTest);
    model.trainFeature = trainFeature;
    model.trainLabel = trainLabel;
    model.testFeature = testFeature;
    model.testLabel = testLabel;
    
    model.basePrecision = getIndividualFitness(ones(1, D), model);          % �������ྫ��
    
    model.classificationModel = classificationModel;                        % ����ģ��
    model.initIndividual = @initIndividual;                                 % ��ʼ������
	model.getIndividualFitness = @getIndividualFitness;                     % ���������Ӧ��
end

%% ��ʼ������
function [individual] = initIndividual(model)
% ��ͳ�Ŵ��㷨����ʼ������
% ������numOfDecVariables��0/1��ɣ�ÿ��Ϊһ������
	numOfDecVariables = model.numOfDecVariables;
	individual = round(rand(1, numOfDecVariables));
end

%% ���������Ӧ��
function [individualFitness] = getIndividualFitness(individual, model)
    global historyIndividualRecord;
    
    id = getIndividualId(individual);                                       % �������id,��ͬ������Ψһid��֮��Ӧ
    index = searchIndex(id, historyIndividualRecord(:, 1));                 % ���Ҹ����Ƿ������ʷ������У�����ֵ>0����ڣ����򲻴���
    
    if index == -1                                                          % ����ø��岻����ʷ������У���ͨ��������������������Ӧ��
        [precision] = getPrecision(model, individual);
        numOfhistoryRecord = size(historyIndividualRecord, 1);              % ��ʷ���и�����Ŀ
        historyIndividualRecord(numOfhistoryRecord + 1, 1) = id;            % �����¼�¼
        historyIndividualRecord(numOfhistoryRecord + 1, 2) = precision;     % ���ྫ�ȣ�������Ӧ�ȣ�
        index = numOfhistoryRecord + 1;
    end
    individualFitness = historyIndividualRecord(index, 2);                  % ��Ӧ��Խ��Խ��,���ྫ��
end

%% ������ྫ��
function [precision] = getPrecision(model, individual)
    trainFeature = model.trainFeature(:, individual == 1);
    testFeature = model.testFeature(:, individual == 1);
    trainLabel = model.trainLabel;
    testLabel = model.testLabel;
    [preLabel] = predictOfCtree(trainFeature, trainLabel, testFeature);        % ����������
    numOfCorrect = length(find(preLabel == testLabel));                        % ��ȷ������Ŀ
    precision = numOfCorrect / length(preLabel);                               % ����׼ȷ��
end


function [classLoss] = getClassLoss2(data)
    Y = data(:, 1);
    X = data(:, 2: end);
    SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto');
    CVSVMModel = crossval(SVMModel);
    classLoss = kfoldLoss(CVSVMModel);
end


%% Ϊÿ���������Ψһid
function id = getIndividualId(individual)
% ��������"1011"��idΪ13
    N = size(individual, 2);
    id = 0;
    for  i = 1 : N
        id = id + individual(i) * 2^(i-1);
    end
end


%% ��һά����X���ҵ�y��λ��
function index = searchIndex(y, X)
    index = find (y == X);
    if isempty(index)
        index = -1;
    else
        index = index(1);
    end
end























