function [model] = initModelOfFs_SparseEa(dataSetName, classificationModel)

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
    model.updateFeatureScore = @updateFeatureScore;
    
    model.variation = @variation;                                           % sparse��Ⱥ����������
    model.featureScore = getFeatureScore(model);                            % ��������
end


%% ��ʼ������
function [individual] = initIndividual(model)
    individual = sparseInitIndividual(model);
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

%% ��1-Upper�����ѡȡ��������ͬ������
function [x, y] = getTwoRandValue(Upper)
    R = randperm(Upper);
    x = R(1);
    y = R(2);
end

%% sparse��ʼ������
function [individual] = sparseInitIndividual(model)
    featureScore = model.featureScore;
    numOfDecVariables = model.numOfDecVariables;
    
	individual = zeros(1, numOfDecVariables);
    for j = 1 : numOfDecVariables * rand
        [m, n] = getTwoRandValue(numOfDecVariables);                   % ���ѡȡ��������ͬ������
        if featureScore(m) < featureScore(n)
            individual(m) = 1;
        else
            individual(n) = 1;
        end
    end
end

function [featureScore] = getFeatureScore(model)
    numOfDecVariables = model.numOfDecVariables;
	featureMat = eye(numOfDecVariables);                                    % ��λ����
    featureScore = getFitness(featureMat, model);                             % ������Ⱥ��Ӧ��
end

%% sparse�Ŵ�����
function [newPopulation] = variation(population, popFitness, model)
    pop1 = selectionOperationOfTournament(population, popFitness);          % ѡ�����
    pop2 = selectionOperationOfTournament(population, popFitness);
    P = [pop1; pop2];                                                       % ��ģΪ2N����Ⱥ
    featureScore = model.featureScore;
    [populationSize, numOfVariables] = size(P);
    
    O = zeros(populationSize/2, numOfVariables);
    i = 1;
    while size(P,1) > 0
        populationSize = size(P,1);
        [m,n] = getTwoRandValue(populationSize);
        p = P(m,:);
        q = P(n,:);
        if m < n
            P(n,:)=[];
            P(m,:)=[];
        else
            P(m,:)=[];
            P(n,:)=[];
        end
        o = p;
        
        % ����
        if rand() < 0.5
            goal = p & (1-q);
            num = sum(goal);
            if num >0
                Y = find(goal==1);
                X = randperm(num);

                m = Y(X(1));
                n = Y(X(end));
                if featureScore(m) > featureScore(n)
                    o(m) = 0;
                else
                    o(n) = 0;
                end
            end
        else
            goal = (1-p) & q;
            num = sum(goal);
            if num >0
                Y = find(goal == 1);
                X = randperm(num);
                m = Y(X(1));
                n = Y(X(end));
                if featureScore(m) < featureScore(n)
                    o(m) = 1;
                else
                    o(n) = 1;
                end
            end
        end
            
        % ����
        if rand() < 0.5
            goal = o;
            num = sum(goal);
            if num >0
                Y = find(goal==1);
                X = randperm(num);
                m = Y(X(1));
                n = Y(X(end));
                if featureScore(m) > featureScore(n)
                    o(m) = 0;
                else
                    o(n) = 0;
                end
            end
        else
            goal = 1 - o;
            num = sum(goal);
            if num >0
                Y = find(goal==1);
                X = randperm(num);
                m = Y(X(1));
                n = Y(X(end));
                if featureScore(m) < featureScore(n)
                    o(m) = 1;
                else
                    o(n) = 1;
                end
            end
        end
        
        O(i,:)=o;
        i=i+1;
    end
    newPopulation = O;
end


function featureScore = updateFeatureScore(population, popFitness)
    [popFitness, index] = sort(popFitness);                             % ������Ӧ�ȴ�С��������
    population = population(index, :);
    individualScore = (1 : length(popFitness));
    numOfDecVariables = size(population, 2);                            % ������
    maskScore = population .* repmat(individualScore, [numOfDecVariables 1])';
    featureScore = sum(maskScore)';
end











