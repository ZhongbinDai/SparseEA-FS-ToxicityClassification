function [model] = initModelOfFs_SparseEa(dataSetName, classificationModel)

	dataSetPath = '..\TestData\';                                            % 测试数据集所在路径
    variateName = dataSetName(1: find(dataSetName == '.') - 1);             % 获取文件名(不含后缀名)
    load([dataSetPath dataSetName]);                                        % 载入数据
    dataSet = eval(variateName);                                            % 所载入的数据赋值给dataSet
    if isa(dataSet, 'table') 
        dataSet = table2array(dataSet);
    end
    
    global historyIndividualRecord;                                         % 全局变量,历史个体库,第1列为个体ID，
    historyIndividualRecord = zeros(1, 2);
    
    model.dataSet = dataSet;
    model.sampleLabel = dataSet(:, 1);                                      % 样本标签
    model.sampleFeature = dataSet(:, 2: end);                               % 样本特征
    [N, D] = size(model.sampleFeature);
    model.numOfDecVariables = D;                                            % 特征数
    model.numOfSample = N;                                                  % 样本数量
    
    rateOfTrain = 0.8;
    rateOfTest = 1 - rateOfTrain;
    [trainFeature, trainLabel, testFeature, testLabel] = divideTrainAndTestData(model.sampleFeature, model.sampleLabel, rateOfTrain, rateOfTest);
    model.trainFeature = trainFeature;
    model.trainLabel = trainLabel;
    model.testFeature = testFeature;
    model.testLabel = testLabel;
    
    model.basePrecision = getIndividualFitness(ones(1, D), model);          % 基础分类精度
    
    model.classificationModel = classificationModel;                        % 分类模型
    model.initIndividual = @initIndividual;                                 % 初始化个体
	model.getIndividualFitness = @getIndividualFitness;                     % 计算个体适应度
    model.updateFeatureScore = @updateFeatureScore;
    
    model.variation = @variation;                                           % sparse种群交叉变异操作
    model.featureScore = getFeatureScore(model);                            % 特征评分
end


%% 初始化个体
function [individual] = initIndividual(model)
    individual = sparseInitIndividual(model);
end



%% 计算个体适应度
function [individualFitness] = getIndividualFitness(individual, model)
    global historyIndividualRecord;
    
    id = getIndividualId(individual);                                       % 计算个体id,不同个体有唯一id与之对应
    index = searchIndex(id, historyIndividualRecord(:, 1));                 % 查找个体是否存在历史个体库中，返回值>0则存在，否则不存在
    
    if index == -1                                                          % 如果该个体不在历史个体库中，则通过构造决策树分类计算适应度
        [precision] = getPrecision(model, individual);
        numOfhistoryRecord = size(historyIndividualRecord, 1);              % 历史库中个体数目
        historyIndividualRecord(numOfhistoryRecord + 1, 1) = id;            % 增加新记录
        historyIndividualRecord(numOfhistoryRecord + 1, 2) = precision;     % 分类精度（个体适应度）
        index = numOfhistoryRecord + 1;
    end
    individualFitness = historyIndividualRecord(index, 2);                  % 适应度越大越好,分类精度
end

%% 计算分类精度
function [precision] = getPrecision(model, individual)
    trainFeature = model.trainFeature(:, individual == 1);
    testFeature = model.testFeature(:, individual == 1);
    trainLabel = model.trainLabel;
    testLabel = model.testLabel;
    [preLabel] = predictOfCtree(trainFeature, trainLabel, testFeature);        % 决策树分类
    numOfCorrect = length(find(preLabel == testLabel));                        % 正确分类数目
    precision = numOfCorrect / length(preLabel);                               % 分类准确率
end

%% 为每个个体算出唯一id
function id = getIndividualId(individual)
% 例：个体"1011"其id为13
    N = size(individual, 2);
    id = 0;
    for  i = 1 : N
        id = id + individual(i) * 2^(i-1);
    end
end

%% 从一维矩阵X中找到y的位置
function index = searchIndex(y, X)
    index = find (y == X);
    if isempty(index)
        index = -1;
    else
        index = index(1);
    end
end

%% 从1-Upper中随机选取两个不相同的整数
function [x, y] = getTwoRandValue(Upper)
    R = randperm(Upper);
    x = R(1);
    y = R(2);
end

%% sparse初始化个体
function [individual] = sparseInitIndividual(model)
    featureScore = model.featureScore;
    numOfDecVariables = model.numOfDecVariables;
    
	individual = zeros(1, numOfDecVariables);
    for j = 1 : numOfDecVariables * rand
        [m, n] = getTwoRandValue(numOfDecVariables);                   % 随机选取两个不相同的整数
        if featureScore(m) < featureScore(n)
            individual(m) = 1;
        else
            individual(n) = 1;
        end
    end
end

function [featureScore] = getFeatureScore(model)
    numOfDecVariables = model.numOfDecVariables;
	featureMat = eye(numOfDecVariables);                                    % 单位矩阵
    featureScore = getFitness(featureMat, model);                             % 计算种群适应度
end

%% sparse遗传操作
function [newPopulation] = variation(population, popFitness, model)
    pop1 = selectionOperationOfTournament(population, popFitness);          % 选择操作
    pop2 = selectionOperationOfTournament(population, popFitness);
    P = [pop1; pop2];                                                       % 规模为2N的种群
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
        
        % 交叉
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
            
        % 变异
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
    [popFitness, index] = sort(popFitness);                             % 根据适应度从小到大排序
    population = population(index, :);
    individualScore = (1 : length(popFitness));
    numOfDecVariables = size(population, 2);                            % 特征数
    maskScore = population .* repmat(individualScore, [numOfDecVariables 1])';
    featureScore = sum(maskScore)';
end











