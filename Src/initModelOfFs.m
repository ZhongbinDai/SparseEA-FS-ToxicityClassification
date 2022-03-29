function [model] = initModelOfFs(dataSetName, classificationModel)

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
end

%% 初始化个体
function [individual] = initIndividual(model)
% 传统遗传算法，初始化个体
% 个体由numOfDecVariables个0/1组成，每行为一个个体
	numOfDecVariables = model.numOfDecVariables;
	individual = round(rand(1, numOfDecVariables));
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


function [classLoss] = getClassLoss2(data)
    Y = data(:, 1);
    X = data(:, 2: end);
    SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto');
    CVSVMModel = crossval(SVMModel);
    classLoss = kfoldLoss(CVSVMModel);
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























