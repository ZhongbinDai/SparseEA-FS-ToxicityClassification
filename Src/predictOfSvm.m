function [preLabel] = predictOfSvm(trainFeature, trainLabel, testFeature)
    classOfLabel = unique(trainLabel);
    SVMModels = cell(size(classOfLabel));
    for j = 1 : numel(classOfLabel)
        indx = trainLabel == classOfLabel(j);
        % 构造分类器
        SVMModels{j} = fitcsvm(trainFeature,indx,'ClassNames',[false true],'Standardize',true, 'KernelFunction','rbf','BoxConstraint',1);
    end
    Scores = zeros(size(testFeature, 1), numel(classOfLabel));
    for j = 1 : numel(classOfLabel)
        % 分别预测概率
        [~,score] = predict(SVMModels{j}, testFeature);
        Scores(:,j) = score(:,2);
    end
    [~,maxScore] = max(Scores,[],2);
    preLabel = classOfLabel(maxScore);
end

