function [preLabel] = predictOfCtree(trainFeature, trainLabel, testFeature)
    ctree = ClassificationTree.fit(trainFeature, trainLabel, 'minleaf', 1);     % 分类器
    % view(ctree);
    % view(ctree,'mode','graph');
    preLabel = predict(ctree, testFeature);                                     % 分类结果
end

