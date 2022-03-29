function [newPopulation] = crossoverOperation(population, crossoverRate)
% ½»²æ²Ù×÷
    [populationSize, numOfDecVariables] = size(population);
    newPopulation = ones(size(population));
    for i = 1 : 2 : populationSize - 1
        if(rand < crossoverRate)
            cPoint = round(rand * numOfDecVariables);
            newPopulation(i, :) = [population(i, 1 : cPoint), population(i + 1, cPoint + 1 : numOfDecVariables)];
            newPopulation(i + 1, :) = [population(i + 1, 1 : cPoint), population(i, cPoint + 1 : numOfDecVariables)];
        else
            newPopulation(i, :) = population(i, :);
            newPopulation(i + 1, :) = population(i + 1,:);
        end
    end
end

