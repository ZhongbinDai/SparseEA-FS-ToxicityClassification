function [newPopulation] = mutationOperationOf01(population, mutationRate)
% 01变异操作
	[populationSize, numOfDecVariables] = size(population);
    newPopulation = ones(size(population));
    for i = 1 : populationSize
        if(rand < mutationRate)
            mPoint = round(rand * numOfDecVariables);
            if mPoint <= 0
                mPoint = 1;
            end
            newPopulation(i, :) = population(i, :);
            if newPopulation(i, mPoint) == 0
                newPopulation(i, mPoint) = 1;
            elseif newPopulation(i, mPoint) == 1
                newPopulation(i, mPoint) = 0;
            end
        else
            newPopulation(i, :) = population(i, :);
        end
    end

end