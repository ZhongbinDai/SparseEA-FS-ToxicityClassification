function [population, popFitness] = metropolisRule(population, popFitness, newPopulation, newPopFitness, Temperature)
% Metropolis准则:以概率接受新状态
    populationSize = size(population, 1);
    for k = 1 : populationSize
        delta_e = newPopFitness(k) - popFitness(k);                         % 新老距离的差值,相当于能量
        if delta_e > 0 || exp(delta_e / Temperature) > rand()               % 新路线好于旧路线,用新路线代替旧路线;或以概率选择是否接受新解;
            population(k, :) = newPopulation(k, :);
            popFitness(k) = newPopFitness(k);
        end
    end
    
end

