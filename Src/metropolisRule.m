function [population, popFitness] = metropolisRule(population, popFitness, newPopulation, newPopFitness, Temperature)
% Metropolis׼��:�Ը��ʽ�����״̬
    populationSize = size(population, 1);
    for k = 1 : populationSize
        delta_e = newPopFitness(k) - popFitness(k);                         % ���Ͼ���Ĳ�ֵ,�൱������
        if delta_e > 0 || exp(delta_e / Temperature) > rand()               % ��·�ߺ��ھ�·��,����·�ߴ����·��;���Ը���ѡ���Ƿ�����½�;
            population(k, :) = newPopulation(k, :);
            popFitness(k) = newPopFitness(k);
        end
    end
    
end

