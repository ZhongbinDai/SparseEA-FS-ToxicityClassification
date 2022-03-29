Sheet01 = xlsread('..\TestData\分类数据.xlsx', 'Sheet1');
Sheet02 = xlsread('..\TestData\分类数据.xlsx', 'Sheet2');
Sheet03 = xlsread('..\TestData\分类数据.xlsx', 'Sheet3');

FsData2 = [Sheet01; Sheet02];                                               % 二分类数据集
multi_class_result = Sheet03(:, 2:5) * [1 2 3 4]';                          % 多分类结果
Sheet05 = [multi_class_result, Sheet02(:, 2:end)];                         
FsData5 = [Sheet01; Sheet05];                                               % 五分类数据集


save('..\TestData\FsData2.mat', 'FsData2');
save('..\TestData\FsData5.mat', 'FsData5');
