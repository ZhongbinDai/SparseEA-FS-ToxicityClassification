path = '..\TestData\';
fileName = 'class-data.xlsx';

Sheet01 = xlsread([path fileName], 'Sheet1');
Sheet02 = xlsread([path fileName], 'Sheet2');
Sheet03 = xlsread([path fileName], 'Sheet3');

FsData2 = [Sheet01; Sheet02];                                               % ���������ݼ�
multi_class_result = Sheet03(:, 2:5) * [1 2 3 4]';                          % �������
Sheet05 = [multi_class_result, Sheet02(:, 2:end)];                         
FsData5 = [Sheet01; Sheet05];                                               % ��������ݼ�
FsData4 = [multi_class_result, Sheet02(:, 3:end)];

save('..\TestData\FsData2.mat', 'FsData2');
save('..\TestData\FsData4.mat', 'FsData4');
save('..\TestData\FsData5.mat', 'FsData5');
