Sheet01 = xlsread('..\TestData\��������.xlsx', 'Sheet1');
Sheet02 = xlsread('..\TestData\��������.xlsx', 'Sheet2');
Sheet03 = xlsread('..\TestData\��������.xlsx', 'Sheet3');

FsData2 = [Sheet01; Sheet02];                                               % ���������ݼ�
multi_class_result = Sheet03(:, 2:5) * [1 2 3 4]';                          % �������
Sheet05 = [multi_class_result, Sheet02(:, 2:end)];                         
FsData5 = [Sheet01; Sheet05];                                               % ��������ݼ�


save('..\TestData\FsData2.mat', 'FsData2');
save('..\TestData\FsData5.mat', 'FsData5');
