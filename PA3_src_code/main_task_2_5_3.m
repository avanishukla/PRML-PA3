%class1 -> tallbuilding, class2 -> highway, class3 -> street
% clc;
% clearvars;
% mydir = 'D:\prml\PRML2018_assignment_3\image_dataset\Features\';
% 
% class1_path = strcat(mydir,'tallbuilding\');
% foldername = fullfile(class1_path);
% pth = genpath(foldername);
% pathTest = regexp([pth ';'],'(.*?);','tokens');
% list = dir([pathTest{1,1}{:} '\*.jpg_color_edh_entropy']);
% class1_cell = cell(numel(list),1);
% 
% if(~isempty(list))
%    for j = 1: numel(list)
%         fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
%         fileread = fscanf(fileID,'%f',[23 Inf]);
%         fileread = fileread';
%         class1_cell{j,1} = fileread;
%         fclose(fileID);
%    end
% end
% 
% size_train_class1 = floor(0.7*size(class1_cell,1));
% size_validation_class1 = floor(0.1*size(class1_cell,1));
% size_test_class1 = size(class1_cell,1) - size_train_class1 - size_validation_class1;
% train1_cell = class1_cell(1:size_train_class1,1);
% validation1_cell = class1_cell(size_train_class1+1:size_train_class1+size_validation_class1,1);
% test1_cell = class1_cell(size_train_class1+size_validation_class1+1:size_train_class1+size_validation_class1+size_test_class1,1);
% train_data1 = cell2mat(train1_cell);
% val_data1 = cell2mat(validation1_cell);
% test_data1 = cell2mat(test1_cell);
% 
% class2_path = strcat(mydir,'highway\');
% foldername = fullfile(class2_path);
% pth = genpath(foldername);
% pathTest = regexp([pth ';'],'(.*?);','tokens');
% list = dir([pathTest{1,1}{:} '\*.jpg_color_edh_entropy']);
% class2_cell = cell(numel(list),1);
% if(~isempty(list))
%    for j = 1: numel(list)
%         fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
%         fileread = fscanf(fileID,'%f',[23 Inf]);
%         fileread = fileread';
%         class2_cell{j,1} = fileread;
%         fclose(fileID);
%    end
% end
% 
% size_train_class2 = floor(0.7*size(class2_cell,1));
% size_validation_class2 = floor(0.1*size(class2_cell,1));
% size_test_class2 = size(class2_cell,1) - size_train_class2 - size_validation_class2;
% train2_cell = class2_cell(1:size_train_class2,1);
% validation2_cell = class2_cell(size_train_class2+1:size_train_class2+size_validation_class2,1);
% test2_cell = class2_cell(size_train_class2+size_validation_class2+1:size_train_class2+size_validation_class2+size_test_class2,1);
% train_data2 = cell2mat(train2_cell);
% val_data2 = cell2mat(validation2_cell);
% test_data2 = cell2mat(test2_cell);
% 
% class3_path = strcat(mydir,'street\');
% foldername = fullfile(class3_path);
% pth = genpath(foldername);
% pathTest = regexp([pth ';'],'(.*?);','tokens');
% list = dir([pathTest{1,1}{:} '\*.jpg_color_edh_entropy']);
% class3_cell = cell(numel(list),1);
% if(~isempty(list))
%    for j = 1: numel(list)
%         fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
%         fileread = fscanf(fileID,'%f',[23 Inf]);
%         fileread = fileread';
%         class3_cell{j,1} = fileread;
%         fclose(fileID);
%    end
% end
% 
% size_train_class3 = floor(0.7*size(class3_cell,1));
% size_validation_class3 = floor(0.1*size(class3_cell,1));
% size_test_class3 = size(class3_cell,1) - size_train_class3 - size_validation_class3;
% train3_cell = class3_cell(1:size_train_class3,1);
% validation3_cell = class3_cell(size_train_class3+1:size_train_class3+size_validation_class3,1);
% test3_cell = class3_cell(size_train_class3+size_validation_class3+1:size_train_class3+size_validation_class3+size_test_class3,1);
% train_data3 = cell2mat(train3_cell);
% val_data3 = cell2mat(validation3_cell);
% test_data3 = cell2mat(test3_cell);
% 
% train1_label = ones(size(train_data1,1),1);
% train2_label = 2.*ones(size(train_data2,1),1);
% train3_label = 3.*ones(size(train_data3,1),1);
% train_data = [train_data1;train_data2;train_data3];
% train_label = [train1_label;train2_label;train3_label];
% 
% val1_label = ones(size(val_data1,1),1);
% val2_label = 2.*ones(size(val_data2,1),1);
% val3_label = 3.*ones(size(val_data3,1),1);
% val_data = [val_data1;val_data2;val_data3];
% val_label = [val1_label;val2_label;val3_label];
% 
% test1_label = ones(size(test_data1,1),1);
% test2_label = 2.*ones(size(test_data2,1),1);
% test3_label = 3.*ones(size(test_data3,1),1);
% test_data = [test_data1;test_data2;test_data3];
% test_label = [test1_label;test2_label;test3_label];

%model = svmtrain(train_label,train_data,'-s 0 -t 2 -c 1 -b 1');
%model = svmtrain(train_label,train_data,'-s 0 -t 1 -c 10 -d 5 -g 0.0435 -b 1');

[train_predict_label, train_accuracy, train_prob_values] = svmpredict(train_label, train_data, model,'-b 1');
[val_predict_label, val_accuracy, val_prob_values] = svmpredict(val_label, val_data, model,'-b 1');
[test_predict_label, test_accuracy, test_prob_values] = svmpredict(test_label, test_data, model,'-b 1');

train_image_label = zeros(size(train_label,1)/36,1);
train_image_plabel = zeros(size(train_prob_values,1)/36,1);
for i = 1 : size(train_prob_values,1)/36
    sum = zeros(1,3);
    for j = 1:36
        sum(1,1) = sum(1,1) + log(train_prob_values(j+(i-1)*36,1));
        sum(1,2) = sum(1,2) + log(train_prob_values(j+(i-1)*36,2));
        sum(1,3) = sum(1,3) + log(train_prob_values(j+(i-1)*36,3));
    end
    train_image_plabel(i,1) = find(sum==max(sum));
    train_image_label(i,1) = train_label(1+(i-1)*36,1);
end

val_image_label = zeros(size(val_label,1)/36,1);
val_image_plabel = zeros(size(val_prob_values,1)/36,1);
for i = 1 : size(val_prob_values,1)/36
    sum = zeros(1,3);
    for j = 1:36
        sum(1,1) = sum(1,1) + log(val_prob_values(j+(i-1)*36,1));
        sum(1,2) = sum(1,2) + log(val_prob_values(j+(i-1)*36,2));
        sum(1,3) = sum(1,3) + log(val_prob_values(j+(i-1)*36,3));
    end
    val_image_plabel(i,1) = find(sum==max(sum));
    val_image_label(i,1) = val_label(1+(i-1)*36,1);
end

test_image_label = zeros(size(test_label,1)/36,1);
test_image_plabel = zeros(size(test_prob_values,1)/36,1);
for i = 1 : size(test_prob_values,1)/36
    sum = zeros(1,3);
    for j = 1:36
        sum(1,1) = sum(1,1) + log(test_prob_values(j+(i-1)*36,1));
        sum(1,2) = sum(1,2) + log(test_prob_values(j+(i-1)*36,2));
        sum(1,3) = sum(1,3) + log(test_prob_values(j+(i-1)*36,3));
    end
    test_image_plabel(i,1) = find(sum==max(sum));
    test_image_label(i,1) = test_label(1+(i-1)*36,1);
end

acc_train = 0;
acc_val = 0;
acc_test = 0;
confusion_mat_train = zeros(3,3);
for i=1:size(train_image_label,1)
    row = train_image_label(i);
    col = train_image_plabel(i);
    if row==col
        acc_train = acc_train +1;
    end
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_val = zeros(3,3);
for i=1:size(val_image_label,1)
    row = val_image_label(i);
    col = val_image_plabel(i);
    if row==col
        acc_val = acc_val +1;
    end
    confusion_mat_val(row,col) = confusion_mat_val(row,col) +1; 
end

confusion_mat_test = zeros(3,3);
for i=1:size(test_image_label,1)
    row = test_image_label(i);
    col = test_image_plabel(i);
    if row==col
        acc_test = acc_test +1;
    end
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end

acc_train = acc_train*100/size(train_image_label,1);
acc_val = acc_val*100/size(val_image_label,1);
acc_test = acc_test*100/size(test_image_label,1);
