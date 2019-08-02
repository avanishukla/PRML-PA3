%class1 -> ai, class2 -> chA, class3 -> lA
clc;
clearvars;
mydir = 'D:\prml\PRML2018_assignment_3\ocr_dataset\ocr_data\HandWritten_data\DATA\FeaturesHW\';

fid = fopen(strcat(mydir,'ai.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
data1 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    data1{i,1} = C{1,1}{3*i,1}; 
end
fid = fopen(strcat(mydir,'chA.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
data2 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    data2{i,1} = C{1,1}{3*i,1}; 
end
fid = fopen(strcat(mydir,'lA.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
data3 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    data3{i,1} = C{1,1}{3*i,1}; 
end

maxcol = 0;
for i = 1 : size(data1,1)
    tmp = sscanf(data1{i,1},'%f');
    if tmp(1,1) > maxcol
        maxcol = tmp(1,1);
    end
end
for i = 1 : size(data2,1)
    tmp = sscanf(data2{i,1},'%f');
    if tmp(1,1) > maxcol
        maxcol = tmp(1,1);
    end
end
for i = 1 : size(data3,1)
    tmp = sscanf(data3{i,1},'%f');
    if tmp(1,1) > maxcol
        maxcol = tmp(1,1);
    end
end

class1 = zeros(size(data1,1),2*maxcol+1);
for i = 1 : size(data1,1)
    tmp = sscanf(data1{i,1},'%f')';
    class1(i,1:size(tmp,2)) = tmp;
end
class1 = class1(:,2:size(class1,2));

class2 = zeros(size(data2,1),2*maxcol+1);
for i = 1 : size(data2,1)
    tmp = sscanf(data2{i,1},'%f')';
    class2(i,1:size(tmp,2)) = tmp;
end
class2 = class2(:,2:size(class2,2));

class3 = zeros(size(data3,1),2*maxcol+1);
for i = 1 : size(data3,1)
    tmp = sscanf(data3{i,1},'%f')';
    class3(i,1:size(tmp,2)) = tmp;
end
class3 = class3(:,2:size(class3,2));

size_class1_train = floor(0.7 * size(class1,1));
size_class1_val = floor(0.1 * size(class1,1));
size_class1_test = size(class1,1) - size_class1_train - size_class1_val;
class1_train = class1(1:size_class1_train,:);
class1_val = class1(size_class1_train+1:size_class1_train+size_class1_val,:);
class1_test = class1(size_class1_train+size_class1_val+1:size(class1,1),:);

size_class2_train = floor(0.7 * size(class2,1));
size_class2_val = floor(0.1 * size(class2,1));
size_class2_test = size(class2,1) - size_class2_train - size_class2_val;
class2_train = class2(1:size_class2_train,:);
class2_val = class2(size_class2_train+1:size_class2_train+size_class2_val,:);
class2_test = class2(size_class2_train+size_class2_val+1:size(class2,1),:);

size_class3_train = floor(0.7 * size(class3,1));
size_class3_val = floor(0.1 * size(class3,1));
size_class3_test = size(class3,1) - size_class3_train - size_class3_val;
class3_train = class3(1:size_class3_train,:);
class3_val = class3(size_class3_train+1:size_class3_train+size_class3_val,:);
class3_test = class3(size_class3_train+size_class3_val+1:size(class3,1),:);

train1_label = ones(size(class1_train,1),1);
train2_label = 2.*ones(size(class2_train,1),1);
train3_label = 3.*ones(size(class3_train,1),1);
train_data = [class1_train;class2_train;class3_train];
train_label = [train1_label;train2_label;train3_label];

val1_label = ones(size(class1_val,1),1);
val2_label = 2.*ones(size(class2_val,1),1);
val3_label = 3.*ones(size(class3_val,1),1);
val_data = [class1_val;class2_val;class3_val];
val_label = [val1_label;val2_label;val3_label];

test1_label = ones(size(class1_test,1),1);
test2_label = 2.*ones(size(class2_test,1),1);
test3_label = 3.*ones(size(class3_test,1),1);
test_data = [class1_test;class2_test;class3_test];
test_label = [test1_label;test2_label;test3_label];

model = svmtrain(train_label,train_data,'-s 0 -t 0 -c 1');

[train_predict_label, accuracy_train, train_prob_values] = svmpredict(train_label, train_data, model);
[val_predict_label, accuracy_val, val_prob_values] = svmpredict(val_label, val_data, model);
[test_predict_label,accuracy_test, test_prob_values] = svmpredict(test_label, test_data, model);

acc_train = 0;
acc_val = 0;
acc_test = 0;
confusion_mat_train = zeros(3,3);
for i=1:size(train_label,1)
    row = train_label(i);
    col = train_predict_label(i);
    if row==col
        acc_train = acc_train +1;
    end
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_val = zeros(3,3);
for i=1:size(val_label,1)
    row = val_label(i);
    col = val_predict_label(i);
    if row==col
        acc_val = acc_val +1;
    end
    confusion_mat_val(row,col) = confusion_mat_val(row,col) +1; 
end

confusion_mat_test = zeros(3,3);
for i=1:size(test_label,1)
    row = test_label(i);
    col = test_predict_label(i);
    if row==col
        acc_test = acc_test +1;
    end
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end

acc_train = acc_train*100/size(train_label,1);
acc_val = acc_val*100/size(val_label,1);
acc_test = acc_test*100/size(test_label,1);