clc;
clearvars;
mydir = 'D:\prml\PRML2018_assignment_3\nonlinearly_separable\';

fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
class1_train = fscanf(fileID,'%f',[2 Inf]);
class1_train = class1_train';
fclose(fileID);
fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
class2_train = fscanf(fileID,'%f',[2 Inf]);
class2_train = class2_train';
fclose(fileID);
train1_label = ones(size(class1_train,1),1);
train2_label = 2.*ones(size(class2_train,1),1);
train_data = [class1_train;class2_train];
train_label = [train1_label;train2_label];

fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
class1_val = fscanf(fileID,'%f',[2 Inf]);
class1_val = class1_val';
fclose(fileID);
fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
class2_val = fscanf(fileID,'%f',[2 Inf]);
class2_val = class2_val';
fclose(fileID);
val1_label = ones(size(class1_val,1),1);
val2_label = 2.*ones(size(class2_val,1),1);
val_data = [class1_val;class2_val];
val_label = [val1_label;val2_label];

fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
class1_test = fscanf(fileID,'%f',[2 Inf]);
class1_test = class1_test';
fclose(fileID);
fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
class2_test = fscanf(fileID,'%f',[2 Inf]);
class2_test = class2_test';
fclose(fileID);
test1_label = ones(size(class1_test,1),1);
test2_label = 2.*ones(size(class2_test,1),1);
test_data = [class1_test;class2_test];
test_label = [test1_label;test2_label];

%train_data = zscore(train_data);
%val_data = zscore(val_data);
%test_data = zscore(test_data);

%scatter(class1_train(:,1),class1_train(:,2),'m');
%hold on;
%scatter(class2_train(:,1),class2_train(:,2),'g');
%hold off;

c=10;
%model = svmtrain(train_label,train_data,'-s 0 -t 1 -c 10 -d 3 -g 2');  %polynomial
model = svmtrain(train_label,train_data,'-s 0 -t 2 -c 10 -g 0.5');  %gaussian

w = model.SVs' * model.sv_coef;
b = -model.rho;
[train_predict_label, train_accuracy, train_prob_values] = svmpredict(train_label, train_data, model);
[val_predict_label, val_accuracy, val_prob_values] = svmpredict(val_label, val_data, model);
[test_predict_label, test_accuracy, test_prob_values] = svmpredict(test_label, test_data, model);

groundTruth = train_label;
d = train_data;

figure

% plot training data
hold on;

pos = find(groundTruth==1);
scatter(d(pos,1), d(pos,2), 'r')
pos = find(groundTruth==2);
scatter(d(pos,1), d(pos,2), 'b')

% now plot support vectors
hold on;
sv = full(model.SVs);
bsv = [];
ubsv = [];
for i =1:size(sv,1)
    if abs(model.sv_coef(i,1))==c
        bsv = [bsv;sv(i,:)];
    else
        ubsv = [ubsv;sv(i,:)];
    end
end
if size(bsv,1) ~=0
    plot(bsv(:,1),bsv(:,2),'ko');
end
if size(ubsv,1) ~=0
    plot(ubsv(:,1),ubsv(:,2),'mo');
end

% now plot decision area
lp = min([d(:,1);d(:,2)]);
rp = max([d(:,1);d(:,2)]);

[xi,yi] = meshgrid([lp:0.01:rp],[lp:0.01:rp]);
dd = [xi(:),yi(:)];
tic;[predicted_label, accuracy, decision_values] = svmpredict(zeros(size(dd,1),1), dd, model);toc
pos = find(predicted_label==1);
hold on;
redcolor = [1 0.8 0.8];
bluecolor = [0.8 0.8 1];
h1 = plot(dd(pos,1),dd(pos,2),'s','color',redcolor,'MarkerSize',10,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
pos = find(predicted_label==2);
hold on;
h2 = plot(dd(pos,1),dd(pos,2),'s','color',bluecolor,'MarkerSize',10,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);
uistack(h1, 'bottom');
uistack(h2, 'bottom');

confusion_mat_train = zeros(2,2);
for i=1:size(train_label,1)
    row = train_label(i);
    col = train_predict_label(i);
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_test = zeros(2,2);
for i=1:size(test_label,1)
    row = test_label(i);
    col = test_predict_label(i);
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end
