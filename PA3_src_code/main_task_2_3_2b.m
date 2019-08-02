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

train1_label = [ones(size(class1_train,1),1) zeros(size(class1_train,1),1)];
train2_label = [zeros(size(class2_train,1),1) ones(size(class2_train,1),1)];
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

val1_label = [ones(size(class1_val,1),1) zeros(size(class1_val,1),1)];
val2_label = [zeros(size(class2_val,1),1) ones(size(class2_val,1),1)];
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

test1_label = [ones(size(class1_test,1),1) zeros(size(class1_test,1),1)];
test2_label = [zeros(size(class2_test,1),1) ones(size(class2_test,1),1)];
test_data = [class1_test;class2_test];
test_label = [test1_label;test2_label];


train_data = train_data';
train_label = train_label';
net = patternnet([8 7],'traingdm');
net.trainParam.epochs = 10000;
net.divideFcn = 'dividetrain';
%net.trainParam.mc = 0.9;
%net.trainParam.min_grad = 0;
%net.trainParam.goal = 0;
net = train(net,train_data,train_label);
iw = net.IW;
lw = net.LW;
b = net.b;

view(net)

train_predict = net(train_data);
%perf_train = perform(net,train_label,train_predict);
train_predict_label = vec2ind(train_predict);

val_data = val_data';
val_label = val_label';
val_predict = net(val_data);
val_predict_label = vec2ind(val_predict);

test_data = test_data';
test_label = test_label';
test_predict = net(test_data);
test_predict_label = vec2ind(test_predict);
%{
IW = cell2mat(net.IW);
b1 = net.b{1};
h1 = tansig(IW*train_data + repmat( b1,1, 500));
LW1 = net.LW{2,1};
b2 = net.b{2};
h2 = tansig(LW1*h1 + repmat( b2,1, 500));
IW = net.LW{3,2};
b3 = net.b{3};
output = softmax(LW2*h2 + repmat( b3,1, 500));
surf(train_data(1,:),train_data(2,:),repmat(output(1,:),500,1));
%}
%{
groundTruth = train_label';
d = train_data';

figure

% plot training data
hold on;

pos = find(groundTruth(:,1)==1);
scatter(d(pos,1), d(pos,2), 'r')
pos = find(groundTruth(:,2)==1);
scatter(d(pos,1), d(pos,2), 'b')

% now plot decision area
lp = min([d(:,1);d(:,2)]);
rp = max([d(:,1);d(:,2)]);

[xi,yi] = meshgrid([lp:0.01:rp],[lp:0.01:rp]);
dd = [xi(:),yi(:)];
tic;
plot_predict = net(dd');
predicted_label = vec2ind(plot_predict);
toc
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
%}
acc_train = 0;
acc_val = 0;
acc_test = 0;
confusion_mat_train = zeros(2,2);
for i=1:size(train_label,2)
    row = find(train_label(:,i)==1);
    col = train_predict_label(i);
    if row==col
        acc_train = acc_train +1;
    end
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_val = zeros(2,2);
for i=1:size(val_label,2)
    row = find(val_label(:,i)==1);
    col = val_predict_label(i);
    if row==col
        acc_val = acc_val +1;
    end
    confusion_mat_val(row,col) = confusion_mat_val(row,col) +1; 
end

confusion_mat_test = zeros(2,2);
for i=1:size(test_label,2)
    row = find(test_label(:,i)==1);
    col = test_predict_label(i);
    if row==col
        acc_test = acc_test +1;
    end
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end

acc_train = acc_train*100/size(train_label,2);
acc_val = acc_val*100/size(val_label,2);
acc_test = acc_test*100/size(test_label,2);