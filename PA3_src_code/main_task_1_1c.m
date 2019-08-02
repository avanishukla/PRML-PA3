%poly basis fn

clc;
clearvars;
mydir = '/home/neha/workspace/data_assign3_group8/';

%class1--------------------------------------------------------------------------------
class1_path = strcat(mydir,'highway');
foldername = fullfile(class1_path);
pth = genpath(foldername);
pathTest = regexp([pth ':'],'(.*?):','tokens');
list = dir([pathTest{1,1}{:} '/*.jpg_color_edh_entropy']);
%list = list(1:10);
class1_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[23 Inf]);
           fileread = fileread';
        class1_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class1 = floor(0.7*size(class1_cell,1));
size_validation_class1 = floor(0.1*size(class1_cell,1));
size_test_class1 = size(class1_cell,1) - size_train_class1 - size_validation_class1;
train1_cell = class1_cell(1:size_train_class1,1);
validation1_cell = class1_cell(size_train_class1+1:size_train_class1+size_validation_class1,1);
test1_cell = class1_cell(size_train_class1+size_validation_class1+1:size_train_class1+size_validation_class1+size_test_class1,1);

train1_data = cell2mat(train1_cell);
test1_data = cell2mat(test1_cell);
valid1_data = cell2mat(validation1_cell);

%class2--------------------------------------------------------------------------------
class2_path = strcat(mydir,'tallbuilding');
foldername = fullfile(class2_path);
pth = genpath(foldername);
pathTest = regexp([pth ':'],'(.*?):','tokens');
list = dir([pathTest{1,1}{:} '/*.jpg_color_edh_entropy']);

%list = list(1:10);

class2_cell = cell(numel(list),1);

if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[23 Inf]);
           fileread = fileread';
        class2_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class2 = floor(0.7*size(class2_cell,1));
size_validation_class2 = floor(0.1*size(class2_cell,1));
size_test_class2 = size(class2_cell,1) - size_train_class2 - size_validation_class2;
train2_cell = class2_cell(1:size_train_class2,1);
validation1_cell = class2_cell(size_train_class2+1:size_train_class2+size_validation_class2,1);
test2_cell = class2_cell(size_train_class2+size_validation_class2+1:size_train_class2+size_validation_class2+size_test_class2,1);

train2_data = cell2mat(train2_cell);
test2_data = cell2mat(test2_cell);
valid2_data = cell2mat(validation1_cell);

%class3--------------------------------------------------------------------------------
class3_path = strcat(mydir,'street');
foldername = fullfile(class3_path);
pth = genpath(foldername);
pathTest = regexp([pth ':'],'(.*?):','tokens');
list = dir([pathTest{1,1}{:} '/*.jpg_color_edh_entropy']);
%list = list(1:10);
class3_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[23 Inf]);
           fileread = fileread';
        class3_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class3 = floor(0.7*size(class3_cell,1));
size_validation_class3 = floor(0.1*size(class3_cell,1));
size_test_class3 = size(class3_cell,1) - size_train_class3 - size_validation_class3;
train3_cell = class3_cell(1:size_train_class3,1);
validation3_cell = class3_cell(size_train_class3 + 1:size_train_class3 + size_validation_class3,1);
test3_cell = class3_cell(size_train_class3 +size_validation_class3+1:size_train_class3 + size_validation_class3+size_test_class3,1);

train3_data = cell2mat(train3_cell);
test3_data = cell2mat(test3_cell);
valid3_data = cell2mat(validation3_cell);


total_train_data = [train1_data ; train2_data ;train3_data];

mean_total = mean(total_train_data);
cov_total = cov(total_train_data);

[eig_vectors , eig_val] = eig(cov_total);
sotred_eig_vectors = sortem(eig_vectors,eig_val);
eigen_two_vect = sotred_eig_vectors(:,[1:2]) ;

newdata1 = total_train_data *eigen_two_vect ;
size_train = size(newdata1);

train1_size = size(train1_data);
train2_size = size(train2_data);
train3_size = size(train3_data);


%%%%%%%%%%
k=2 ;
%%%%%%%%%%
deg = 19;
eta  = 0.0001;
%%%%%%%%%%
s  = 11;
%%%%%%%%%%
mon_mat = createmonomials(k, deg);

sizeforw = size(mon_mat);

R = [-1 1];
w1 = rand(sizeforw(1),1)*range(R)+min(R);
w2 = rand(sizeforw(1),1)*range(R)+min(R);
w3 = rand(sizeforw(1),1)*range(R)+min(R);

wphi1 = ones(train1_size(1,1),k);

newtrain1 = newdata1(1:train1_size, :);
newtrain2 = newdata1(train1_size+1:train1_size+train2_size, :);
newtrain3 = newdata1(train1_size+train2_size+1:size_train, :);

design_mat1 = [];
for i = 1 :  train1_size % 10 ke jagah a1
    vec = newtrain1(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat1 = [design_mat1;ad];
end
design_mat2 = [];
for i = 1 :  train2_size % 10 ke jagah a1
    vec = newtrain2(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat2 = [design_mat2;ad];
end
design_mat3 = [];
for i = 1 :  train3_size % 10 ke jagah a1
    vec = newtrain3(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat3 = [design_mat3;ad];
end



a1  = zeros(train1_size(1,1),3);
a2  = zeros(train2_size(1,1),3);
a3  = zeros(train3_size(1,1),3);

    y1 = zeros(train1_size(1,1),3);
    y2 = zeros(train2_size(1,1),3);
    y3 = zeros(train3_size(1,1),3);

for i=1:5000
  
    for n = 1:train1_size(1,1)
        a1(n,1) = w1' * design_mat1(n,:)';
        a1(n,2) = w2' * design_mat1(n,:)';
        a1(n,3) = w3' * design_mat1(n,:)';
    end
    for n = 1:train2_size(1,1)
        a2(n,1) = w1' * design_mat2(n,:)';
        a2(n,2) = w2' * design_mat2(n,:)';
        a2(n,3) = w3' * design_mat2(n,:)';
    end
    for n = 1:train3_size(1,1)
        a3(n,1) = w1' * design_mat3(n,:)';
        a3(n,2) = w2' * design_mat3(n,:)';
        a3(n,3) = w3' * design_mat3(n,:)';
    end
    
    
    for n = 1:train1_size(1,1)
        sum = exp(a1(n,1))+exp(a1(n,2))+exp(a1(n,3));
        
        y1(n,1) = exp(a1(n,1))/sum;
        y1(n,2) = exp(a1(n,2))/sum;
        y1(n,3) = exp(a1(n,3))/sum;
        if isnan(y1(n,1))
            y1(n,1)  = 0;
        end
        if isnan(y1(n,2))
            y1(n,2)  = 0;
        end
        if isnan(y1(n,3))
            y1(n,3)  = 0;
        end
    end
    for n = 1:train2_size(1,1)
        
        sum = exp(a2(n,1))+exp(a2(n,2))+exp(a2(n,3));
        
        y2(n,1) = exp(a2(n,1))/sum;
        y2(n,2) = exp(a2(n,2))/sum;
        y2(n,3) = exp(a2(n,3))/sum;
        if isnan(y2(n,1))
            y2(n,1)  = 0;
        end
        if isnan(y2(n,2))
            y2(n,2)  = 0;
        end
        if isnan(y2(n,3))
            y2(n,3)  = 0;
        end
    end
    for n = 1:train3_size(1,1)
        sum = exp(a3(n,1))+exp(a3(n,2))+exp(a3(n,3));
        
        y3(n,1) = exp(a3(n,1))/sum;
        y3(n,2) = exp(a3(n,2))/sum;
        y3(n,3) = exp(a3(n,3))/sum;
        if isnan(y3(n,1))
            y3(n,1)  = 0;
        end
        if isnan(y3(n,2))
            y3(n,2)  = 0;
        end
        if isnan(y3(n,3))
            y3(n,3)  = 1;
        end
    end
    sum1 = zeros(sizeforw(1),1);
    sum2 = zeros(sizeforw(1),1);
    sum3 = zeros(sizeforw(1),1);    
    
     for n = 1:train1_size(1,1)
        sum1 = sum1 + (y1(n,1)-1)*(design_mat1(n,:)');
        sum2 = sum2 + y1(n,2)*(design_mat1(n,:)');
        sum3 = sum3 + y1(n,3)*(design_mat1(n,:)');
     end
     for n = 1:train2_size(1,1)
        sum1= sum1 + y2(n,1)*(design_mat2(n,:)');
        sum2 = sum2 + (y2(n,2)-1)*(design_mat2(n,:)');
        sum3 = sum3 + y2(n,3)*(design_mat2(n,:)');
     end
      for n = 1:train3_size(1,1)
        sum1 = sum1 + y3(n,1)*(design_mat3(n,:)');
        sum2 = sum2 + y3(n,2)*(design_mat3(n,:)');
        sum3 = sum3 + (y3(n,3)-1)*(design_mat3(n,:)');
      end
      
    w1 = w1 - eta*sum1;
    w2 = w2 - eta*sum2;
    w3 = w3 - eta*sum3;
    
    error = 0;
    for n = 1:train1_size(1,1)
        error = error - y1(n,1);
    end
    for n = 1:train2_size(1,1)
        error = error - y2(n,2);
    end
    for n = 1:train3_size(1,1)
        error = error - y3(n,3);
    end
   % error
 
    
end

confusion_mat_train = plot_confusion_matrix(size_train,train1_size,train2_size,train3_size,y1,y2,y3);
      % confusion_mat_train 

 
accurately_classified = confusion_mat_train(1,1)+confusion_mat_train(2,2)+confusion_mat_train(3,3);
accuracy_train = (accurately_classified/(size_train(1,1)/36))*100;



totalvalid = [valid1_data ; valid2_data ; valid3_data];
newvaliddata = totalvalid * eigen_two_vect ;

totaltest = [test1_data ; test2_data ; test3_data];
newtestdata = totaltest * eigen_two_vect ;

size_val = size(newvaliddata);

size1_val = size(valid1_data);
size2_val = size(valid2_data);
size3_val = size(valid3_data);


design_mat_val1 = [];
for i = 1 :  size1_val(1,1) % 10 ke jagah a1
    vec = valid1_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val1 = [design_mat_val1;ad];
end

design_mat_val2 = [];
for i = 1 :  size2_val(1,1) % 10 ke jagah a1
    vec = valid2_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val2 = [design_mat_val2;ad];
end


design_mat_val3 = [];
for i = 1 :  size3_val(1,1) % 10 ke jagah a1
    vec = valid3_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val3 = [design_mat_val3;ad];
end
val_a1  = zeros(size1_val(1,1),3);
val_a2  = zeros(size2_val(1,1),3);
val_a3  = zeros(size3_val(1,1),3);

for n = 1:size1_val(1,1)
        val_a1(n,1) = w1' * design_mat1(n,:)';
        val_a1(n,2) = w2' * design_mat1(n,:)';
        val_a1(n,3) = w3' * design_mat1(n,:)';
end
    for n = 1:size2_val(1,1)
        val_a2(n,1) = w1' * design_mat2(n,:)';
        val_a2(n,2) = w2' * design_mat2(n,:)';
        val_a2(n,3) = w3' * design_mat2(n,:)';
    end
    for n = 1:size3_val(1,1)
        val_a3(n,1) = w1' * design_mat3(n,:)';
        val_a3(n,2) = w2' * design_mat3(n,:)';
        val_a3(n,3) = w3' * design_mat3(n,:)';
    end
    
    y1_val = zeros(size1_val(1,1),3);
    y2_val = zeros(size2_val(1,1),3);
    y3_val = zeros(size3_val(1,1),3);
    
    for n = 1:size1_val(1,1)
        sum = exp(val_a1(n,1))+exp(val_a1(n,2))+exp(val_a1(n,3));
        y1_val(n,1) = exp(val_a1(n,1))/sum;
        y1_val(n,2) = exp(val_a1(n,2))/sum;
        y1_val(n,3) = exp(val_a1(n,3))/sum;
    end
    for n = 1:size2_val(1,1)
        sum = exp(val_a2(n,1))+exp(val_a2(n,2))+exp(val_a2(n,3));
        y2_val(n,1) = exp(val_a2(n,1))/sum;
        y2_val(n,2) = exp(val_a2(n,2))/sum;
        y2_val(n,3) = exp(val_a2(n,3))/sum;
    end
    for n = 1:size3_val(1,1)
        sum = exp(val_a3(n,1))+exp(val_a3(n,2))+exp(val_a3(n,3));
        y3_val(n,1) = exp(val_a3(n,1))/sum;
        y3_val(n,2) = exp(val_a3(n,2))/sum;
        y3_val(n,3) = exp(val_a3(n,3))/sum;
    end
   
confusion_mat_val = plot_confusion_matrix(size_val,size1_val,size2_val,size3_val,y1_val,y2_val,y3_val);

 
accurately_classified = confusion_mat_val(1,1)+confusion_mat_val(2,2)+confusion_mat_val(3,3);
accuracy_val = (accurately_classified/(size_val(1,1)/36))*100;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

totaltest = [test1_data ; test2_data ; test3_data];
newtestdata = totaltest * eigen_two_vect ;

size_test = size(newtestdata);
size1_test = size(test1_data);
size2_test = size(test2_data);
size3_test = size(test3_data);

design_mat_test1 = [];
for i = 1 :  size1_test(1,1) % 10 ke jagah a1
    vec = test1_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test1 = [design_mat_test1;ad];
end

design_mat_test2 = [];
for i = 1 :  size2_test(1,1) % 10 ke jagah a1
    vec = test2_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test2 = [design_mat_test2;ad];
end


design_mat_test3 = [];
for i = 1 :  size3_val(1,1) % 10 ke jagah a1
    vec = test3_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test3 = [design_mat_test3;ad];
end
test_a1  = zeros(size1_test(1,1),3);
test_a2  = zeros(size2_test(1,1),3);
test_a3  = zeros(size3_test(1,1),3);

for n = 1:size1_val(1,1)
        test_a1(n,1) = w1' * design_mat_test1(n,:)';
        test_a1(n,2) = w2' * design_mat_test1(n,:)';
        test_a1(n,3) = w3' * design_mat_test1(n,:)';
end
    for n = 1:size2_val(1,1)
        test_a2(n,1) = w1' * design_mat_test2(n,:)';
        test_a2(n,2) = w2' * design_mat_test2(n,:)';
        test_a2(n,3) = w3' * design_mat_test2(n,:)';
    end
    for n = 1:size3_val(1,1)
        test_a3(n,1) = w1' * design_mat_test3(n,:)';
        test_a3(n,2) = w2' * design_mat_test3(n,:)';
        test_a3(n,3) = w3' * design_mat_test3(n,:)';
    end
    
    y1_test = zeros(size1_test(1,1),3);
    y2_test = zeros(size2_test(1,1),3);
    y3_test = zeros(size3_test(1,1),3);
    
    for n = 1:size1_test(1,1)
        sum = exp(test_a1(n,1))+exp(test_a1(n,2))+exp(test_a1(n,3));
        y1_test(n,1) = exp(test_a1(n,1))/sum;
        y1_test(n,2) = exp(test_a1(n,2))/sum;
        y1_test(n,3) = exp(test_a1(n,3))/sum;
    end
    for n = 1:size2_test(1,1)
        sum = exp(test_a2(n,1))+exp(test_a2(n,2))+exp(test_a2(n,3));
        y2_test(n,1) = exp(test_a2(n,1))/sum;
        y2_test(n,2) = exp(test_a2(n,2))/sum;
        y2_test(n,3) = exp(test_a2(n,3))/sum;
    end
    for n = 1:size3_test(1,1)
        sum = exp(test_a3(n,1))+exp(test_a3(n,2))+exp(test_a3(n,3));
        y3_test(n,1) = exp(test_a3(n,1))/sum;
        y3_test(n,2) = exp(test_a3(n,2))/sum;
        y3_test(n,3) = exp(test_a3(n,3))/sum;
    end
   
confusion_mat_test = plot_confusion_matrix(size_val,size1_val,size2_val,size3_val,y1_test,y2_test,y3_test);

accurately_classified = confusion_mat_test(1,1)+confusion_mat_test(2,2)+confusion_mat_test(3,3);
accuracy_test = (accurately_classified/(size_test(1,1)/36))*100;





function matrix = plot_confusion_matrix(total_data_train,train_size1,train_size2,train_size3,y1,y2,y3)
    
    data = total_data_train/36;
    train_size1 = train_size1/36;
    train_size2 = train_size2/36;
    train_size3 = train_size3/36;
    confusion_mat_train = zeros(3,3);
    y = [y1;y2;y3];
    
    for index=1:data
        p_c1 = 0;
        p_c2 = 0;
        p_c3 = 0;
        st = 1 +(index-1)*36;
        for row =st:st+35
            ll = max(max(y(row,1),y(row,2)),y(row,3));
            if ll == y(row,1)
                p_c1 = p_c1+1;
            elseif ll == y(row,2)
                 p_c2 = p_c2+1;
             else
                 p_c3 = p_c3 + 1;
            end
        
        end
        
        c1 = p_c1;
        c2 = p_c2;
        c3 = p_c3;
    %{
    for index = 1:total_data_train
        c1 = y(index,1) ;
        c2 = y(index,2) ;
        c3 = y(index,3) ;
        %}
       label_count = max(max(c1,c2),c3);
        if index <= train_size1(1,1)
            if label_count == c1
                confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
            elseif label_count == c2
                confusion_mat_train(1,2) = confusion_mat_train(1,2)+1;
            elseif label_count == c3
                confusion_mat_train(1,3) = confusion_mat_train(1,3)+1;
            end
        
        elseif index > train_size1(1,1) && index <= train_size1(1,1)+train_size2(1,1)
            if label_count == c2
                confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
            elseif label_count == c1
                confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
            elseif label_count == c3
                confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
            end
        else
            if label_count == c3
                confusion_mat_train(3,3) = confusion_mat_train(3,3)+1;
            elseif label_count == c2
                confusion_mat_train(3,2) = confusion_mat_train(3,2)+1;
            elseif label_count == c1
                confusion_mat_train(3,1) = confusion_mat_train(3,1)+1;
            end
        end   
   end
   
    matrix = confusion_mat_train;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

function [P2,D2]=sortem(P,D)
    D2=diag(sort(diag(D),'descend')); % make diagonal matrix out of sorted diagonal values of input D
    [c, ind]=sort(diag(D),'descend'); % store the indices of which columns the sorted eigenvalues come from
    P2=P(:,ind); % arrange the columns in this order
end

%%%%%%%%%%%%%%%%%%%%%%%%%555
function monomials = createmonomials(k , deg)
    y = zeros(1,k);
    for n = 1:deg
            m = nchoosek(k+n-1 , k-1);
            d = [zeros(m,1) , nchoosek((1 : (k+n-1))' , k-1), ones(m,1)*(k+n )];
            x = diff(d,1,2)-1 ;
            y = [y;x] ;
    end
    monomials = y;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
function design_matrix = calculate_design_matrix_values(data,mon_mat)
    design_matrix = [];
    [r c] = size(mon_mat); 
    %[rows cols] = size(data);
    i=1;
    for j = 1: r
        m =1;
        for k1 = 1 : c
            if mon_mat(j,k1)~= 0
                m= m*data(i,k1)^mon_mat(j,k1);
            end
        end
        design_matrix=[design_matrix;m];
    end
end