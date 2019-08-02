
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


total_train_data = [train1_data ; train2_data ; train3_data];

mean_total = mean(total_train_data);
cov_total = cov(total_train_data);

[eig_vectors , eig_val] = eig(cov_total);
sotred_eig_vectors = sortem(eig_vectors,eig_val);
eigen_two_vect = sotred_eig_vectors(:,[1:2]) ;

newdata1 = total_train_data *eigen_two_vect ;

total_train_data =  newdata1;

%}


%%%%%%%%%%
k = 30;%30
%%%%%%%%%%
%%%%%%%%%%
s  = 17;%7
%%%%%%%%%%
eta  = 0.00001; %0.00001;

[idx , mu] = kmeans(newdata1 , k , 'MaxIter' , 500) ;

size_train = size(newdata1);
w = rand(k,3,1);
sizeforw = size(w);
wphi = ones(size_train(1,1),k);

for c = 1:k
    for r = 1:size_train(1,1)
      sm = norm(newdata1(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      wphi(r,c) = exp(-power);
    end
end

%a= wphi*w ;

y = zeros(size_train(1,1),3);

sum1 = zeros(sizeforw(1,1),1);
sum2 = zeros(sizeforw(1,1),1);
sum3 = zeros(sizeforw(1,1),1);

train1_size = size(train1_data);
train2_size = size(train2_data);
train3_size = size(train3_data);
train_size = size(total_train_data);
thershold = 0.01;


count=0;
while count < 10000    %ther > threshold %1500
   % count
    %eta = 1/(31*count);
    count = count + 1;
    a= wphi*w ;
     for n = 1:train_size(1,1)
        sum = exp(a(n,1))+exp(a(n,2))+exp(a(n,3));
        y(n,1) = exp(a(n,1))/sum;
        y(n,2) = exp(a(n,2))/sum;
        y(n,3) = exp(a(n,3))/sum;
     end
         
      for i = 1:train1_size(1,1)
        sum1 = sum1 + (y(i,1)-1)*(wphi(i,:)');
      end
      for i = train1_size(1,1)+1 : train1_size(1,1) + train2_size(1,1)
        sum1 = sum1 + y(i,1)*(wphi(i,:)');
      end
      for  i = train1_size(1,1)+ train2_size(1,1)+1 : train_size(1,1) 
        sum1 = sum1 + y(i,1)*(wphi(i,:)');
      end
       for i = 1:train1_size(1,1)
        sum2 = sum2 + (y(i,2))*(wphi(i,:)');
      end
      for i = train1_size(1,1)+1 : train1_size(1,1) + train2_size(1,1)
        sum2 = sum2 + (y(i,2)-1)*(wphi(i,:)');
      end
      for  i = train1_size(1,1)+ train2_size(1,1)+1 : train_size(1,1) 
        sum2 = sum2 + y(i,2)*(wphi(i,:)');
      end
       for i = 1:train1_size(1,1)
        sum3 = sum3 + (y(i,3))*(wphi(i,:)');
      end
      for i = train1_size(1,1)+1 : train1_size(1,1) + train2_size(1,1)
        sum3 = sum3 + (y(i,3))*(wphi(i,:)');
      end
      for  i = train1_size(1,1)+ train2_size(1,1)+1 : train_size(1,1) 
        sum3 = sum3 + (y(i,3)-1)*(wphi(i,:)');
      end
      for j = 1 : sizeforw(1,1) %%%%%%%%%% check this
            w(j,1) = w(j,1) - eta*sum1(j,1);
            w(j,2) = w(j,2) - eta*sum2(j,1);
            w(j,3) = w(j,3) - eta*sum3(j,1);
            
      end
      error = 0;
      for i = 1:train1_size(1,1)
          error = error - log(y(i,1));
      end
      for i = 1:train2_size(1,1)
          error = error - log(y(train1_size(1,1)+i,1));
      end
      for i = 1:train3_size(1,1)
          error = error - log(y(train1_size(1,1)+train1_size(1,1)+i,1));
      end
      %if error <= 0.001
      %    error
     %     break
    %  end
    %error
      


 end



confusion_mat_train = plot_confusion_matrix(train_size,train1_size,train2_size,train3_size,y);
 
  accurately_classified = confusion_mat_train(1,1)+confusion_mat_train(2,2)+confusion_mat_train(3,3);
accuracy_total = train_size(1,1)/36;
accuracy_train = (accurately_classified/accuracy_total)*100;




totalvalid = [valid1_data ; valid2_data ; valid3_data];
newvaliddata = totalvalid * eigen_two_vect ;

totaltest = [test1_data ; test2_data ; test3_data];
newtestdata = totaltest * eigen_two_vect ;

size_val = size(newvaliddata);
size_test = size(newtestdata);
size1_val = size(valid1_data);
size2_val = size(valid2_data);
size3_val = size(valid3_data);

sizeforw_val = size(w);
w_val_phi = ones(size_val(1,1),k);
sizeforw_test= size(w);
w_test_phi = ones(size_test(1,1),k);

for c = 1:k
    for r = 1:size_val(1,1)
      sm = norm(newvaliddata(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      w_val_phi(r,c) = exp(-power);
    end
end

for c = 1:k
    for r = 1:size_test(1,1)
      sm = norm(newtestdata(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      w_test_phi(r,c) = exp(-power);
    end
end

a_val = w_val_phi * w;
y_val = zeros(size_val(1,1),3);
a_test = w_test_phi * w;
y_test = zeros(size_test(1,1),3);

     for n = 1:size_val(1,1)
        sum = exp(a_val(n,1))+exp(a_val(n,2))+exp(a_val(n,3));
        y_val(n,1) = exp(a_val(n,1))/sum;
        y_val(n,2) = exp(a_val(n,2))/sum;
        y_val(n,3) = exp(a_val(n,3))/sum;
     end
test1_test = size(test1_data);
test2_val = size(test2_data);
test3_val = size(test3_data);

confusion_mat_val = plot_confusion_matrix(size_val(1,1),size1_val(1,1),size2_val(1,1),size3_val(1,1),y_val);

accurately_classified_val = confusion_mat_val(1,1)+confusion_mat_val(2,2)+confusion_mat_val(3,3);
accuracy_valid = (accurately_classified_val/(size_val(1,1)/36))*100;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%testdata$$$$$$$$$$$$$$$$$$$$$$$$
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%

     for n = 1:size_test(1,1)
        sum = exp(a_test(n,1))+exp(a_test(n,2))+exp(a_test(n,3));
        y_test(n,1) = exp(a_test(n,1))/sum;
        y_test(n,2) = exp(a_test(n,2))/sum;
        y_test(n,3) = exp(a_test(n,3))/sum;
     end
size1_test = size(test1_data);
size2_test = size(test2_data);
size3_test = size(test3_data);
size_test = size(newtestdata) ;

confusion_mat_test = plot_confusion_matrix(size_test(1,1),size1_test(1,1),size2_test(1,1),size3_test(1,1),y_test);

accurately_classified_test = confusion_mat_test(1,1)+confusion_mat_test(2,2)+confusion_mat_test(3,3);
accuracy_test = (accurately_classified_test/(size_test(1,1)/36))*100;








function matrix = plot_confusion_matrix(total_data_train,train_size1,train_size2,train_size3,y)
    
    data = total_data_train/36;
    train_size1 = train_size1/36;
    train_size2 = train_size2/36;
    train_size3 = train_size3/36;
   % y=[y1;y2;y3];
   
    confusion_mat_train = zeros(3,3);
    
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