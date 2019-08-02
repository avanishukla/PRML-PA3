
clc;
clearvars;
mydir = 'C:\Users\Arti\Downloads\HandWritten_data\DATA\FeaturesHW\';

fid = fopen(strcat(mydir,'ai.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);

class1 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    class1{i,1} = C{1,1}{3*i,1}; 
end

fid = fopen(strcat(mydir,'chA.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
class2 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    class2{i,1} = C{1,1}{3*i,1}; 
end

fid = fopen(strcat(mydir,'lA.ldf'),'r');
C = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
class3 = cell(size(C{1,1},1)/3,1);
for i = 1 : size(C{1,1},1)/3
    class3{i,1} = C{1,1}{3*i,1}; 
end

size1 = size(class1);
size2 = size(class2);
size3 = size(class3);

size_class1_train = floor(0.6 * size(class1,1));
size_class1_test = size(class1,1) - size_class1_train;
class1_train = class1(1:size_class1_train,:);
class1_test = class1(size_class1_train+1:size(class1,1),:);

size_class2_train = floor(0.6 * size(class2,1));
size_class2_test = size(class2,1) - size_class2_train ;
class2_train = class2(1:size_class2_train,:);
class2_test = class2(size_class2_train+1:size(class2,1),:);

size_class3_train = floor(0.6 * size(class3,1));
size_class3_test = size(class3,1) - size_class3_train;
class3_train = class3(1:size_class3_train,:);
class3_test = class3(size_class3_train+1:size(class3,1),:);


sum = 0;
newtraindata1 = [];
sizes1 = [];

for z = 1:size_class1_train(1,1)
    mat = str2num(class1_train{z,1}); 
    k=1;
    sizes1 = [sizes1 ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtraindata1 = [newtraindata1 ; data ];
end

sum = 0;
newtraindata2 = [];
sizes2 = [];
for z = 1:size_class2_train(1,1)
    mat = str2num(class2_train{z,1}); 
    k=1;
    sizes2 = [sizes2 ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtraindata2 = [newtraindata2 ; data ];
end

sum = 0;
newtraindata3 = [];
sizes3 = [];
for z = 1:size_class3_train(1,1)
    mat = str2num(class3_train{z,1}); 
    k=1;
    sizes3 = [sizes3 ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtraindata3 = [newtraindata3 ; data ];
end

sum = 0;
newtestdata1 = [];
sizes1t = [];

for z = 1:size_class1_test(1,1)
    mat = str2num(class1_test{z,1}); 
    k=1;
    sizes1t = [sizes1t ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtestdata1 = [newtestdata1 ; data ];
end

sum = 0;
newtestdata2 = [];
sizes2t  = [];
for z = 1:size_class2_test(1,1)
    mat = str2num(class2_test{z,1}); 
    k=1;
    sizes2t = [sizes2t ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtestdata2 = [newtestdata2 ; data ];
end

sum = 0;
newtestdata3 = [];
sizes3t = [];
for z = 1:size_class3_test(1,1)
    mat = str2num(class3_test{z,1}); 
    k=1;
    sizes3t = [sizes3t ; mat(1,k)];
    k=2;
    data = zeros(mat(1,1), 2);
    sum = sum +mat(1,1) ;
    for i = 1: mat(1,1)
        for j= 1:2
            data(i,j) = mat(1,k);
            k = k+1;
        end
    end
    newtestdata3 = [newtestdata3 ; data ];
end

khyperparameter = 13;

   
    [idx1 , mu1] = kmeans(newtraindata1 , khyperparameter , 'MaxIter' , 500);
    [idx2 , mu2] = kmeans(newtraindata2 , khyperparameter , 'MaxIter' , 500);
    [idx3 , mu3] = kmeans(newtraindata3 , khyperparameter , 'MaxIter' , 500);

    filename = sprintf('ctrain1_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');
    
     s1 = size(sizes1);
     s2 = size(sizes2);
     k = 1;
    for i =1 : s1(1,1)
        for j = 1 : sizes1(i)
            fprintf(fid,'%d ',idx1(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    filename = sprintf('ctrain2_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');
     
     k = 1;
    for i =1 : s2(1,1)
        for j = 1 : sizes2(i)
            fprintf(fid,'%d ',idx2(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    filename = sprintf('ctrain3_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');
    s3 = size(sizes3);

     k = 1;
    for i =1 : s3(1,1)
        for j = 1 : sizes3(i)
            fprintf(fid,'%d ',idx3(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    [idx1t , mu1t] = kmeans(newtestdata1 , khyperparameter , 'MaxIter' , 500);
    [idx2t , mu2t] = kmeans(newtestdata2 , khyperparameter , 'MaxIter' , 500);
    [idx3t , mu3t] = kmeans(newtestdata3 , khyperparameter , 'MaxIter' , 500);

    s1 = size(sizes1t);
    filename = sprintf('ctest1_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');

     k = 1;
    for i =1 : s1(1,1)
        for j = 1 : sizes1t(i)
            fprintf(fid,'%d ',idx1t(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    filename = sprintf('ctest2_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');
    s2 = size(sizes2t);

     k = 1;
    for i =1 : s2(1,1)
        for j = 1 : sizes2t(i)
            fprintf(fid,'%d ',idx2t(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    filename = sprintf('ctest3_%d.seq',khyperparameter);
    fid = fopen(filename , 'w');
     s3t = size(sizes3t);

     k = 1;
    for i =1 : s3t(1,1)
        for j = 1 : sizes3t(i)
            fprintf(fid,'%d ',idx3t(k));
            k = k+1;
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    