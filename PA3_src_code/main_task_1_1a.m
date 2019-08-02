
clc;

clearvars;
mydir = 'C:\Users\Arti\Downloads\linearly_separable\';
fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x1 = A(1:2:end);
y1 = A(2:2:end);
X1 = [x1 y1];

fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x2 = A(1:2:end);
y2 = A(2:2:end);
X2 = [x2 y2];

fileID = fopen(strcat(mydir,'class3_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x3 = A(1:2:end);
y3 = A(2:2:end);
X3 = [x3 y3];

c = vertcat(X1,X2,X3);
%%%%%%%%%%%%%%%%%%%%%%%%5
k = 2;
deg = 1;

eta = 0.000001; %%%%%%%%%%%0.00001
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mon_mat = createmonomials(k, deg);
sizeforw = size(mon_mat);
train1_data = X1;
train2_data = X2;
train3_data = X3;

design_mat1 = [];
for i = 1 :  250 % 10 ke jagah a1
    vec = train1_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat1 = [design_mat1;ad];
end

design_mat2 = [];
for i = 1 : 250 % 10 ke jagah a1
    vec = train2_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat2 = [design_mat2;ad];
end

design_mat3 = [];
for i = 1 :  250 % 10 ke jagah a1
    vec = train3_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat3 = [design_mat3;ad];
end

w1 = rand(sizeforw(1),1);
w2 = rand(sizeforw(1),1);
w3 = rand(sizeforw(1),1);

train_size = size(train1_data);

a1  = zeros(train_size(1,1),3);
a2  = zeros(train_size(1,1),3);
a3  = zeros(train_size(1,1),3);

for iter = 1:200
    iter
    %eta = 1/(300*iter)
    for n = 1:train_size(1,1)
        a1(n,1) = w1' * design_mat1(n,:)';
        a1(n,2) = w2' * design_mat1(n,:)';
        a1(n,3) = w3' * design_mat1(n,:)';
  
        a2(n,1) = w1' * design_mat2(n,:)';
        a2(n,2) = w2' * design_mat2(n,:)';
        a2(n,3) = w3' * design_mat2(n,:)';
        
        a3(n,1) = w1' * design_mat3(n,:)';
        a3(n,2) = w2' * design_mat3(n,:)';
        a3(n,3) = w3' * design_mat3(n,:)';
    end
    
    y1 = zeros(train_size(1,1),3);
    y2 = zeros(train_size(1,1),3);
    y3 = zeros(train_size(1,1),3);
    
    for n = 1:train_size(1,1)
        sum = exp(a1(n,1))+exp(a1(n,2))+exp(a1(n,3));
        y1(n,1) = exp(a1(n,1))/sum;
        y1(n,2) = exp(a1(n,2))/sum;
        y1(n,3) = exp(a1(n,3))/sum;

        sum = exp(a2(n,1))+exp(a2(n,2))+exp(a2(n,3));
        y2(n,1) = exp(a2(n,1))/sum;
        y2(n,2) = exp(a2(n,2))/sum;
        y2(n,3) = exp(a2(n,3))/sum;
   
        sum = exp(a3(n,1))+exp(a3(n,2))+exp(a3(n,3));
        y3(n,1) = exp(a3(n,1))/sum;
        y3(n,2) = exp(a3(n,2))/sum;
        y3(n,3) = exp(a3(n,3))/sum;
    end
    sum1 = zeros(sizeforw(1),1);
    sum2 = zeros(sizeforw(1),1);
    sum3 = zeros(sizeforw(1),1);
    
    for i = 1:train_size(1,1)
        sum1 = sum1 + (y1(i,1)-1)*(design_mat1(i,:)');
        sum1 = sum1 + y2(i,1)*(design_mat2(i,:)');
        sum1 = sum1 + y3(i,1)*(design_mat3(i,:)');
 
        sum2 = sum2 + y1(i,2)*(design_mat1(i,:)');
        sum2 = sum2 + (y2(i,2)-1)*(design_mat2(i,:)');
        sum2 = sum2 + y3(i,2)*(design_mat3(i,:)');

        sum3 = sum3 + y1(i,3)*(design_mat1(i,:)');
        sum3 = sum3 + y2(i,3)*(design_mat2(i,:)');
        sum3 = sum3 + (y3(i,3)-1)*(design_mat3(i,:)');
    end
    w1 = w1 - eta*sum1;
    w2 = w2 - eta*sum2;
    w3 = w3 - eta*sum3;
    n = norm(w1);
    n
end
%%%%%%%%%%%%%%%CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%
confusion_mat_train = zeros(3,3);
for i = 1:train_size(1,1)
    label_count = max(max(y1(i,1),y1(i,2)),y1(i,3));
        if label_count == y1(i,1)
           confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
        elseif label_count == y1(i,2)
           confusion_mat_train(1,2) = confusion_mat_train(1,2)+1;
        elseif label_count == y1(i,3)
            confusion_mat_train(1,3) = confusion_mat_train(1,3)+1;
        end

    label_count = max(max(y2(i,1),y2(i,2)),y2(i,3));
        if label_count == y2(i,2)
           confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
        elseif label_count == y2(i,1)
           confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
        elseif label_count == y2(i,3)
            confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
        end

    label_count = max(max(y3(i,1),y3(i,2)),y3(i,3));
        if label_count == y3(i,3)
           confusion_mat_train(3,3) = confusion_mat_train(3,3)+1;
        elseif label_count == y3(i,2)
           confusion_mat_train(3,2) = confusion_mat_train(3,2)+1;
        elseif label_count == y3(i,1)
            confusion_mat_train(3,1) = confusion_mat_train(3,1)+1;
        end
end

accurately_classified = confusion_mat_train(1,1)+confusion_mat_train(2,2)+confusion_mat_train(3,3);
accuracy_total = train_size(1,1)+train_size(1,1)+train_size(1,1);
accuracy_train = (accurately_classified/accuracy_total)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%555 validation
fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val1 = [A(1:2:end) A(2:2:end)] ;
fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val2 = [A(1:2:end) A(2:2:end)];
fileID = fopen(strcat(mydir,'class3_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val3 = [A(1:2:end) A(2:2:end)];

k=2 ;
deg=2;
val_size = 150;

design_mat_val1 = [];
for i = 1 :  val_size % 10 ke jagah a1
    vec = val1(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val1 = [design_mat_val1;ad];
end

design_mat_val2 = [];
for i = 1 : val_size % 10 ke jagah a1
    vec = val2(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val2 = [design_mat_val2;ad];
end

design_mat_val3 = [];
for i = 1 :  val_size % 10 ke jagah a1
    vec = val3(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_val3 = [design_mat_val3;ad];
end

val_a1  = zeros(train_size(1,1),3);
val_a2  = zeros(train_size(1,1),3);
val_a3  = zeros(train_size(1,1),3);

%%%%%%%%%%%%%%%%%%%%%%55
for n = 1:val_size(1,1)
        val_a1(n,1) = w1' * design_mat_val1(n,:)';
        val_a1(n,2) = w2' * design_mat_val1(n,:)';
        val_a1(n,3) = w3' * design_mat_val1(n,:)';
  
        val_a2(n,1) = w1' * design_mat_val2(n,:)';
        val_a2(n,2) = w2' * design_mat_val2(n,:)';
        val_a2(n,3) = w3' * design_mat_val2(n,:)';
        
        val_a3(n,1) = w1' * design_mat_val3(n,:)';
        val_a3(n,2) = w2' * design_mat_val3(n,:)';
        val_a3(n,3) = w3' * design_mat_val3(n,:)';
end

    
%%%%%%%%%%%%%%%%%%

val_y1 = zeros(val_size(1,1),3);
val_y2 = zeros(val_size(1,1),3);
val_y3 = zeros(val_size(1,1),3);

for n = 1:val_size(1,1)
    sum = exp(val_a1(n,1))+exp(val_a1(n,2))+exp(val_a1(n,3));
    val_y1(n,1) = exp(val_a1(n,1))/sum;
    val_y1(n,2) = exp(val_a1(n,2))/sum;
    val_y1(n,3) = exp(val_a1(n,3))/sum;

    sum = exp(val_a2(n,1))+exp(val_a2(n,2))+exp(val_a2(n,3));
    val_y2(n,1) = exp(val_a2(n,1))/sum;
    val_y2(n,2) = exp(val_a2(n,2))/sum;
    val_y2(n,3) = exp(val_a2(n,3))/sum;

    sum = exp(val_a3(n,1))+exp(val_a3(n,2))+exp(val_a3(n,3));
    val_y3(n,1) = exp(val_a3(n,1))/sum;
    val_y3(n,2) = exp(val_a3(n,2))/sum;
    val_y3(n,3) = exp(val_a3(n,3))/sum;
end
%%%%%%%%%%%%%%%CONFUSION MATRIX VALIDATION %%%%%%%%%%%%%%%%%%%%
confusion_mat_valid = zeros(3,3);
for i = 1:val_size(1,1)
    label_count = max(max(val_y1(i,1),val_y1(i,2)),val_y1(i,3));
        if label_count == val_y1(i,1)
           confusion_mat_valid(1,1) = confusion_mat_valid(1,1)+1;
        elseif label_count == val_y1(i,2)
           confusion_mat_valid(1,2) = confusion_mat_valid(1,2)+1;
        elseif label_count == val_y1(i,3)
            confusion_mat_valid(1,3) = confusion_mat_valid(1,3)+1;
        end

    label_count = max(max(val_y2(i,1),val_y2(i,2)),val_y2(i,3));
        if label_count == val_y2(i,2)
           confusion_mat_valid(2,2) = confusion_mat_valid(2,2)+1;
        elseif label_count == val_y2(i,1)
           confusion_mat_valid(2,1) = confusion_mat_valid(2,1)+1;
        elseif label_count == val_y2(i,3)
            confusion_mat_valid(2,3) = confusion_mat_valid(2,3)+1;
        end

    label_count = max(max(val_y3(i,1),val_y3(i,2)),val_y3(i,3));
        if label_count == val_y3(i,3)
           confusion_mat_valid(3,3) = confusion_mat_valid(3,3)+1;
        elseif label_count == val_y3(i,2)
           confusion_mat_valid(3,2) = confusion_mat_valid(3,2)+1;
        elseif label_count == val_y3(i,1)
            confusion_mat_valid(3,1) = confusion_mat_valid(3,1)+1;
        end
end

accurately_classified_val = confusion_mat_valid(1,1)+confusion_mat_valid(2,2)+confusion_mat_valid(3,3);
accuracy_valid_total =val_size * 3;
accuracy_valid = (accurately_classified_val / accuracy_valid_total)*100;

%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%% test
fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test1 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test2 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'class3_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test3 = [A(1:2:end) A(2:2:end)] ;


test_size = 100;
design_mat_test1 = [];
for i = 1 :  test_size % 10 ke jagah a1
    vec = test1(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test1 = [design_mat_test1;ad];
end

design_mat_test2 = [];
for i = 1 : test_size % 10 ke jagah a1
    vec = test2(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test2 = [design_mat_test2;ad];
end

design_mat_test3 = [];
for i = 1 :  test_size % 10 ke jagah a1
    vec = test3(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    design_mat_test3 = [design_mat_test3;ad];
end

test_a1  = zeros(train_size(1,1),3);
test_a2  = zeros(train_size(1,1),3);
test_a3  = zeros(train_size(1,1),3);

%%%%%%%%%%%%%%%%%%%%55

    for n = 1:test_size(1,1)
        test_a1(n,1) = w1' * design_mat_test1(n,:)';
        test_a1(n,2) = w2' * design_mat_test1(n,:)';
        test_a1(n,3) = w3' * design_mat_test1(n,:)';
        test_a2(n,1) = w1' * design_mat_test2(n,:)';
        test_a2(n,2) = w2' * design_mat_test2(n,:)';
        test_a2(n,3) = w3' * design_mat_test2(n,:)';
        test_a3(n,1) = w1' * design_mat_test3(n,:)';
        test_a3(n,2) = w2' * design_mat_test3(n,:)';
        test_a3(n,3) = w3' * design_mat_test3(n,:)';
    end
    
    test_y1 = zeros(test_size(1,1),3);
    test_y2 = zeros(test_size(1,1),3);
    test_y3 = zeros(test_size(1,1),3);
    
    for n = 1:test_size(1,1)
        sum = exp(test_a1(n,1))+exp(test_a1(n,2))+exp(test_a1(n,3));
        test_y1(n,1) = exp(test_a1(n,1))/sum;
        test_y1(n,2) = exp(test_a1(n,2))/sum;
        test_y1(n,3) = exp(test_a1(n,3))/sum;

        sum = exp(test_a2(n,1))+exp(test_a2(n,2))+exp(test_a2(n,3));
        test_y2(n,1) = exp(test_a2(n,1))/sum;
        test_y2(n,2) = exp(test_a2(n,2))/sum;
        test_y2(n,3) = exp(test_a2(n,3))/sum;
   
        sum = exp(test_a3(n,1))+exp(test_a3(n,2))+exp(test_a3(n,3));
        test_y3(n,1) = exp(test_a3(n,1))/sum;
        test_y3(n,2) = exp(test_a3(n,2))/sum;
        test_y3(n,3) = exp(test_a3(n,3))/sum;
    end

%%%%%%%%%%%%%%%CONFUSION MATRIX test %%%%%%%%%%%%%%%%%%%%
confusion_mat_test = zeros(3,3);
for i = 1:test_size(1,1)
    label_count = max(max(test_y1(i,1),test_y1(i,2)),test_y1(i,3));
        if label_count == test_y1(i,1)
           confusion_mat_test(1,1) = confusion_mat_test(1,1)+1;
        elseif label_count == test_y1(i,2)
           confusion_mat_test(1,2) = confusion_mat_test(1,2)+1;
        elseif label_count == test_y1(i,3)
            confusion_mat_test(1,3) = confusion_mat_test(1,3)+1;
        end

    label_count = max(max(test_y2(i,1),test_y2(i,2)),test_y2(i,3));
        if label_count == test_y2(i,2)
           confusion_mat_test(2,2) = confusion_mat_test(2,2)+1;
        elseif label_count == test_y2(i,1)
           confusion_mat_test(2,1) = confusion_mat_test(2,1)+1;
        elseif label_count == test_y2(i,3)
            confusion_mat_test(2,3) = confusion_mat_test(2,3)+1;
        end

    label_count = max(max(test_y3(i,1),test_y3(i,2)),test_y3(i,3));
        if label_count == test_y3(i,3)
           confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
        elseif label_count == test_y3(i,2)
           confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
        elseif label_count == test_y3(i,1)
            confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
        end
end

accurately_classified_test = confusion_mat_test(1,1)+confusion_mat_test(2,2)+confusion_mat_test(3,3);
accuracy_test_total =test_size * 3;
accuracy_test = (accurately_classified_test / accuracy_test_total)*100;

%%%%%%%%%%%%%55555555555555555555
%{
xaxis = -10:0.05:20;%0.05
yaxis = -20:0.05:20;
[x, y] = meshgrid(xaxis, yaxis);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);

plot_a  = zeros(total_plot_data,3);
plot_a2  = zeros(total_plot_data,3);
plot_a3  = zeros(total_plot_data,3);

plot_y = zeros(total_plot_data(1,1),3);
    
plot_design_mat = [];

for i = 1 :  total_plot_data % 10 ke jagah a1
    i
    vec = plot_data(i,1:end) ;
    a = calculate_design_matrix_values(vec,mon_mat);
    ad=a';
    plot_design_mat = [plot_design_mat;ad];
end

for n=1:total_plot_data 
        plot_a(n,1) = w1' * plot_design_mat(n,:)';
        plot_a(n,2) = w2' * plot_design_mat(n,:)';
        plot_a(n,3) = w3' * plot_design_mat(n,:)';

        sum = exp(plot_a(n,1))+exp(plot_a(n,2))+exp(plot_a(n,3));
        plot_y(n,1) = exp(plot_a(n,1))/sum;
        plot_y(n,2) = exp(plot_a(n,2))/sum;
        plot_y(n,3) = exp(plot_a(n,3))/sum;
      
        label_count = max(max(plot_y(n,1),plot_y(n,2)),plot_y(n,3));
        label_count
        if (plot_y(n,1) == plot_y(n,2) )
            predicted_plot(n,1) = 1;
        elseif label_count == plot_y(n,1)
           predicted_plot(n,1) = 1;
        elseif label_count == plot_y(n,2)
            predicted_plot(n,1) = 2;
        else
            predicted_plot(n,1) = 3;
        end
end

plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');

hold on;
scatter(train1_data(:,1),train1_data(:,2),'m','x');
scatter(train2_data(:,1),train2_data(:,2),'w','x');
scatter(train3_data(:,1),train3_data(:,2),'r','x');
xlabel('x1')
ylabel('x2')
str  = { strcat('Decision Region Plot')};
title(str,'FontSize',15);

%t = text(-5,10,'CLASS 1','Color','r','FontSize',14);
%t1 = text(5,15,'CLASS 2','Color','g','FontSize',14);
%t3 = text(5,-15,'CLASS 3','Color','b','FontSize',14);
hold off;
%%%%%%%%%%%%%%%%plot
%}
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

