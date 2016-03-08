%Xuanyu Zhao

%1st part C=0.1

%use libsvm to train and predict 
[train_label, train_inst] = libsvmread('a1a');
libsvmmodel1 = libsvmtrain(train_label, train_inst, '-c 0.1 -t 0');
[test_label, test_inst] = libsvmread('a1a.t');
[predict_label, accuracy, dec_values] = libsvmpredict(test_label, test_inst, libsvmmodel1);
%record w and b calculated by libsvm when C=0.1
w_lib1=transpose(libsvmmodel1.SVs)*libsvmmodel1.sv_coef; 
b_lib1=-libsvmmodel1.rho;

my_trainX=full(train_inst);
my_trainY=train_label;
my_testX=full(test_inst);
my_testY=test_label;
C1= 0.1;

%implement my primal problem when C=0.1
cvx_begin
variable w_primal1(119)
variable b_primal1
variable Slack1(1605) %non-negative slack variables

minimize( 0.5*w_primal1'*w_primal1+C1*sum(Slack1) )
subject to

    Slack1>=0;
    my_trainY.*(my_trainX*w_primal1+b_primal1)-1+Slack1>=0 ;

cvx_end

%dual problem when C=0.1

cvx_begin
X_Cartesian=my_trainX*my_trainX';

variable alphaV1(1605)  %parametre alphai
alpha_times_Y1=alphaV1.*my_trainY;
Y_Cartesian=my_trainY*my_trainY';
%maximize Dual Lagrange function
maximize( sum(alphaV1)-0.5*(alpha_times_Y1'*X_Cartesian*alpha_times_Y1) )
subject to
    alphaV1>=0; % 0<=alphai<=C, for any i
    alphaV1<=C1;
    sum(alphaV1.*my_trainY)==0 ;

cvx_end

%calculate w and b in Dual problem when C=0.1
w_dual1=my_trainX'*(alphaV1.*my_trainY);
svindex1=find(alphaV1>1e-8 & alphaV1<C1-1e-8);%store indice of support vectors C=0.1
%svindex=find(alphaV>1e-6 & alphaV<C-1e-6);1.1
%svindex1=find(alphaV1>1e-4 & alphaV1<C1-1e-4);10.1
%b is average (yi-xi innerproduct w)
b_dual1=(1/length(svindex1))*sum(my_trainY(svindex1)-my_trainX(svindex1,:)*w_dual1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2nd part C=1.1

%use libsvm to train and predict 

libsvmmodel2 = libsvmtrain(train_label, train_inst, '-c 1.1 -t 0');

[predict_label, accuracy, dec_values] = libsvmpredict(test_label, test_inst, libsvmmodel2);
%record w and b calculated by libsvm when C=1.1
w_lib2=transpose(libsvmmodel2.SVs)*libsvmmodel2.sv_coef; 
b_lib2=-libsvmmodel2.rho;


C2= 1.1;

%implement my primal problem when C=1.1
cvx_begin
variable w_primal2(119)
variable b_primal2
variable Slack2(1605) %non-negative slack variables

minimize( 0.5*w_primal2'*w_primal2+C2*sum(Slack2) )
subject to

    Slack2>=0;
    my_trainY.*(my_trainX*w_primal2+b_primal2)-1+Slack2>=0 ;

cvx_end

%dual problem when C=1.1

cvx_begin

variable alphaV2(1605)  %parametre alphai
alpha_times_Y2=alphaV2.*my_trainY;

%maximize Dual Lagrange function
maximize( sum(alphaV2)-0.5*(alpha_times_Y2'*X_Cartesian*alpha_times_Y2) )
subject to
    alphaV2>=0; % 0<=alphai<=C, for any i
    alphaV2<=C2;
    sum(alphaV2.*my_trainY)==0 ;

cvx_end

%calculate w and b in Dual problem when C=1.1
w_dual2=my_trainX'*(alphaV2.*my_trainY);

svindex2=find(alphaV2>1e-6 & alphaV2<C2-1e-6);
%svindex1=find(alphaV1>1e-4 & alphaV1<C1-1e-4);10.1
%b is average (yi-xi innerproduct w)
b_dual2=(1/length(svindex2))*sum(my_trainY(svindex2)-my_trainX(svindex2,:)*w_dual2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%3rd part C=10.1

%use libsvm to train and predict 

libsvmmodel3 = libsvmtrain(train_label, train_inst, '-c 10.1 -t 0');

[predict_label, accuracy, dec_values] = libsvmpredict(test_label, test_inst, libsvmmodel3);
%record w and b calculated by libsvm when C=10.1
w_lib3=transpose(libsvmmodel3.SVs)*libsvmmodel3.sv_coef; 
b_lib3=-libsvmmodel3.rho;


C3= 10.1;

%implement my primal problem when C=10.1
cvx_begin
variable w_primal3(119)
variable b_primal3
variable Slack3(1605) %non-negative slack variables

minimize( 0.5*w_primal3'*w_primal3+C3*sum(Slack3) )
subject to

    Slack3>=0;
    my_trainY.*(my_trainX*w_primal3+b_primal3)-1+Slack3>=0 ;

cvx_end

%dual problem when C=10.1

cvx_begin

variable alphaV3(1605)  %parametre alphai
alpha_times_Y3=alphaV3.*my_trainY;

%maximize Dual Lagrange function
maximize( sum(alphaV3)-0.5*(alpha_times_Y3'*X_Cartesian*alpha_times_Y3) )
subject to
    alphaV3>=0; % 0<=alphai<=C, for any i
    alphaV3<=C3;
    sum(alphaV3.*my_trainY)==0 ;

cvx_end

%calculate w and b in Dual problem when C=10.1
w_dual3=my_trainX'*(alphaV3.*my_trainY);

svindex3=find(alphaV3>1e-4 & alphaV3<C3-1e-4);
%b is average (yi-xi innerproduct w)
b_dual3=(1/length(svindex3))*sum(my_trainY(svindex3)-my_trainX(svindex3,:)*w_dual3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Final part: Test my SVMs classifier

%predict train data with my classifier when C=0.1 
predictTraindata_Y_P1=sign(my_trainX*w_primal1+b_primal1);
predictTraindata_Y_D1=sign(my_trainX*w_dual1+b_dual1);
%difference between prediction result of Primal classifier and Dual
%Classifier C=0.1 on train data
sum(predictTraindata_Y_P1~=predictTraindata_Y_D1)
%error rate of Primal classifier and Dual Classifier C=0.1
errorTraindata_P1=sum(my_trainY~=predictTraindata_Y_P1)/size(my_trainY,1);
errorTraindata_D1=sum(my_trainY~=predictTraindata_Y_D1)/size(my_trainY,1);

%predict test data with my classifier when C=0.1 
predictTestdata_Y_P1=sign(my_testX*[w_primal1;zeros(4,1)]+b_primal1);
predictTestdata_Y_D1=sign(my_testX*[w_dual1;zeros(4,1)]+b_dual1);
%difference between prediction result of Primal classifier and Dual
%Classifier C=0.1 on testdata
sum(predictTestdata_Y_P1~=predictTestdata_Y_D1)
%error rate of Primal classifier and Dual Classifier C=0.1
errorTestdata_P1=sum(my_testY~=predictTestdata_Y_P1)/size(my_testY,1);
errorTestdata_D1=sum(my_testY~=predictTestdata_Y_D1)/size(my_testY,1);

%C=0.1 the difference between w vector under my primal, my dual and libsvm
sum(abs(w_primal1-w_dual1))
sum(abs(w_primal1-w_lib1))
sum(abs(w_dual1-w_lib1))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C=1.1
%predict train data with my classifier when C=1.1 
predictTraindata_Y_P2=sign(my_trainX*w_primal2+b_primal2);
predictTraindata_Y_D2=sign(my_trainX*w_dual2+b_dual2);
%difference between prediction result of Primal classifier and Dual
%Classifier C=1.1 on train data
sum(predictTraindata_Y_P2~=predictTraindata_Y_D2)
%error rate of Primal classifier and Dual Classifier C=1.1
errorTraindata_P2=sum(my_trainY~=predictTraindata_Y_P2)/size(my_trainY,1);
errorTraindata_D2=sum(my_trainY~=predictTraindata_Y_D2)/size(my_trainY,1);

%predict test data with my classifier when C=1.1 
predictTestdata_Y_P2=sign(my_testX*[w_primal2;zeros(4,1)]+b_primal2);
predictTestdata_Y_D2=sign(my_testX*[w_dual2;zeros(4,1)]+b_dual2);
%difference between prediction result of Primal classifier and Dual
%Classifier C=1.1 on testdata
sum(predictTestdata_Y_P2~=predictTestdata_Y_D2)
%error rate of Primal classifier and Dual Classifier C=1.1
errorTestdata_P2=sum(my_testY~=predictTestdata_Y_P2)/size(my_testY,1);
errorTestdata_D2=sum(my_testY~=predictTestdata_Y_D2)/size(my_testY,1);

%C=1.1 the difference between w vector under my primal, my dual and libsvm
sum(abs(w_primal2-w_dual2))
sum(abs(w_primal2-w_lib2))
sum(abs(w_dual2-w_lib2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C=10.1
%predict train data with my classifier when C=10.1 
predictTraindata_Y_P3=sign(my_trainX*w_primal3+b_primal3);
predictTraindata_Y_D3=sign(my_trainX*w_dual3+b_dual3);
%difference between prediction result of Primal classifier and Dual
%Classifier C=10.1 on train data
sum(predictTraindata_Y_P3~=predictTraindata_Y_D3)
%error rate of Primal classifier and Dual Classifier C=10.1
errorTraindata_P3=sum(my_trainY~=predictTraindata_Y_P3)/size(my_trainY,1);
errorTraindata_D3=sum(my_trainY~=predictTraindata_Y_D3)/size(my_trainY,1);

%predict test data with my classifier when C=10.1 
predictTestdata_Y_P3=sign(my_testX*[w_primal3;zeros(4,1)]+b_primal3);
predictTestdata_Y_D3=sign(my_testX*[w_dual3;zeros(4,1)]+b_dual3);
%difference between prediction result of Primal classifier and Dual
%Classifier C=10.1 on testdata
sum(predictTestdata_Y_P3~=predictTestdata_Y_D3)
%error rate of Primal classifier and Dual Classifier C=10.1
errorTestdata_P3=sum(my_testY~=predictTestdata_Y_P3)/size(my_testY,1);
errorTestdata_D3=sum(my_testY~=predictTestdata_Y_D3)/size(my_testY,1);

%C=10.1 the difference between w vector under my primal, my dual and libsvm
sum(abs(w_primal3-w_dual3))
sum(abs(w_primal3-w_lib3))
sum(abs(w_dual3-w_lib3))

