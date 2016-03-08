%change C values here, run MLassign2.m before run testSVM.m
C=0.1;
%primal form SVM
cvx_begin
variable w_primal1(10)
variable b_primal1
variable Slack1(683) %non-negative slack variables

minimize( 0.5*w_primal1'*w_primal1+C*sum(Slack1) )
subject to

    Slack1>=0;
    y.*(X*w_primal1+b_primal1)-1+Slack1>=0 ;

cvx_end

%dual form SVM
cvx_begin
X_Cartesian=X*X';

variable alphaV1(683)  %parametre alphai
alpha_times_Y1=alphaV1.*y;
Y_Cartesian=y*y';
%maximize Dual Lagrange function
maximize( sum(alphaV1)-0.5*(alpha_times_Y1'*X_Cartesian*alpha_times_Y1) )
subject to
    alphaV1>=0; % 0<=alphai<=C, for any i
    alphaV1<=C;
    sum(alphaV1.*y)==0 ;

cvx_end

%calculate w and b in Dual problem when C=
w_dual=X'*(alphaV1.*y);
svindex=find(alphaV1>1e-8 & alphaV1<C-1e-8);%store indice of support vectors C=

%b is average (yi-xi innerproduct w)
b_dual=(1/length(svindex))*sum(y(svindex)-X(svindex,:)*w_dual);


%test SVMs and calculate the error rate
predictTraindata_Y_P1=sign(X*w_primal1+b_primal1);
predictTraindata_Y_D1=sign(X*w_dual+b_dual);

%error rate of Primal classifier and Dual Classifier C=
errorTraindata_P1=sum(y~=predictTraindata_Y_P1)/size(y,1);
errorTraindata_D1=sum(y~=predictTraindata_Y_D1)/size(y,1);