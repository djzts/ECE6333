%% part 3 Kalman filter 
clc
clear all
close all

%% part 0 model setup
T =0.1;
A = [1,0,T  ,0  ;
     0,1,0  ,T  ;
     0,0,0.9,0.4;
     0,0,-0.4,0.9;];
B = [T^2/2 ,   0;
       0   , T^2/2;
       T   ,   0;
       0   ,   T;];
C = [1,0,0,0;
     0,1,0,0;];
D = 1;

%% loop, initialization and other setup
Q = eye(2);
R = 0.01*eye(2);

N_kalman = 50;
n_round = 5000;

 for j = 1:n_round
    % Initialize x_initial and covariance matrix
    mu = randi(10,4,1);
    Sigma = [1,0,0  ,0;
             0,1,0  ,0;
             0,0,1/4,0;
             0,0,0,1/4;];    
    P_pre = Sigma;
    X_kalman_init = mvnrnd(mu,Sigma)';
    
    U(j,:,:) = mvnrnd([0,0],Q,N_kalman)'; %#ok<SAGROW>
    V(j,:,:) = mvnrnd([0,0],R,N_kalman+1)'; %#ok<SAGROW>
    
%% collect data Y(noised sigals)    
    X(j,:,1) = X_kalman_init; %#ok<SAGROW>
    Y_noise(j,:,1) = outputUpdate(X(j,:,1)',V(j,:,1)',C,D); %#ok<SAGROW>

%% Kalman init    
    Xhat_kplus1_k(j,:,1) = X_kalman_init; %#ok<SAGROW>
    [Xhat_k_k(j,:,1),P_post] = Update(Xhat_kplus1_k(j,:,1)',Y_noise(j,:,1)',P_pre,C,R); %#ok<SAGROW>
        
    Y_estimate_kalman(j,:,1) = outputUpdate(Xhat_k_k(j,:,1)',0*V(j,:,1)',C,0*R); %#ok<SAGROW>
    
    
%% steady state init
    X_steady(j,:,1) = X_kalman_init; %#ok<SAGROW>
    Y_steady(j,:,1) = outputUpdate(X_steady(j,:,1)',0*V(j,:,1)',C,0*R); %#ok<SAGROW>
    
%% Kalman layout
    for i = 2:N_kalman+1
        X(j,:,i) = stateUpdate(X(j,:,i-1)',U(j,:,i-1)',A,B);
        Y_noise(j,:,i) = outputUpdate(X(j,:,i)',V(j,:,i)',C,D);

        [Xhat_kplus1_k(j,:,i),P_pre] = Prediction(A, Xhat_k_k(j,:,i-1)',B,U(j,:,i-1)',P_post,Q);
        [Xhat_k_k(j,:,i),P_post] = Update(Xhat_kplus1_k(j,:,i)',Y_noise(j,:,i)',P_pre,C,R);
        
        Y_estimate_kalman(j,:,i) = outputUpdate(Xhat_k_k(j,:,i)',0*V(j,:,i)',C,0*R);
%% steay layout        
        X_steady(j,:,i) = stateUpdate(X_steady(j,:,i-1)',0*U(j,:,i-1)',A,B);
        Y_steady(j,:,i) = outputUpdate(X_steady(j,:,i)',0*V(j,:,1)',C,0*R); 
    end

    
 end

 %% save matrix and data
save('U.mat','U');
save('V.mat','V');
save('Xhat_k_k.mat','Xhat_k_k')
save('Xhat_kplus1_k.mat','Xhat_kplus1_k')
save('Y.mat','Y_noise');
save('X.mat','X');

diff_Xhat_k_k = sum((Xhat_k_k - X).^2)/n_round;
diff_Xhat_kplus1_k = sum((Xhat_kplus1_k - X).^2)/n_round;
diff_X_steady = sum((X_steady - X).^2)/n_round;

%% visualization
figure(4) 
bar(reshape(diff_Xhat_k_k(1,:,:),4,51));
grid on

figure(5)
bar(reshape(diff_Xhat_kplus1_k(1,:,:),4,51));
grid on

figure(6)
bar(reshape(diff_X_steady(1,:,:),4,51));
grid on 



%% alternate data visualization

diff_Xhat_k_k = sum(sum(Xhat_k_k - X)/n_round).^2;
diff_Xhat_kplus1_k = sum(sum(Xhat_kplus1_k - X)/n_round).^2;
diff_X_steady = sum(sum(X_steady - X)/n_round).^2;

figure(7) 
semilogy(reshape(diff_Xhat_k_k(:,:,:),1,51),'g-');
hold on
semilogy(reshape(diff_Xhat_kplus1_k(:,:,:),1,51),'b--');
hold on
semilogy(reshape(diff_X_steady(:,:,:),1,51),'m-.');
grid on

set(legend( '$$|\hat{x}_{k\mid k}|$$','$$|\hat{x}_{k+1\mid k}|$$', 'steady state'),'Interpreter','Latex','FontSize', 10);

%% function set kalman filter


function [X_pre,P_pre] = Prediction(A,X,B,U,P,Q)
    X_pre = A*X + B * U;
    P_pre = A*P*A' + B*Q*B';
end

function [X_post,P_post] = Update(X_pre,yk,P_pre,H,R)
    v = yk - H*X_pre;
    K = P_pre*H'*(inv(H*P_pre*H' + R));
    X_post = X_pre + K*v;
    P_post = (eye(4)-K*H)*P_pre ;
end

%% function set normal system update
function X_update = stateUpdate(X,U,A,B)
    X_update = A*X + B*U;
end

function Y_update = outputUpdate(X,V,C,D) 
    Y_update = C*X +D*V;
end