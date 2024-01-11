function ActiveSamplingDemos
% Series of demonstrations to accompany a paper written for the MDPI
% journal Algorithms 'Bayesian Networks and Causal Reasoning' special
% issue. They are designed to showcase applications of active sampling
% that maximises expected information gain to the selection of data 
% samples either through experimental design or selection from a a 
% large dataset. This can be thought of as analagous to the process of 
% foveal sampling as used by biological systems
%__________________________________________________________________________

close all

BASIS = 'Gaussian'; % Basis set
n     = 16;         % Number of elements in basis
C     = -32;        % Very unlikely to want to terminate sampling early

% Random sampling
%--------------------------------------------------------------------------

ActiveSamplingBasis(BASIS,n,C,'rand')

% Intelligent sampling
% -------------------------------------------------------------------------

ActiveSamplingBasis(BASIS,n,C,'int')

% Balance against cost
%--------------------------------------------------------------------------
% It may be costly in terms of both time and computational resources (or
% data-gathering resources) to sample every possible data point. Including
% an explicit cost for this time gives us an opportunity to balance the
% option to continue sampling against the 

C = 1/4;            % Small probability of terminating sampling early
ActiveSamplingBasis(BASIS,n,C,'int')

hf = exp(-8);       % Introduce unresolvable uncertainty
ActiveSamplingBasis(BASIS,n,C,'int',hf)

% Dynamic sampling
%--------------------------------------------------------------------------
% Next we consider the situation in which the function being approximated
% varies with time.

n = 8;
C = 0;
ActiveSamplingDynamic(BASIS,n,C,'int');

% Clinical Trial
%--------------------------------------------------------------------------
C = 0;
n = 4;

ActiveSamplingRCT(BASIS,n,C,'rand');

ActiveSamplingRCT(BASIS,n,C,'int');

C = 1; % Preference for long-term survival

ActiveSamplingRCT(BASIS,n,C,'int');

% Demo routines
%==========================================================================

function ActiveSamplingBasis(BASIS,n,C,SAMP,hf)
% Demonstration routine in which active sampling is used to infer the
% structure of an arbitrary function. Highlights different sampling
% strategies under different sorts of generative model which can be
% selected by choosing the basis set:
%
% BASIS = 'Gaussian';
% BASIS = 'Cosine';
% BASIS = 'Polynomial';
%
% The n variable determines the number of elements in the basis set
%__________________________________________________________________________

rng default
x = (-100:100)';    % Domain of function

% Generative model
%--------------------------------------------------------------------------
pE = zeros(n,1);    % Prior coefficients for basis set
pC = eye(n)/8;      % Covariance of priors
h  = 1/16;          % Likelihood covariance
try hf = hf; catch, hf = 0; end % non-constant h.

X = AS_Basis(n,BASIS,x);

% Generate sample data:
%--------------------------------------------------------------------------
z = sqrtm(pC)*randn(size(pE)) + pE;

Y = X*z + randn(size(x)).*sqrt(h + hf*x.^2);

figure('Name','Inference Demo 1','Color','White','Position',[360 75 560 550]), clf
subplot(2,2,1)
plot(x,Y,'.','Color',[0.6 0.6 0.8],'MarkerSize',8), hold on
axis square
box off
title('Sample Data')
xlabel('x'), ylabel('y')

% Plot beliefs and predictions over time:
%--------------------------------------------------------------------------
Ep = pE;
Cp = pC;

for i = 1:(min(28,length(x)))

    % Sample action
    %----------------------------------------------------------------------
    if strcmp(SAMP,'int')
        I = AS_Info(Ep,Cp,X,h,hf);      % Expected information gain
    else
        I = ones(size(x));              % All locations equally plausible
    end

    I(end+1) = C;                       % Stop sampling
    I = exp(64*I)/sum(exp(64*I));       % Softmax (with low temperature)
    a = find(cumsum(I)>rand,1,'first'); % Sample action                  
    if a == length(I)
        disp('Sampling Terminated - Sufficient information gained')
        return
    end

    % Sample datapoint from action
    %----------------------------------------------------------------------
    y = X(a,:)*z + randn*sqrt(h + hf*(a-100)^2);
    
    [Ep,Cp] = AS_Inference(Ep,Cp,h+hf*(a-100)^2,X(a,:),y);

    subplot(2,2,2)
    c = diag(X*Cp*X') + h + hf*x.^2;
    AS_PlotIntervals(x,X*Ep,c,[0.8,0.8,0.9]), hold on
    plot(x,X*Ep,'LineWidth',2,'Color',[0.1 0.1 0.4])
    plot(x(a),y,'.','MarkerSize',12,'Color',[0.2 0.2 0.6])
    axis square
    box off
    title('Prediction')
    xlabel('x'), ylabel('y')
    hold off

    subplot(2,2,3)
    bar(1:length(Ep),Ep,'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9])
    hold on
    errorbar(1:length(Ep),Ep,-1.64*sqrt(diag(Cp)),1.64*sqrt(diag(Cp)),'CapSize',0,'LineWidth',1,'Color',[0.2 0.2 0.6],'LineStyle','none')
    hold off
    axis square
    box off
    title('Posterior modes')

    subplot(2,2,4)
    if i == 1
        plot(a,-i,'.','MarkerSize',5,'Color',[0.1 0.1 0.4]), hold on
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    else
        plot(a,-i,'.','MarkerSize',5,'Color',[0.1 0.1 0.4]), hold on
        plot([b a],[-i,-i],'Color',[0.1 0.1 0.4])
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    end
    xlim([1 length(x)])
    b = a;
    axis square
    axis off
    title('Choices')

    drawnow
end
disp('Sampling Terminated - Maximum number of iterations reached')

function ActiveSamplingDynamic(BASIS,n,C,SAMP)
% This demo deals with a situation in which we are interested in
% monitoring the shape of a function that can change dynamically in time

rng default
x = (-100:100)';    % Domain of function

% Generative model
%--------------------------------------------------------------------------
pE = zeros(n^2,1);  % Prior coefficients for basis set
pC = eye(n^2)/8;    % Covariance of priors
h  = 1/16;          % Likelihood covariance

X = AS_Basis(n,BASIS,x);

% Plot beliefs and predictions over time:
%--------------------------------------------------------------------------
Ep = pE;
Cp = pC;

figure('Name','Inference Demo 2','Color','White','Position',[360 75 560 550]), clf
z = sqrtm(pC)*randn(size(pE)) + pE;

% Generate sample data:
%--------------------------------------------------------------------------  
Y = reshape((kron(X,X)*z + randn(length(x)^2,1)*sqrt(h)),[length(x),length(x)]);

for i = 1:(length(x)/8)
    
    j = 1+(i-1)*8;

    subplot(3,2,1)
    plot(x,Y(:,j),'.','Color',[0.6 0.6 0.8],'MarkerSize',8), hold on
    axis square
    box off
    title('Sample Data')
    xlabel('x'), ylabel('y')
    hold off

    % Sample action
    %----------------------------------------------------------------------
    if strcmp(SAMP,'int')
        I = AS_Info(Ep,Cp,kron(X(j,:),X),h); % Expected information gain
    else
        I = ones(size(x));              % All locations equally plausible
    end

    I(end+1) = C;                       % Stop sampling
    I = exp(64*I)/sum(exp(64*I));       % Softmax (with low temperature)
    a = find(cumsum(I)>rand,1,'first'); % Sample action                  
    if a == length(I)
        disp('Sampling Terminated - Sufficient information gained')
        return
    end

    % Sample datapoint from action
    %----------------------------------------------------------------------
    y = kron(X(j,:),X(a,:))*z + randn*sqrt(h);
    
    [Ep,Cp] = AS_Inference(Ep,Cp,h,kron(X(j,:),X(a,:)),y);

    subplot(3,2,2)
    c = diag(kron(X(j,:),X)*Cp*kron(X(j,:),X)') + h;
    AS_PlotIntervals(x,kron(X(j,:),X)*Ep,c,[0.8,0.8,0.9]), hold on
    plot(x,kron(X(j,:),X)*Ep,'LineWidth',2,'Color',[0.1 0.1 0.4])
    plot(x(a),y,'.','MarkerSize',12,'Color',[0.2 0.2 0.6])
    axis square
    box off
    title('Prediction')
    xlabel('x'), ylabel('y')
    hold off

    subplot(3,2,3)
    bar(1:length(Ep),Ep,'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9])
    hold on
    errorbar(1:length(Ep),Ep,-1.64*sqrt(diag(Cp)),1.64*sqrt(diag(Cp)),'CapSize',0,'LineWidth',1,'Color',[0.2 0.2 0.6],'LineStyle','none')
    hold off
    axis square
    box off
    title('Posterior modes')

    subplot(3,2,4)
    if i == 1
        plot(a,-i,'.','MarkerSize',5,'Color',[0.1 0.1 0.4]), hold on
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    else
        plot(a,-i,'.','MarkerSize',5,'Color',[0.1 0.1 0.4]), hold on
        plot([b a],[-i,-i],'Color',[0.1 0.1 0.4])
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    end
    xlim([1 length(x)])
    b = a;
    axis square
    axis off
    title('Choices')

    subplot(3,2,5)
    imagesc((X*reshape(z,[n,n])*X')'), colormap gray
    xlabel('x'), ylabel('t')
    axis square
    title('Time Evolution')

    subplot(3,2,6)
    imagesc((X*reshape(Ep,[n,n])*X')'), colormap gray
    xlabel('x'), ylabel('t')
    axis square
    title('Inferred Time Evolution')

    drawnow
end
disp('Sampling Terminated - Maximum number of iterations reached')

function ActiveSamplingRCT(BASIS,n,C,SAMP)
% This demo deals with the situation in which we have a treatment whose
% effect we want to evaluate. At each iteration, the choices we have
% include whether or not to recruit a new participant, which baseline
% characteristics we want them to have, and whether or not they are given
% the intervention, and when to follow them up. This assumes that the
% treatment in question works relatively rapidly, such that we can assess
% the response for each participant (or cohort of participants) before
% recruiting the next participant (or cohort).

rng default

% Generative model (Supports Gaussian Basis set - priors need adjusting for alternatives)
%--------------------------------------------------------------------------
pE = [ones(n,1); zeros(2,1)];                % Prior coefficients for basis set (prior that survival more likely than not at each time)
pC = eye(n+2)/4;                             % Covariance of priors
h  = 1;                                      % Temperature

t  = (1:20)';                                % Follow-up time (weeks)
N  = 8;                                      % Number of participants per cohort
Nc = 16;                                     % Number of cohorts

X  = AS_Basis(n,BASIS,t);                    % Effect of follow-up time
X  = [X ones(size(X,1),1)/2];                % Effect of sex
X  = [X ones(size(X,1),1)/2];                % Effect of treatment


% Plot beliefs and predictions over time:
%--------------------------------------------------------------------------
Ep = pE;
Cp = pC;

figure('Name','Inference Demo 3','Color','White','Position',[360 75 560 550]), clf
z = sqrtm(pC)*randn(size(pE)) + pE;

% Plot survival for females receiving treatment
subplot(4,4,1)
U = X*z;
U = 1./(1 + exp(-h*U));
bar([1;cumprod(U)],'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9])
box off
title('TF')

% Plot survival for males receiving treatment
subplot(4,4,2)
V = X;
V(:,end-1) = -1/2;
U = V*z;
U = 1./(1 + exp(-h*U));
bar([1;cumprod(U)],'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9])
box off
title('TM')

% Plot survival for females receiving placebo
subplot(4,4,5)
V = X;
V(:,end) = -1/2;
U = V*z;
U = 1./(1 + exp(-h*U));
bar([1;cumprod(U)],'FaceColor',[0.9,0.8,0.8],'EdgeColor',[0.9,0.8,0.8])
box off
title('PF')

% Plot survival for males receiving placebo
subplot(4,4,6)
V = X;
V(:,end-1) = -1/2;
V(:,end) =   -1/2;
U = V*z;
U = 1./(1 + exp(-h*U));
bar([1;cumprod(U)],'FaceColor',[0.9,0.8,0.8],'EdgeColor',[0.9,0.8,0.8])
box off
title('PM')

r  = [1/3 1/2 2/3]; % Randomisation ratios
fm = 1/2;           % Proportion female to male
Rx = [0;0];         % Record number of participants treated in each group

for i = 1:Nc % Iterate over cohorts
   
    % Sample action
    %----------------------------------------------------------------------
    if strcmp(SAMP,'int')
        I = N*AS_InfoRCT(Ep,Cp,X,h,C);       % Expected information gain (or expected free energy)
    else
        I = ones(size(X,1),3);               % All followups/randomisation ratios equally plausible
    end

    I = exp(4*I)/sum(sum(exp(4*I)));         % Softmax (with low temperature)
    [tt,rr] = find(reshape(cumsum(I(:)),size(I))>rand,1,'first'); % Sample action                  

    % Generate data
    %----------------------------------------------------------------------
    Y  = zeros(N,3);

    for j = 1:N
       V          = X;
       V(:,end-1) = (fm > rand)-1/2;
       Y(j,2)     = V(1,end-1);
       V(:,end)   = (r(rr) > rand)-1/2;
       Y(j,3)     = V(1,end);
       V          = V(1:tt,:);
       U          = V*z;
       U          = 1./(1 + exp(-h*U));
       U          = prod(U);
       Y(j,1)     = U > rand;
    end
    Rx = Rx + [sum(Y(:,3) + 1/2); N - sum(Y(:,3) + 1/2)];

    [Ep,Cp] = AS_InferenceRCT(Ep,Cp,h,X(1:tt,:),Y);


    subplot(2,2,2)
    V = X;
    V(:,end-1) = 0; % Remove demographics for plotting

    % Treatment group
    W           = sqrt(diag(V*Cp*V')); % Standard deviations
    cu          = cumprod(1./(1 + exp(-h*(W*1.64 + V*Ep)))) - cumprod(1./(1 + exp(-h*V*Ep)));
    cu          = [0; cu];         % Upper bound
    cl          = cumprod(1./(1 + exp(-h*V*Ep))) - cumprod(1./(1 + exp(-h*(-W*1.64 + V*Ep))));
    cl          = [0; cl];         % Lower bound
    AS_PlotIntervals([0; t],[1; cumprod(1./(1 + exp(-h*V*Ep)))],[cl cu],[0.8,0.8,0.9]), hold on
    plot([0; t],[1; cumprod(1./(1 + exp(-h*V*Ep)))],'LineWidth',2,'Color',[0.1 0.1 0.4])
    
    % Placebo group
    V(:,end)   = -1/2;
    W           = sqrt(diag(V*Cp*V')); % Standard deviations
    cu          = cumprod(1./(1 + exp(-h*(W*1.64 + V*Ep)))) - cumprod(1./(1 + exp(-h*V*Ep)));
    cu          = [0; cu];         % Upper bound
    cl          = cumprod(1./(1 + exp(-h*V*Ep))) - cumprod(1./(1 + exp(-h*(-W*1.64 + V*Ep))));
    cl          = [0; cl];         % Lower bound
    AS_PlotIntervals([0; t],[1; cumprod(1./(1 + exp(-h*V*Ep)))],[cl cu],[0.9,0.8,0.8])
    plot([0; t],[1; cumprod(1./(1 + exp(-h*V*Ep)))],'LineWidth',2,'Color',[0.4 0.1 0.1])

    plot(tt,sum(Y(Y(:,3)>0),1)/sum(Y(:,3)>0),'.','MarkerSize',12,'Color',[0.2 0.2 0.6]), xlim([0 20]), ylim([0 1])
    plot(tt,sum(Y(Y(:,3)<0),1)/(sum(Y(:,3)<0)),'.','MarkerSize',12,'Color',[0.6 0.2 0.2]), xlim([0 20]), ylim([0 1]), hold off
    axis square
    box off
    title('Prediction')

    subplot(2,2,3)
    bar(1:length(Ep),Ep,'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9])
    hold on
    errorbar(1:length(Ep),Ep,-1.64*sqrt(diag(Cp)),1.64*sqrt(diag(Cp)),'CapSize',0,'LineWidth',1,'Color',[0.2 0.2 0.6],'LineStyle','none')
    hold off
    axis square
    box off
    title('Posterior modes')

    subplot(4,2,6)
    a = tt;
    if i == 1
        plot(a,-i,'.','MarkerSize',10*rr,'Color',[0.1 0.1 0.4]), hold on
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    else
        plot(a,-i,'.','MarkerSize',10*rr,'Color',[0.1 0.1 0.4]), hold on
        plot([b a],[-i,-i],'Color',[0.1 0.1 0.4])
        plot([a a],[-i -i-1],'Color',[0.1 0.1 0.4])
    end
    b = a;
    xlim([0 20])
    axis off
    title('Choices')

    subplot(4,2,8)
    bar(i,-Rx(2),'FaceColor',[0.8,0.8,0.9],'EdgeColor',[0.8,0.8,0.9]), xlim([0 Nc+1]), hold on
    bar(i,Rx(1),'FaceColor',[0.9,0.8,0.8],'EdgeColor',[0.9,0.8,0.8]), xlim([0 Nc+1]), hold on
    box off
    title('Treatment groups')
    drawnow
end
% subplot(2,2,3), hold on
% bar(1:length(Ep),z,'FaceColor',[1,0,0],'EdgeColor',[1,0,0],'BarWidth',0.05)

subplot(2,2,2), hold on
V = X;
V(:,end-1) = 0; % Remove demographics for plotting
plot([0; t],[1; cumprod(1./(1 + exp(-h*V*z)))],'-b')
V(:,end)   = -1/2;
plot([0; t],[1; cumprod(1./(1 + exp(-h*V*z)))],'-r')


% Auxiliary routines (plotting, inference, and expected information gain)
%==========================================================================

function [Ep,Cp] = AS_Inference(pE,pC,h,X,y)
% This function computes the exact posteriors associated with the
% parameters of a linear Gaussian model with fixed likelihood precision
ih      = 1/h;
L       = X'*ih*X + inv(pC);
Ep      = L\(pC\pE + X'*ih*y);
Cp      = inv(L);

function [Ep,Cp] = AS_InferenceRCT(pE,pC,h,X,y)
% This function uses a variational Laplace procedure to infer the
% parameters generating the data for a randomised controlled trial.

Ep = pE;
Cp = pC;

for i = 1:16

    LL    = 0;                            % log likelihood
    dLdq  = zeros(size(Ep));              % log likelihood gradients
    dLdqq = zeros(length(Ep),length(Ep)); % Hessian of log likelihood

    % Calculate gradients and Hessians
    for j = 1:size(y,1) % Accumulate over subjects
        
        Y = y(j,1); % Survival to follow-up (1 = yes, 0 = no)

        % Likelihood of survival to follow-up time:
        V          = X;
        V(:,end)   = y(j,end);
        V(:,end-1) = y(j,end-1);
        l          = prod(1./(1 + exp(-h*V*Ep)));    % Survival likelihood
        p          = Y*l + (1 - Y)*(1-l);            % Likelihood of data
        LL         = LL + log(p);                    % Accumulate log likelihood

        % Gradient of log likelihood:
        if size(V,1) > 1
            dldq = l*sum(h*V./(1 + exp(h*V*Ep)));
        else
            dldq = l*(h*V./(1 + exp(h*V*Ep)));
        end
        dpdq = (2*Y - 1)*dldq;
        dLdq = dLdq + dpdq'/p;                       % Accumulate log likelihood gradients

        % Hessian of log likelihood:
        VV = zeros(size(dLdqq));
        for k = 1:size(V,1)
            VV = VV + h^2*V(k,:)'*V(k,:)*exp(V(k,:)*Ep)./((1 + exp(V(k,:)*Ep))^2);
        end

        dldqq = dldq'*dldq/l - l*VV;
        dpdqq = (2*Y - 1)*dldqq;
        dLdqq = dLdqq - dpdq'*dpdq/(p^2) + dpdqq/p; % Accumulate gradients of Hessian

    end
    
    % Augment with priors
    iC = pinv(pC);
    dLdqq = dLdqq - iC;
    dLdq  = dLdq  + iC*(pE - Ep);

    % Newton optimisation
    Ep = Ep - pinv(dLdqq)*dLdq;

end
Cp = - pinv(dLdqq);

function I = AS_Info(~,Cp,X,h,hf)
% Information gain

I = zeros(size(X,1),1);

try hf; catch, hf = 0; end

for i = 1:length(I)
    I(i) = 0.5*log((X(i,:)*Cp)*X(i,:)' + (h + hf*(i-100).^2)) - 0.5*log(h + hf*(i-100).^2);
end

function I = AS_InfoRCT(Ep,Cp,X,h,C)
% This function computes the expected information gain under different
% choices as to how to run an RCT.

I = zeros(size(X,1),3);
R = [1/3 1/2 2/3];      % Available randomisation ratios
R = [R;1-R];

for t = 1:size(X,1)     % Iterate over follow-up time
    for a  = 1:3        % Iterate over randomisation ratios
        E  = 0;         % Expectation for likelihood
        Er = [0;0];     % Conditional expectation for likelihood
        U  = [0;0];
        D  = [0.5;0.5]; % Prior over demographics
        H  = [0;0];     % Entropy conditioned upon treatment group
        for r = 1:2     % Iterate over different treatment groups
            for d = 1:2 % Iterate over demographic groups (sex)
                V          = X(1:t,:);
                V(:,end)   = r-1.5;
                V(:,end-1) = d-1.5;

                % Expand likelihood probability to second order
                l    = prod(1./(1 + exp(-h*V*Ep)));   % Zeroth order
                
                if t > 1
                    dldq = l*sum(h*V./(1 + exp(h*V*Ep))); % First order
                else
                    dldq = l*(h*V./(1 + exp(h*V*Ep)));
                end

                % Check using finite differences
                %----------------------------------------------------------
                % for i = 1:length(dldq)
                %     dEp     = zeros(size(Ep));
                %     dEp(i)  = exp(-2);
                %     dldq(i) = (prod(1./(1 + exp(-h*V*(Ep+dEp)))) - prod(1./(1 + exp(-h*V*(Ep-dEp)))))*exp(2)/2;
                % end
                %----------------------------------------------------------

                VV = zeros(length(dldq));
                for k = 1:size(V,1)
                    VV = VV + h^2*V(k,:)'*V(k,:)*exp(V(k,:)*Ep)./((1 + exp(V(k,:)*Ep))^2);
                end
                dldqq = dldq'*dldq/l - l*VV;          % Second order

                % Check using finite differences
                %----------------------------------------------------------
                % for i = 1:length(dldq)
                %     for j = 1:length(dldq)
                %         dEpi     = zeros(size(Ep));
                %         dEpj     = dEpi;
                %         dEpi(i)  = exp(-2);
                %         dEpj(j)  = exp(-2);
                %         dldqq(i,j) = (prod(1./(1 + exp(-h*V*(Ep+dEpi+dEpj)))) - prod(1./(1 + exp(-h*V*(Ep-dEpi+dEpj)))))*exp(4)/4 ...
                %                     -(prod(1./(1 + exp(-h*V*(Ep+dEpi-dEpj)))) - prod(1./(1 + exp(-h*V*(Ep-dEpi-dEpj)))))*exp(4)/4;
                %     end
                % end
                %----------------------------------------------------------

                % Approximate expectation
                Er(r) = Er(r) + D(d)*(l + trace(Cp*dldqq)/2);
                H(r)  = H(r) + D(d)*(-[l, 1-l]*[log(l);log(1-l)]...
                    - trace(Cp*(dldqq*log(l/(1-l)) + (1/l + 1/(1-l))*(dldq'*dldq)))/2);
                %--------------------------

                % U(r) = D(d)*0.5*log(2*pi*exp(1)*((V(t,:)*Cp*dldq')*(V(t,:)*Cp*dldq')')*h^2);

                %--------------------------
            end
            % Ensure consistent with probability
            Er(Er>0.999) = 0.999;
            Er(Er<0.001) = 0.001;
                 
        end
        E    = R(:,a)'*Er;
        % Ensure consistent with probability
        E(E>0.999) = 0.999;
        E(E<0.001) = 0.001;

        % Add (approximate) entropy for predictive posterior
        I(t,a) = I(t,a) - [E, 1-E]*[log(E);log(1-E)];

        % Subtract (approximate) conditional entropy
        I(t,a) = I(t,a) - R(:,a)'*H;

        % Add contribution of preferences (set to favour long term
        % survival)
        I(t,a) = I(t,a) + [E, 1-E]*[C*(2*exp((t - size(X,1))/8) - 1);-C];

        %------------------------------------------------------------------

        % I(t,a) = I(t,a) + R(:,a)'*U + [E, 1-E]*[C*(2*exp((t - size(X,1))/8) - 1);-C];

        %------------------------------------------------------------------
    end
end

function AS_PlotIntervals(x,y,c,col)
% This function plots 90% credible intervals

if numel(c) == length(c)
    fill([x; flipud(x)],[y-1.64*sqrt(c); flipud(y)+1.64*sqrt(flipud(c))],col,'EdgeColor','none');    
else
    fill([x; flipud(x)],[y-c(:,1); flipud(y)+flipud(c(:,2))],col,'EdgeColor','none');
end

function X = AS_Basis(n,BASIS,x)
% This function generates a set of basis functions.

X = zeros(length(x),n);

if strcmp(BASIS,'Gaussian')

    dn = round(length(x)/n);
    
    for i = 1:n
        X(:,i) = exp(-0.5*((x*0.75 - (min(x) + (i - 0.5)*dn)).^2)/dn^2); % The factor of 0.75 is to avoid edge effects
    end
    
elseif strcmp(BASIS,'Cosine')

    for i = 1:n
        X(:,i) = cos(pi*(x+min(x))*i/length(x));
    end

elseif strcmp(BASIS,'Polynomial')

    for i = 1:n
        X(:,i) = x.^(i-1);
        X(:,i) = 32*X(:,i)/sum(abs(X(:,i)));
    end

else
    disp('Please specify basis set: {Gaussian, Cosine, Polynomial}')
    return
end
