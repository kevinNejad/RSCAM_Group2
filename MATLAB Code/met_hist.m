%MET_HIST.M      Compute mean exit times 
%                Plot a histogram of the sample exit times
%                Record Monte Carlo mean approximation 
%
% Creates pic_methist.eps
%
% DJH July 2005


clf

randn('state',100)                                % set the state of randn
%N = 1e2; dt = 1/N; 
N = 1e4; dt = 1/N;
mu = 0.1; sigma = 0.2;                            % problem parameters

a = 0.5;
b = 2;
Xzero = 1;

M = 5e4;    % number of paths

texit = zeros(M,1);

for s = 1:M
     s
     X = Xzero;
     t = 0;
     while X > a & X < b,
       dW = sqrt(dt)*randn;                         % increments
       X = X*exp( (mu - 0.5*sigma^2)*dt + sigma*dW);
       t = t + dt;
     end
     texit(s) = t - 0.5*dt;
      %t
end

tmean = mean(texit)
tstd = std(texit)

cileft = tmean - 1.96*tstd/sqrt(M)
ciright = tmean + 1.96*tstd/sqrt(M)

%%% Compute exact value %%% 

%%% Temporary varables to break up the formula %%%
temp1 = 1/(0.5*sigma^2 - mu);
temp2 = log(Xzero/a);
powera = 1 - 2*mu/(sigma^2);
powerb = 1 - mu/(0.5*(sigma^2));
temp3 = 1 - (Xzero/a).^powera;
temp4 = 1 - (b/a)^powerb;
temp5 = log(b/a);


%%% Mean Hitting time formula %%%
texact = temp1*( temp2 - (temp3./temp4)*temp5)

hist(texit,50)
xlabel('First exit time','FontSize',16,'FontWeight','Bold') 
set(gca,'FontWeight','Bold','FontSize',12)

