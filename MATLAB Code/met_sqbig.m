%MET_SQBIG.M       Plot Mean Hitting Time curve 
%                  for mean-reverting sqrt process.
%
%                  Call met_sq.m to superimpose curve from solving BVP.
%
%   Creates pic_metsqbig.eps
%
%   DJH, August 2005

clf

% Parameter values
lambda = 1;
mu = 0.5;
sigma = 0.3;

a = 1;
b = 2;

%%%%%%%%%%%%%% Monte Carlo %%%%%%%%%%%%%%%%%%%%%


randn('state',100)                                % set the state of randn
N = 1e3; dt = 1/N; 

Xzerovals = linspace(a,b,40);

for k= 1:length(Xzerovals)
     Xzero = Xzerovals(k);
     Xzero
     M = 1e3;    % number of paths
     texit = zeros(M,1);

     for s = 1:M
          X = Xzero;
          t = 0;
          while X > a & X < b,
            dW = sqrt(dt)*randn;                         % increments
           % X = X*exp( (mu - 0.5*sigma^2)*dt + sigma*dW);
           % X = X + dt*lambda*(mu-X) + dW*sigma*sqrt(abs(X));
            X = X + dt*lambda*(mu-X) + dW*sigma*sqrt(abs(X));
            t = t + dt;
          end
          texit(s) = t - 0.5*dt;
     end



     tmean = mean(texit)
     tstd = std(texit)

     cileft = tmean - 1.96*tstd/sqrt(M)
     ciright = tmean + 1.96*tstd/sqrt(M)

     plot(Xzero,tmean,'blx','MarkerSize',10')
     hold on
     plot([Xzero,Xzero],[cileft,ciright],'r-','LineWidth',2)
end
grid on




