function [P,Z1,Z2,Q, history] = SGPCA(X, param, opts)
t_start = tic;

alpha=param.alpha;
lambda=param.lambda;


k=opts.k;
group_num=opts.group_num;
rho=opts.rho;
delta = opts.delta;
rho_max = opts.rho_max;
MAX_ITER=opts.MAX_ITER;
epsilon=opts.epsilon;
QUIET=opts.QUIET;


[n, d] = size(X);
if (sum(group_num) ~= d)
    error('invalid partition');
end

cum_part = cumsum(group_num);

[U,S,V]=svd(X);
%P = V(:, 1:k);
%Z = V(:, 1:k);
P = zeros(d,k);
Z1 = zeros(d,k);
Z2 = zeros(d,k);
U1 = zeros(d,k);
U2 = zeros(d,k);
Q = V(:, 1:k);


if ~QUIET
    fprintf('%3s\t%5s\t%5s\t%5s\t%5s\n', 'iter','P norm', 'Z1 norm', 'Z2 norm', 'loss');
end

for i = 1:MAX_ITER
     Pold=P;
     Z1old=Z1;
     Z2old=Z2;
     % P-update
     P = (X'*X+2*rho*eye(d))\(X'*X*Q+rho*Z1-U1+rho*Z2-U2);
     % Q-update
     [U,~,V] = svd(X'*X*P,'econ');
     Q = U*V';
     % Z-update
     Z1=shrinkage1(P+U1/rho,alpha*lambda/rho);
     
     start_ind = 1;
     for l = 1:length(group_num)
         sel = start_ind:cum_part(l);
         Z2(sel,:)=shrinkage2(P(sel,:)+U2(sel,:)/rho,(1-alpha)*lambda*group_num(l)/rho);
         start_ind = cum_part(l) + 1;
     end
     % U-update
     U1 = U1+rho*(P-Z1);
     U2 = U2+rho*(P-Z2);
     rho=min(delta*rho, rho_max);

    history.loss(i)  = loss(X, alpha, lambda, rho, cum_part, P, Q, Z1, Z2, U1, U2);

    history.P_norm(i)  = norm(P-Z1);
    history.Z1_norm(i)  = norm(P-Z2);
    %history.P_norm(i)  = norm(P-Pold);
    %history.Z1_norm(i)  = norm(Z1-Z1old);
    history.Z2_norm(i)  = norm(Z2-Z2old);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', i, history.P_norm(i), history.Z1_norm(i), history.Z2_norm(i), history.loss(i));
    end

    if (history.P_norm(i) + history.Z1_norm(i) + history.Z2_norm(i) < epsilon)
         break;
    end


end
if ~QUIET
    toc(t_start);
end
end


function obj = loss(X, alpha, lambda, rho, cum_part, P, Q, Z1, Z2, U1, U2)
    obj = 0;
    start_ind = 1;
    for i = 1:length(cum_part)
        sel = start_ind:cum_part(i);
        obj = obj + norm(Z2(sel, :));
        start_ind = cum_part(i) + 1;
    end
    obj = (1/2*norm(X-X*P*Q')^2 + (1-alpha)*lambda*obj+alpha*lambda*sum(abs(Z1), 'all') + rho*norm(P-Z1)^2/2+ trace(U1'*(P-Z1)) + rho*norm(P-Z2)^2/2+ trace(U2'*(P-Z2)));
end

function z = shrinkage1(x, tau)
    z = max(0, 1-tau./abs(x)).*x;
end

function z = shrinkage2(x, tau)
    z = max(0, 1-tau/norm(x))*x;
end
