function [W, D2, Vt, D1, Zt] = PARATUCK2(X, r2, r1)
%PARATUCK2 Computes PARATUCK2 decomposition of given three-way tensor

    [I, J, K] = size(X);
    
    D2 = randn(K, r2);
    Vt = randn(r2, r1);
    D1 = randn(K, r1);
    Zt = randn(r1, J);
    
    for i=1:100
        W = updateW(X, D2, Vt, D1, Zt, I, J, K, r2);
    
        D2 = updateD2(X, W, Vt, Zt, D1, K, r2);
    
        Vt = updateVt(X, W, D1, D2, Zt, I, J, K, r1, r2);
    
        D1 = updateD1(X, W, Vt, Zt, D2, K, r1);
        
        Zt = updateZt(X, W, D1, Vt, D2, I, J, K, r1);

        apprX = zeros(I, J, K);
        for j=1:K
            apprX(:,:,j) = ...
                W * diag(D2(j,:)) * Vt * diag(D1(j,:)) * Zt;
        end
        
        error = frob(X - apprX)^2 / frob(X)^2;
        if(error < 0.000001)
            break
        end
    
    end
    error
end


function [W] = updateW(X, D2, Vt, D1, Zt, I, J, K, r2)
    unfoldX = zeros(I, J*K);
    for i=1:K
        unfoldX(:, (i-1) * J + 1 : i * J) = X(:,:,i);
    end

    F = zeros(r2, J*K);
    for i=1:K
        F(:,(i-1) * J + 1 : i * J) = ...
            diag(D2(i,:)) * Vt * diag(D1(i,:)) * Zt;
    end

    W = unfoldX / F;
end

function [D2] = updateD2(X, W, Vt, Zt, D1, K, r2)
    D2 = zeros(K, r2);
    for k=1:K
        Fk = Zt' * diag(D1(k,:)) * Vt';
        xk = reshape(X(:,:,k), numel(X(:,:,k)), 1);

        D2(k,:) = (kr(Fk, W) \ xk)';
    end
end

function [Vt] = updateVt(X, W, D1, D2, Zt, I, J, K, r1, r2)
    x = zeros(I*J*K, 1);
    for k=1:K
        x((k-1) * I * J + 1 : k * I * J, :) = reshape(X(:,:,k), 1, numel(X(:,:,k)));
    end

    Z = zeros(I*J*K, r1 * r2);
    for i=1:K
        Z((i-1) * I * J + 1 : i * I * J, :) = ...
            kron(Zt'*diag(D1(i,:)), W*diag(D2(i,:)));
    end

    vt = Z \ x; 

    Vt = reshape(vt, r2, r1);
end

function [D1] = updateD1(X, W, Vt, Zt, D2, K, r1)
    D1 = zeros(K, r1);
    for k=1:K
        Fk = W * diag(D2(k,:))' * Vt;
        xk = reshape(X(:,:,k)', numel(X(:,:,k)), 1);

        D1(k,:) = (kr(Fk, Zt') \ xk)';
    end
end

function [Zt] = updateZt(X, W, D1, Vt, D2, I, J, K, r1)
    unfoldX = zeros(I*K, J);
    for i=1:K
        unfoldX((i-1) * I + 1 : i * I,:) = X(:,:,i);
    end

    F = zeros(I * K, r1);
    for i=1:K
        F((i-1) * I + 1 : i * I,:) = ...
            W*diag(D2(i,:))*Vt*diag(D1(i,:));
    end

    Zt = F \ unfoldX;
end
