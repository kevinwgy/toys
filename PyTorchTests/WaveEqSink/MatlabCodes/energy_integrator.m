function [int_U, int_S] = energy_integrator(U, S, x, int_loc, simcode)

%%%%%%%%%%%% generating MID vector
dx = mean(diff(x));
n_mats = length(int_loc); n = length(x);
int_id = zeros(n_mats, n);

for i = 1:n_mats
    int_id(i, :) = int_loc(i) < x;
end
MID = sum(int_id, 1) + 1;
MID_ind = find(diff(MID)); MID_ind = [0 MID_ind length(MID)];

int_U = zeros(height(U), n_mats+1);
int_S = zeros(height(U), n_mats+1);

for i = 1:height(U)
    for j = 1:n_mats+1
        u = U(i, MID_ind(j)+1:MID_ind(j+1));
        s = S(i, MID_ind(j)+1:MID_ind(j+1));
        int_U(i, j) = sum(u*dx);
        int_S(i, j) = sum(s*dx);
    end
end

% int_U = [int_U sum(int_U, 2)];
% int_S = [int_S sum(int_S, 2)];
