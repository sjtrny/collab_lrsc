paths = [genpath('libs'), 'common:'];
addpath(paths);

load('data/TIRlib');

rng(1)

n_spectra = 5;

n_cluster = 4;
n_space = n_cluster;
cluster_size = 100;

truth = reshape(repmat(1:n_cluster,cluster_size,1),1,n_cluster*cluster_size)';

B1 = zeros(321,n_cluster*cluster_size);

for j = 1 : n_cluster
    spectra_indices = randi(size(A,2),n_spectra,1);

    spectra_proportions = rand(1,n_spectra);
    spectra_sum = sum(spectra_proportions);
    spectra_proportions = spectra_proportions/spectra_sum;

    spectra = A(:, spectra_indices) * spectra_proportions';

    sub_b = repmat(spectra, [1 cluster_size]);

    B1(:,((j-1)*cluster_size)+1:j*cluster_size) = sub_b;

end

B2 = zeros(321,n_cluster*cluster_size);

for j = 1 : n_cluster
    spectra_indices = randi(size(A,2),n_spectra,1);

    spectra_proportions = rand(1,n_spectra);
    spectra_sum = sum(spectra_proportions);
    spectra_proportions = spectra_proportions/spectra_sum;

    spectra = A(:, spectra_indices) * spectra_proportions';

    sub_b = repmat(spectra, [1 cluster_size]);

    B2(:,((j-1)*cluster_size)+1:j*cluster_size) = sub_b;

end


noise_mag = 0;

noise = randn(size(B1));
w = noise * noise_mag;
X1 = B1 + w;

noise = randn(size(B2));
w = noise * noise_mag;
X2 = B2 + w;

X_1normed = normalize(X1);
X_2normed = normalize(X2);

Xs = {X_1normed, X_2normed};

tic;
[Z_nn] = solve_collab_nn(Xs, 1, 1);
toc;

Z_nn_final = sqrt(sum(Z_nn.^2, 3));

[collab_clusters_nn,~,~] = ncutW((abs(Z_nn_final)+abs(Z_nn_final')), n_space);
clusters_collab_nn = condense_clusters(collab_clusters_nn,1);
missrate_collab_nn = Misclassification(clusters_collab_nn, truth);
