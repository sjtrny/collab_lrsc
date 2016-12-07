paths = [genpath('libs'), 'common:'];
addpath(paths);

rng(1)

n_cluster = 5;
        
face_num_list = 1:39;
face_num_list(14) = [];

lambda_1 = 0.05;
lambda_2 = 1;
lambda_nn = 0.1;
lambda_l1l2 = 0.01;

% Up to 0.1 works, testing higher
% at 0.2 NN gets worse results than all others
% Limit of 0.15 (PSNR 36, 12)
noise_mag = 0;

num_runs = 10;

missrate_collab_nn = zeros(num_runs, 1);
missrate_collab_l1l2 = zeros(num_runs, 1);
% missrate_lrr_1 = zeros(num_runs, 1);
missrate_lrr_2 = zeros(num_runs, 1);
missrate_lrr_3 = zeros(num_runs, 1);
% missrate_lrr_4 = zeros(num_runs, 1);
diff_collab_nn = zeros(num_runs, 1);
diff_collab_l1l2 = zeros(num_runs, 1);
diff_lrr = zeros(num_runs, 1);
psnr_list = zeros(num_runs, 2);

for k = 1 : num_runs

%     face_inds = randi(length(face_num_list),3,1);
    face_inds = randsample(face_num_list, n_cluster);
    B = [];
    truth = [];
    for j = 1 : n_cluster
        face_string = sprintf('%02d',face_inds(j));
        im = double(imread(['data/faces/yaleB' face_string '_P00A-005E-10.pgm']));
        im = imresize(im, [48 42]);
        faces = repmat(reshape(im, 48*42,1), 1, 20);
        
        B = [B faces];
        truth = [truth ones(20, 1)'*j];
    end

    B = B/255;

    X_im = normalize(B) + randn(size(B))*noise_mag;
    
    X_lap = zeros(48*42, size(B, 2));
    filt = fspecial('log', 10);

    for j = 1 : size(B, 2)
        im = imfilter(reshape(B(:,j), 48, 42), filt, 'replicate');
        X_lap(:, j) = reshape(im, 48*42, 1);
    end

    X_lap_n = normalize(X_lap) + randn(size(B))*noise_mag;
    
    X_sob = zeros(48*42, size(B, 2));
    filt = fspecial('sobel');

    for j = 1 : size(B, 2)
        im = imfilter(reshape(B(:,j), 48, 42), filt, 'replicate');
        X_sob(:, j) = reshape(im, 48*42, 1);
    end
    
    X_sob_n = normalize(X_sob) + randn(size(B))*noise_mag;
    
    psnr_list(k, :) = [psnr(X_lap, X_lap_n), psnr(X_sob, X_sob_n)];

    Z_lrr_lap = lrr_exact_fro(normalize(X_lap_n), lambda_2);
    [clusters_lap,~,~] = ncutW((abs(Z_lrr_lap)+abs(Z_lrr_lap')), n_cluster);
    clusters_lap = condense_clusters(clusters_lap,1);
    missrate_lrr_2(k, 1) = Misclassification(clusters_lap, truth);
    
    Z_lrr_sob = lrr_exact_fro(normalize(X_sob_n), lambda_2);
    [clusters_sob,~,~] = ncutW((abs(Z_lrr_sob)+abs(Z_lrr_sob')), n_cluster);
    clusters_sob = condense_clusters(clusters_sob,1);
    missrate_lrr_3(k, 1) = Misclassification(clusters_sob, truth);

    Xs = {normalize(X_lap_n), normalize(X_sob_n)};

    Z_nn = solve_collab_nn(Xs, [lambda_2, lambda_2], lambda_nn);
    Z_nn_final = sqrt(sum(Z_nn.^2, 3));
    [nn_clusters,~,~] = ncutW((abs(Z_nn_final)+abs(Z_nn_final')), n_cluster);
    nn_clusters = condense_clusters(nn_clusters,1);
    missrate_collab_nn(k, 1) = Misclassification(nn_clusters, truth);

    Z_l1l2 = solve_collab_l1l2(Xs, [lambda_2, lambda_2], lambda_l1l2);
    Z_l1l2_final = sqrt(sum(Z_l1l2.^2, 3));
    [l1l2_clusters,~,~] = ncutW((abs(Z_l1l2_final)+abs(Z_l1l2_final')), n_cluster);
    l1l2_clusters = condense_clusters(l1l2_clusters,1);
    missrate_collab_l1l2(k, 1) = Misclassification(l1l2_clusters, truth);
    
    diff_collab_nn(k, 1) = norm(Z_nn(:,:,1) - Z_nn(:,:,2), 'fro');
    diff_collab_l1l2(k, 1) = norm(Z_l1l2(:,:,1) - Z_l1l2(:,:,2), 'fro');
    diff_lrr(k, 1) = norm(Z_lrr_lap - Z_lrr_sob, 'fro');
end

save(['face_test_noise_' num2str(noise_mag) '.mat' ], 'lambda_2', 'lambda_nn', 'lambda_l1l2', 'missrate_collab_nn', 'missrate_collab_l1l2', 'missrate_lrr_2', 'missrate_lrr_3',...
'diff_collab_nn', 'diff_collab_l1l2', 'diff_lrr', 'noise_mag', 'psnr_list');
