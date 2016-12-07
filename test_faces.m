paths = [genpath('libs'), 'common:'];
addpath(paths);

rng(1)

n_cluster = 5;
        
face_num_list = 1:39;
face_num_list(14) = [];

lambda_1 = 0.05;
lambda_2 = 1;
lambda_nn = 0.1;
lambda_l1l2 = 0.1;

% Absolute max 0.2
noise_mag = 0.05;

num_runs = 10;

missrate_collab_nn = zeros(num_runs, 1);
missrate_collab_l1l2 = zeros(num_runs, 1);
missrate_lrr_1 = zeros(num_runs, 1);
missrate_lrr_2 = zeros(num_runs, 1);
missrate_lrr_3 = zeros(num_runs, 1);
% missrate_lrr_4 = zeros(num_runs, 1);
diff_collab_nn = zeros(num_runs, 1);
diff_collab_l1l2 = zeros(num_runs, 1);
diff_lrr = zeros(num_runs, 1);
psnr_list = zeros(num_runs, 1);

for k = 1 : num_runs

%     face_inds = randi(length(face_num_list),3,1);
    face_inds = randsample(face_num_list, n_cluster);
    B = [];
    truth = [];
    for j = 1 : n_cluster
        face_string = sprintf('%02d',face_inds(j));
        [faces, num_faces] = load_faces(['B' face_string]);
        B = [B faces];
        truth = [truth ones(20, 1)'*j];
    end

    B = B/255;

    X_im = B + randn(size(B))*noise_mag;
    
    psnr_list(k) = psnr(B, X_im);

    
    X_lap = zeros(48*42, size(B, 2));
    filt = fspecial('log', 10);

    for j = 1 : size(B, 2)
        im = imfilter(reshape(X_im(:,j), 48, 42), filt, 'replicate');
        X_lap(:, j) = reshape(im, 48*42, 1);
    end

    X_sob = zeros(48*42, size(B, 2));
    filt = fspecial('sobel');

    for j = 1 : size(B, 2)
        im = imfilter(reshape(X_im(:,j), 48, 42), filt, 'replicate');
        X_sob(:, j) = reshape(im, 48*42, 1);
    end


    Z_lrr_im = lrr_exact_fro(normalize(X_im), lambda_1);
    [clusters_im,~,~] = ncutW((abs(Z_lrr_im)+abs(Z_lrr_im')), n_cluster);
    clusters_im = condense_clusters(clusters_im,1);
    missrate_lrr_1(k, 1) = Misclassification(clusters_im, truth);

    Z_lrr_lap = lrr_exact_fro(normalize(X_lap), lambda_2);
    [clusters_lap,~,~] = ncutW((abs(Z_lrr_lap)+abs(Z_lrr_lap')), n_cluster);
    clusters_lap = condense_clusters(clusters_lap,1);
    missrate_lrr_2(k, 1) = Misclassification(clusters_lap, truth);
    
    Z_lrr_sob = lrr_exact_fro(normalize(X_sob), lambda_2);
    [clusters_sob,~,~] = ncutW((abs(Z_lrr_sob)+abs(Z_lrr_sob')), n_cluster);
    clusters_sob = condense_clusters(clusters_sob,1);
    missrate_lrr_3(k, 1) = Misclassification(clusters_sob, truth);

    X_im_2 = B + randn(size(B))*noise_mag;

    Xs = {normalize(X_im), normalize(X_im_2)};

    Z_nn = solve_collab_nn(Xs, [lambda_1, lambda_1], lambda_nn);
    Z_nn_final = sqrt(sum(Z_nn.^2, 3));
    [nn_clusters,~,~] = ncutW((abs(Z_nn_final)+abs(Z_nn_final')), n_cluster);
    nn_clusters = condense_clusters(nn_clusters,1);
    missrate_collab_nn(k, 1) = Misclassification(nn_clusters, truth);

    Z_l1l2 = solve_collab_l1l2(Xs, [lambda_1, lambda_1], lambda_l1l2);
    Z_l1l2_final = sqrt(sum(Z_l1l2.^2, 3));
    [l1l2_clusters,~,~] = ncutW((abs(Z_l1l2_final)+abs(Z_l1l2_final')), n_cluster);
    l1l2_clusters = condense_clusters(l1l2_clusters,1);
    missrate_collab_l1l2(k, 1) = Misclassification(l1l2_clusters, truth);
    
    diff_collab_nn(k, 1) = norm(Z_nn(:,:,1) - Z_nn(:,:,2), 'fro');
    diff_collab_l1l2(k, 1) = norm(Z_l1l2(:,:,1) - Z_l1l2(:,:,2), 'fro');
    diff_lrr(k, 1) = norm(Z_lrr_im - Z_lrr_lap, 'fro');
    
end

save(['face_test_noise_' num2str(noise_mag) '.mat' ], 'lambda_1', 'lambda_2', 'lambda_nn', 'lambda_l1l2', 'missrate_collab_nn', 'missrate_collab_l1l2', 'missrate_lrr_1', 'missrate_lrr_2', 'missrate_lrr_3',...
'diff_collab_nn', 'diff_collab_l1l2', 'diff_lrr', 'noise_mag', 'psnr_list');
