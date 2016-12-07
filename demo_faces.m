paths = [genpath('libs'), 'common:'];
addpath(paths);

run('libs/vlfeat/toolbox/vl_setup');

rng(1)

n_cluster = 3;
        
face_num_list = 1:39;
face_num_list(14) = [];

noise_mag = 0.2;

face_inds = randsample(face_num_list, 3);
B = [];
truth = [];
for j = 1 : n_cluster
    face_string = sprintf('%02d',face_inds(j));
    [faces, num_faces] = load_faces(['B' face_string]);
    B = [B faces];
    truth = [truth ones(10, 1)'*j];
end

B = B/255;

X_im = B + randn(size(B))*noise_mag;

X_hog = zeros(6*5*31, size(B, 2));

for k = 1 : size(B, 2)
    hog = vl_hog(single(reshape(X_im(:,k), 48, 42)), 8) ;

    X_hog(:, k) = reshape(hog, 6*5*31, 1);
end

X_hog = normalize(X_hog);

X_lap = zeros(48*42, size(B, 2));
filt = fspecial('log');

for k = 1 : size(B, 2)
    im = imfilter(reshape(X_im(:,k), 48, 42), filt, 'replicate');
    X_lap(:, k) = reshape(im, 48*42, 1);
end

X_lap = normalize(X_lap);

Z_lrr_lap = lrr_exact_fro(normalize(X_lap), 1);
[clusters_lap,~,~] = ncutW((abs(Z_lrr_lap)+abs(Z_lrr_lap')), n_cluster);
clusters_lap = condense_clusters(clusters_lap,1);
figure, imagesc(clusters_lap);

X_sob = zeros(48*42, size(B, 2));
filt = fspecial('sobel');

for k = 1 : size(B, 2)
    im = imfilter(reshape(X_im(:,k), 48, 42), filt, 'replicate');
    X_sob(:, k) = reshape(im, 48*42, 1);
end

X_sob = normalize(X_sob);


Xs = {normalize(X_im), normalize(X_lap)};

Z_lrr_collab = solve_collab_nn(Xs, [0.05, 1], 0.1);
Z_nn_final = sqrt(sum(Z_lrr_collab.^2, 3));
[collab_clusters,~,~] = ncutW((abs(Z_nn_final)+abs(Z_nn_final')), n_cluster);
clusters_collab = condense_clusters(collab_clusters,1);
figure, imagesc(clusters_collab);

