filenames = {'0', '0.025', '0.05', '0.075', '0.1', '0.125', '0.15'};

mean_error = zeros(length(filenames), 4);
median_error = zeros(length(filenames), 4);
max_error = zeros(length(filenames), 4);
min_error = zeros(length(filenames), 4);
std_d_error = zeros(length(filenames), 4);
mean_diff_l1l2 = zeros(length(filenames), 1);
mean_diff_nn = zeros(length(filenames), 1);
mean_diff_lrr = zeros(length(filenames), 1);
mean_psnr = zeros(length(filenames), 2);

for k = 1 : length(filenames)
    load(['face_test_noise_' filenames{k} '.mat'])
    
    mean_error(k,:) = [ mean(missrate_lrr_2), mean(missrate_lrr_3), mean(missrate_collab_l1l2), mean(missrate_collab_nn)];
    median_error(k,:) = [median(missrate_lrr_2), median(missrate_lrr_3), median(missrate_collab_l1l2), median(missrate_collab_nn)];
    max_error(k,:) = [max(missrate_lrr_2), max(missrate_lrr_3), max(missrate_collab_l1l2), max(missrate_collab_nn)];
    min_error(k,:) = [min(missrate_lrr_2), min(missrate_lrr_3), min(missrate_collab_l1l2), min(missrate_collab_nn)];
%     std_d_error(k,:) = [std(missrate_lrr_2), std(missrate_lrr_3), std(missrate_collab_l1l2), std(missrate_collab_nn)];
    mean_diff_l1l2(k) = mean(diff_collab_l1l2);
    mean_diff_nn(k) = mean(diff_collab_nn);
    mean_diff_lrr(k) = mean(diff_lrr);

    mean_psnr(k,:) = mean(psnr_list); 
   
end

clear 'missrate_lrr_1' 'missrate_lrr_2' 'missrate_lrr_3' 'missrate_collab_l1l2' 'missrate_collab_nn' 'diff_collab_l1l2'...
    'diff_collab_nn' 'diff_lrr' 'psnr_list' 'noise_mag' 'lambda_1' 'lambda_2' 'lambda2_l1l2' 'lambda2_nn'