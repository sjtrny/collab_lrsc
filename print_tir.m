process_tir

mean_error = mean_error * 100;
mean_error = 100 - mean_error;

h1 = plot(1:length(filenames), mean_error(:,1), '-om');
hold on
h2 = plot(1:length(filenames), mean_error(:,2), '-ob');
hold on
h3 = plot(1:length(filenames), mean_error(:,3), '-or');
hold on
h4 = plot(1:length(filenames), mean_error(:,4), '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3, h4], 'LRR (X1)', 'LRR (X2)', 'MLAP', 'LRCRL (Ours)', 'Location','southwest')

xlim([1 6]);
ylim([0 100]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

ylabel('Subspace Clustering Accuracy (SCA)', 'FontSize', 18);

% title('Mean SCE', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_mean.eps');

close all

median_error = median_error * 100;
median_error = 100 - median_error;

h1 = plot(1:length(filenames), median_error(:,1), '-om');
hold on
h2 = plot(1:length(filenames), median_error(:,2), '-ob');
hold on
h3 = plot(1:length(filenames), median_error(:,3), '-or');
hold on
h4 = plot(1:length(filenames), median_error(:,4), '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3, h4], 'LRR (X1)', 'LRR (X2)', 'MLAP', 'LRCRL (Ours)', 'Location','southwest')

xlim([1 6]);
ylim([0 100]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

ylabel('Subspace Clustering Accuracy (SCA)', 'FontSize', 18);

% title('Median SCE', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_median.eps');

close all

min_error = min_error * 100;
min_error = 100 - min_error;

h1 = plot(1:length(filenames), min_error(:,1), '-om');
hold on
h2 = plot(1:length(filenames), min_error(:,2), '-ob');
hold on
h3 = plot(1:length(filenames), min_error(:,3), '-or');
hold on
h4 = plot(1:length(filenames), min_error(:,4), '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3, h4], 'LRR (X1)', 'LRR (X2)', 'MLAP', 'LRCRL (Ours)', 'Location','southwest')

xlim([1 6]);
ylim([0 100]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

ylabel('Subspace Clustering Accuracy (SCA)', 'FontSize', 18);

% title('Minimum SCE', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_min.eps');

close all

max_error = max_error * 100;
max_error = 100 - max_error;

h1 = plot(1:length(filenames), max_error(:,1), '-om');
hold on
h2 = plot(1:length(filenames), max_error(:,2), '-ob');
hold on
h3 = plot(1:length(filenames), max_error(:,3), '-or');
hold on
h4 = plot(1:length(filenames), max_error(:,4), '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3, h4], 'LRR (X1)', 'LRR (X2)', 'MLAP', 'LRCRL (Ours)', 'Location','southwest')

xlim([1 6]);
ylim([0 100]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

ylabel('Subspace Clustering Accuracy (SCA)', 'FontSize', 18);

% title('Maximum SCE', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_max.eps');

close all

std_d_error = std_d_error * 100;
std_d_error = 100 - std_d_error;

h1 = plot(1:length(filenames), std_d_error(:,1), '-om');
hold on
h2 = plot(1:length(filenames), std_d_error(:,2), '-ob');
hold on
h3 = plot(1:length(filenames), std_d_error(:,3), '-or');
hold on
h4 = plot(1:length(filenames), std_d_error(:,4), '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3, h4], 'LRR (X1)', 'LRR (X2)', 'MLAP', 'LRCRL (Ours)', 'Location','southwest')

xlim([1 6]);
ylim([0 100]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

ylabel('Subspace Clustering Accuracy (SCA)', 'FontSize', 18);

% title('Standard Deviation of SCE', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_std.eps');

close all

h1 = plot(1:length(filenames), mean_diff_lrr, '-ob');
hold on
h2 = plot(1:length(filenames), mean_diff_l1l2, '-or');
hold on
h3 = plot(1:length(filenames), mean_diff_nn, '-ok');

set(gca, 'fontsize', 14);

legend([h1, h2, h3], 'LRR', 'MLAP', 'LRCRL (Ours)', 'Location','northwest')

xlim([1 6]);
ylim([0 3]);

set(gca, 'XTick', 1:length(filenames))
set(gca, 'XTickLabel', mean(mean_psnr, 2))

xlabel('PSNR', 'FontSize', 18);

% ylabel('NORM OF DIFF', 'FontSize', 18);
% 
% title('Difference between each Z_i', 'FontSize', 20);

print(gcf, '-depsc2', 'tir_diff.eps');

close all