%Program for Peak Signal to Noise Ratio Calculation

%Author : Athi Narayanan S
%M.E, Embedded Systems,
%K.S.R College of Engineering
%Erode, Tamil Nadu, India.
%http://sites.google.com/site/athisnarayanan/
%s_athi1983@yahoo.co.in

function val = psnr(origImg, distImg, max)

if ~exist('max', 'var')
    max = 1;
end

origImg = double(origImg);
distImg = double(distImg);

[M, N] = size(origImg);
error = origImg - distImg;
MSE = sum(sum(error .* error)) / (M * N);

if(MSE > 0)
    val = 10*log(max/MSE);
else
    val = 99;
end