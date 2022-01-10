% Part i)
M = 512;
N = 256;

image_i = ones(M, N);
image_i_fft = fft2(image_i);
figure;
image(fftshift(image_i_fft), 'CDataMapping','scaled');

% Part ii)
image_ii = zeros(M,N);
a = 0;
for m = 1:M
    for n = 1:N
        image_ii(m,n) = sin(20*pi*m/M) + cos(6*pi*n/N);
    end
end
figure;
image(image_ii, 'CDataMapping','scaled')
image_ii_fft = fft2(image_ii);
figure;
image(fftshift(abs(image_ii_fft)), 'CDataMapping','scaled');

% Part iii)
image_iii = zeros(M,N);
a = 0;
for m = 1:M
    for n = 1:N
        image_iii(m,n) = sin(20*pi*m/M)*cos(6*pi*n/N);
    end
end
figure;
image(image_iii, 'CDataMapping','scaled')
image_iii_fft = fft2(image_iii);
figure;
image(fftshift(abs(image_iii_fft)), 'CDataMapping','scaled');