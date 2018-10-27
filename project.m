% Project 1 Motion Detection Using Simple Image Filtering
% Quanwei Hao, Zexi Han

% i. Read in a sequence of image frames and make them grayscale
% 'RedChair', 240x320; 
% 'Office', 240x320;
% 'EnterExitCrossingPaths2cor', 288x384

% Import images and convert to grayscale
file_path = 'RedChair/';
img_path_list = dir(strcat(file_path, '*.jpg'));
num_img = length(img_path_list);

% Set image height and width
img_h = 240;
img_w = 320;

images_cl = zeros(img_h, img_w, 3, num_img);
images_gs = zeros(img_h, img_w, num_img);
figure;
if num_img > 0
    for i = 1:num_img
        image_name = img_path_list(i).name;
        image = imread(strcat(file_path, image_name));
        images_cl(:, :, :, i) = image;
        image_grayscale = rgb2gray(image); 
        images_gs(:, :, i) = image_grayscale; 
        imshow(uint8(images_cl(:, :, :, i)));
    end
end

% Define 0.5[-1,0,1] filter.
filter_1d = 0.5 * [-1, 0, 1];

% Define 1D derivative of a Gaussian filter.
% sigma = 4, x = linspace(-9, 9, 5)
% sigma = 3, x = linspace(-9, 9, 5)
% sigma = 2, x = linspace(-9, 9, 5)
sigma = 3;
sigma_str = strcat("_sigma_", string(sigma));
x = linspace(-2, 2, 5);
y = gaussmf(x, [sigma, 0]);
filter_dGaus_1d = gradient(y);

% ii. Apply a 1-D differential operator at each pixel to compute a 
% temporal derivative.

% Automaticly compute the threshold based on the standard deviation of the
% temporal gradient of the noise.

% Type 1 threshold for temporal derivative filters.
threshold_1d = compute_threshold_whole_image(images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d = compute_threshold_whole_image(images_gs, filter_dGaus_1d, img_h, img_w, num_img);

% Type 2 threshold for temporal derivative filters.
threshold_1d_2 = compute_threshold_single_pixel(images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d_2 = compute_threshold_single_pixel(images_gs, filter_dGaus_1d, img_h, img_w, num_img);

fprintf("Simple\n");
fprintf("Type 1 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d);
fprintf("Type 1 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d);
fprintf("Type 2 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d_2);
fprintf("Type 2 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 0.5[-1,0,1] filter.
filtered_images_gs_1d = derivative_filter(img_h, img_w, filter_1d, threshold_1d, images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/simple/1d/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks to the original images.
images_show = apply_masks(images_cl,filtered_images_gs_1d,img_h,img_w,num_img);

% Display and save results.
figure;
output_path_cl = strcat('output/',file_path,'result/simple/1d/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name), 'jpg');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 1D derivative of a Gaussian filter.
filtered_images_gs_dGaus_1d = derivative_filter(img_h,img_w, filter_dGaus_1d, threshold_dGaus_1d, images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/simple/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_dGaus_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_dGaus_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks to the original images.
images_show = apply_masks(images_cl, filtered_images_gs_dGaus_1d, img_h, img_w, num_img);

% Display and save results.
figure;
output_path_cl = strcat('output/',file_path,'result/simple/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end


% ii. Apply a 2D spatial smoothing filter to the frames before applying the 
% temporal derivative filter.

% (1) 2D spatial smoothing: 3x3 box filter.
smoothed_images_gs = zeros(img_h, img_w, num_img);

% Image smoothing.
for k = 1:num_img
    smoothed_images_gs(:, :, k) = imboxfilt(images_gs(:, :, k), 3);
end

% Automaticly compute the threshold based on the standard deviation of the
% temporal gradient of the noise.

% Type 1 threshold for temporal derivative filters.
threshold_1d = compute_threshold_whole_image(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d = compute_threshold_whole_image(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

% Type 2 threshold for temporal derivative filters.
threshold_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

fprintf("3x3 box filter + derivative filters\n");
fprintf("Type 1 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d);
fprintf("Type 1 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d);
fprintf("Type 2 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d_2);
fprintf("Type 2 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 0.5[-1,0,1] filter.
filtered_images_gs_1d = derivative_filter(img_h, img_w, filter_1d, threshold_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/3x3/1d/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks
images_show = apply_masks(images_cl, filtered_images_gs_1d, img_h, img_w, num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/3x3/1d/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D Derivative Gaussian filter
filtered_images_gs_dGaus_1d = derivative_filter(img_h, img_w, filter_dGaus_1d, threshold_dGaus_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/3x3/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_dGaus_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_dGaus_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks.
images_show = apply_masks(images_cl, filtered_images_gs_dGaus_1d, img_h, img_w, num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/3x3/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end

% (2) 2D spatial smoothing: 5x5 box filter.

% Image smoothing.
for k = 1:num_img
    smoothed_images_gs(:, :, k) = imboxfilt(images_gs(:, :, k), 5);
end

% Automaticly compute the threshold based on the standard deviation of the
% temporal gradient of the noise.

% Type 1 threshold for temporal derivative filters.
threshold_1d = compute_threshold_whole_image(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d = compute_threshold_whole_image(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

% Type 2 threshold for temporal derivative filters.
threshold_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

fprintf("5x5 box filter + derivative filters\n");
fprintf("Type 1 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d);
fprintf("Type 1 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d);
fprintf("Type 2 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d_2);
fprintf("Type 2 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 0.5[-1,0,1] filter
filtered_images_gs_1d = derivative_filter(img_h, img_w, filter_1d, threshold_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/5x5/1d/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks.
images_show = apply_masks(images_cl, filtered_images_gs_1d, img_h, img_w, num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/5x5/1d/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D Derivative Gaussian filter
filtered_images_gs_dGaus_1d = derivative_filter(img_h, img_w, filter_dGaus_1d, threshold_dGaus_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/5x5/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_dGaus_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_dGaus_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks.
images_show = apply_masks(images_cl, filtered_images_gs_dGaus_1d, img_h, img_w, num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/5x5/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end

% (3) Spatial smoothing: 2D Gaussian filter.

% Image smoothing.
for k = 1:num_img
    smoothed_images_gs(:, :, k) = imgaussfilt(images_gs(:, :, k), 5);
end

% Automaticly compute the threshold based on the standard deviation of the
% temporal gradient of the noise.

% Type 1 threshold for temporal derivative filters.
threshold_1d = compute_threshold_whole_image(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d = compute_threshold_whole_image(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

% Type 2 threshold for temporal derivative filters.
threshold_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_1d, img_h, img_w, num_img);
threshold_dGaus_1d_2 = compute_threshold_single_pixel(smoothed_images_gs, filter_dGaus_1d, img_h, img_w, num_img);

fprintf("2D Gaussian filter + derivative filters\n");
fprintf("Type 1 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d);
fprintf("Type 1 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d);
fprintf("Type 2 threshold for 0.5[-1,0,1] filter %.2f\n", threshold_1d_2);
fprintf("Type 2 threshold for 1D derivative of a Gaussian filter %.2f\n", threshold_dGaus_1d_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 0.5[-1,0,1] filter.
filtered_images_gs_1d = derivative_filter(img_h, img_w, filter_1d, threshold_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/gaussian/1d/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks.
images_show = apply_masks(images_cl,filtered_images_gs_1d,img_h,img_w,num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/gaussian/1d/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply 1D Derivative Gaussian filter
filtered_images_gs_dGaus_1d = derivative_filter(img_h, img_w, filter_dGaus_1d, threshold_dGaus_1d, smoothed_images_gs, num_img);

% Display and save masks.
figure;
output_path_gs = strcat('output/',file_path,'mask/gaussian/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(filtered_images_gs_dGaus_1d(:, :, k)), [0 1]);
    imwrite(filtered_images_gs_dGaus_1d(:, :, k), strcat(output_path_gs, img_path_list(k).name));
end

% Apply masks.
images_show = apply_masks(images_cl, filtered_images_gs_dGaus_1d, img_h, img_w, num_img);

% Display and save the results.
figure;
output_path_cl = strcat('output/',file_path,'result/gaussian/dGaus_1d',sigma_str,'/');
for k = 1:num_img
    imshow(uint8(images_show(:, :, :, k)));
    imwrite(uint8(images_show(:, :, :, k)), strcat(output_path_cl, img_path_list(k).name));
end