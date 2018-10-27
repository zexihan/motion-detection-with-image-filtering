function t = compute_threshold_whole_image(images, filter, img_h, img_w, img_num)
noise = zeros(img_h, img_w, img_num);
for i = 1:img_h
    for j = 1:img_w
        pixel_t = reshape(images(i, j, 1:img_num), [1, img_num]);
        pixel_d = imfilter(pixel_t, filter);
        noise(i,j,:) = pixel_d;
    end
end
noise = reshape(noise, [1, img_h * img_w * img_num]);
t = 3 * std(noise);
end