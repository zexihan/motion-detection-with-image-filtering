function img = derivative_filter(img_h, img_w, mask, threshold, images, num_img)
img = zeros(img_h,img_w,num_img);
for i = 1:img_h
    for j = 1:img_w
        pixel_t = reshape(images(i, j, :), [1, num_img]);
        pixel_d = imfilter(pixel_t, mask);
        pixel_d = abs(pixel_d) > threshold;
        img(i, j, :) = reshape(pixel_d, [1, 1, num_img]);
    end
end
end