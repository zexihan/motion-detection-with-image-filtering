function t = compute_threshold_single_pixel(images, mask, x, y, img_num)
pixel_t = reshape(images(x, y, :), [1, img_num]);
pixel_d = imfilter(pixel_t, mask);
t = 3 * std(pixel_d);
end