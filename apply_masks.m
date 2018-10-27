function image = apply_masks(images, filtimages, img_h, img_w, num_img)
images_show = zeros(img_h, img_w, 3, num_img);
for k = 1:num_img
    images_show(:, :, :, k) = images(:, :, :, k);
    for i = 1:img_h
        for j = 1:img_w
            if filtimages(i, j, k) == 1
                images_show(i, j, 1, k) = 247;
                images_show(i, j, 2, k) = 194;
                images_show(i, j, 3, k) = 66;
            end
        end
    end
end

image = images_show;

end