% path
addpath(genpath('./affine_fit'));
addpath(genpath('./plane_line_intersect'));
addpath(genpath('./geom3d/'));
data_path = '../result_gen/';
d = dir([data_path, '*.mat']);
save_path = '../result_gen_depth/';
im_h = 512;
im_w = 1024;
gt_h = 256;%512; smaller size for rendering speed
gt_w = 512;%1024;

for i = 1:numel(d)

    disp(i);

    %load prediction
    load([data_path d(i).name]); 
    % get depth
    trans_eval2;
    % recenter
    xyz(:,2) = xyz(:,2) - c_h;
    xyz = [xyz(:,1) xyz(:,3) xyz(:,2)];
    
    % check if camera center is out of layout boundary, if it's out, then set as zero
    in = inpolygon(0,0,xyz(1:2:end,1),xyz(1:2:end,2));
    if ~in
        im_depth = zeros(im_h, im_w);
        save([save_path d(i).name], 'im_depth');
        continue

    % top plane
    [n1,~,p1] = affine_fit(xyz(1:2:end,:));
    % bottom plane
    [n2,~,p2] = affine_fit(xyz(2:2:end,:));
    n_all = [n1';n2'];
    p_all =[p1;p2];
    poly = zeros(size(cor_id,1),9);
    for j = 1:2:size(cor_id,1)-2
        poly_t= xyz(j:j+2,:);
        poly(j,:) = poly_t(:);
        poly_t= xyz(j+1:j+3,:);
        poly(j+1,:) = poly_t(:);
    end
    poly_t= xyz([j+2,j+3,1],:);
    poly(end-1,:) = poly_t(:);
    poly_t= xyz([j+3,1,2],:);
    poly(end,:) = poly_t(:);
    
    if 1
    % ray casting
    im_depth = zeros(gt_h, gt_w);
    [im_X,im_Y] = meshgrid(1:gt_w, 1:gt_h);
    im_cor = [im_X(:),im_Y(:)];
    [uv_im] = coords2uv(im_cor, gt_w, gt_h);
    [ xyz_im ] = uv2xyzN(uv_im);
    cen = [0 0 0];
    
    for j = 1:size(xyz_im,1)
        check  = zeros(2+size(cor_id,1)/2,1);
        I = zeros(2+size(cor_id,1)/2,3);
        [I1,check1]=plane_line_intersect(n_all(1,:),p_all(1,:),cen,xyz_im(j,:)*100);
        [I2,check2]=plane_line_intersect(n_all(2,:),p_all(2,:),cen,xyz_im(j,:)*100);
        I(1,:) = I1; check(1) = check1;
        I(2,:) = I2; check(2) = check2;
        ray = [cen xyz_im(j,:)];
        for k = 1:2:size(poly,1)
            [inter1, inside1]= intersectRayPolygon3d(ray, reshape(poly(k,:),3,3));
            [inter2, inside2]= intersectRayPolygon3d(ray, reshape(poly(k+1,:),3,3));
            %[I1,check1]=plane_line_intersect(n_all(k,:),p_all(k,:),cen,xyz_im(j,:)*100);
            if sum(isnan(inter1))==0
                check((k+1)/2+2)= 1;
                I((k+1)/2+2,:)= inter1;
            elseif sum(isnan(inter2))==0
                check((k+1)/2+2)= 1;
                I((k+1)/2+2,:)= inter2;
            end
        end
        id = find(check == 1);
        
        if isempty(id)
           im_depth(im_cor(j,2), im_cor(j,1)) = im_depth(im_cor(j-1,2), im_cor(j-1,1));
           continue
        end
        dist = I(id,:);
        dist = sqrt(sum(dist.*dist,2));
        [dep, idx] = min(dist);
        im_depth(im_cor(j,2), im_cor(j,1)) = dep;
    end
    %keyboard
    % save
    save([save_path d(i).name], 'im_depth');
end
end
