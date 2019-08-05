import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

from pano import pano_connect_points


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xy(coor, z=50, coorW=1024, coorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u)
    y = -c * np.cos(u)
    return np.hstack([x[:, None], y[:, None]])


def tri2halfspace(pa, pb, p):
    v1 = pa - p
    v2 = pb - p
    vn = np.cross(v1, v2)
    if -vn @ p > 0:
        vn = -vn
    return [*vn, -vn @ p]


def xyzlst2halfspaces(xyz_floor, xyz_ceil):
    '''
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)


def eval_3diou(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor,
               ch=-1.6, coorW=1024, coorH=512):
    '''
    Evaluate 3D IoU of "convex layout".
    Instead of voxelization, this function use halfspace intersection
    to evaluate the volume.
    Input parameters:
        dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor
    have to be in shape [N, 2] and in the format of:
        [[x, y], ...]
    listing the corner position from left to right on the equirect image.
    '''
    dt_floor_coor = np.array(dt_floor_coor)
    dt_ceil_coor = np.array(dt_ceil_coor)
    gt_floor_coor = np.array(gt_floor_coor)
    gt_ceil_coor = np.array(gt_ceil_coor)
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
    N = len(dt_floor_coor)
    dt_floor_xyz = np.hstack([
        np_coor2xy(dt_floor_coor, ch, coorW, coorH),
        np.zeros((N, 1)) + ch,
    ])
    gt_floor_xyz = np.hstack([
        np_coor2xy(gt_floor_coor, ch, coorW, coorH),
        np.zeros((N, 1)) + ch,
    ])
    dt_c = np.sqrt((dt_floor_xyz[:, :2] ** 2).sum(1))
    gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
    dt_v2 = np_coory2v(dt_ceil_coor[:, 1], coorH)
    gt_v2 = np_coory2v(gt_ceil_coor[:, 1], coorH)
    dt_ceil_z = dt_c * np.tan(dt_v2)
    gt_ceil_z = gt_c * np.tan(gt_v2)

    dt_ceil_xyz = dt_floor_xyz.copy()
    dt_ceil_xyz[:, 2] = dt_ceil_z
    gt_ceil_xyz = gt_floor_xyz.copy()
    gt_ceil_xyz[:, 2] = gt_ceil_z

    dt_halfspaces = xyzlst2halfspaces(dt_floor_xyz, dt_ceil_xyz)
    gt_halfspaces = xyzlst2halfspaces(gt_floor_xyz, gt_ceil_xyz)

    in_halfspaces = HalfspaceIntersection(np.concatenate([dt_halfspaces, gt_halfspaces]),
                                          np.zeros(3))
    dt_halfspaces = HalfspaceIntersection(dt_halfspaces, np.zeros(3))
    gt_halfspaces = HalfspaceIntersection(gt_halfspaces, np.zeros(3))

    in_volume = ConvexHull(in_halfspaces.intersections).volume
    dt_volume = ConvexHull(dt_halfspaces.intersections).volume
    gt_volume = ConvexHull(gt_halfspaces.intersections).volume
    un_volume = dt_volume + gt_volume - in_volume

    return in_volume / un_volume


def eval_PE(dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor, H=512, W=1024):
    '''
    Evaluate pixel surface error (3 labels: ceiling, wall, floor)
    Input parameters:
        dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor
    have to be in shape [N, 2] and in the format of:
        [[x, y], ...]
    listing the corner position from left to right on the equirect image.
    '''
    y0 = np.zeros(W)
    y1 = np.zeros(W)
    y0_gt = np.zeros(W)
    y1_gt = np.zeros(W)
    for j in range(dt_ceil_coor.shape[0]):
        coorxy = pano_connect_points(dt_ceil_coor[j], dt_ceil_coor[(j+1)%4], -50)
        y0[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(dt_floor_coor[j], dt_floor_coor[(j+1)%4], 50)
        y1[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_ceil_coor[j], gt_ceil_coor[(j+1)%4], -50)
        y0_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_floor_coor[j], gt_floor_coor[(j+1)%4], 50)
        y1_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

    surface = np.zeros((H, W), dtype=np.int32)
    surface[np.round(y0).astype(int), np.arange(W)] = 1
    surface[np.round(y1).astype(int), np.arange(W)] = 1
    surface = np.cumsum(surface, axis=0)
    surface_gt = np.zeros((H, W), dtype=np.int32)
    surface_gt[np.round(y0_gt).astype(int), np.arange(W)] = 1
    surface_gt[np.round(y1_gt).astype(int), np.arange(W)] = 1
    surface_gt = np.cumsum(surface_gt, axis=0)

    return (surface != surface_gt).sum() / (H * W), surface, surface_gt


def augment(x_img, flip, rotate):
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for rotate in rotate:
        shift = int(round(rotate * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return np.array(x_imgs_augmented), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs = []
    for x_img, aug in zip(x_imgs_augmented, aug_type):
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()
    return np.array(x_imgs)
