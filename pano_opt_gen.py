import torch
import torch.optim as optim
import numpy as np
from PIL import Image
#import pano
import pano_gen as pano
import time

def vecang(vec1, vec2):
    vec1 = vec1 / np.sqrt((vec1 ** 2).sum())
    vec2 = vec2 / np.sqrt((vec2 ** 2).sum())
    return np.arccos(np.dot(vec1, vec2))


def rotatevec(vec, theta):
    x = vec[0] * torch.cos(theta) - vec[1] * torch.sin(theta)
    y = vec[0] * torch.sin(theta) + vec[1] * torch.cos(theta)
    return torch.cat([x, y])


def pts_linspace(pa, pb, pts=300):
    pa = pa.view(1, 2)
    pb = pb.view(1, 2)
    w = torch.arange(0, pts + 1, dtype=pa.dtype).view(-1, 1)
    return (pa * (pts - w) + pb * w) / pts


def xyz2uv(xy, z=-1):
    c = torch.sqrt((xy ** 2).sum(1))
    u = torch.atan2(xy[:, 1], xy[:, 0]).view(-1, 1)
    v = torch.atan2(torch.zeros_like(c) + z, c).view(-1, 1)
    return torch.cat([u, v], dim=1)


def uv2idx(uv, w, h):
    col = (uv[:, 0] / (2 * np.pi) + 0.5) * w - 0.5
    row = (uv[:, 1] / np.pi + 0.5) * h - 0.5
    return torch.cat([col.view(-1, 1), row.view(-1, 1)], dim=1)


def wallidx(xy, w, h, z1, z2):
    col = (torch.atan2(xy[1], xy[0]) / (2 * np.pi) + 0.5) * w - 0.5
    c = torch.sqrt((xy ** 2).sum())
    row_s = (torch.atan2(torch.zeros_like(c) + z1, c) / np.pi + 0.5) * h - 0.5
    row_t = (torch.atan2(torch.zeros_like(c) + z2, c) / np.pi + 0.5) * h - 0.5

    pa = torch.cat([col.view(1), row_s.view(1)])
    pb = torch.cat([col.view(1), row_t.view(1)])
    return pts_linspace(pa, pb)


def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (H, W)
    coordinates: (2, ...)
    '''
    h = input.shape[0]
    w = input.shape[1]

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    f00 = input[co_floor[0], co_floor[1]]
    f10 = input[co_floor[0], co_ceil[1]]
    f01 = input[co_ceil[0], co_floor[1]]
    f11 = input[co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    return fx1 + d2 * (fx2 - fx1)


def pc2cor_id(pc, pc_vec, pc_theta, pc_height):
    
    if pc_theta.numel()==1:
        ps = torch.stack([
            (pc + pc_vec),
            (pc + rotatevec(pc_vec, pc_theta)),
            (pc - pc_vec),
            (pc + rotatevec(pc_vec, pc_theta - np.pi))
        ])
    else:
        ps = pc + pc_vec
        ps = ps.view(-1,2)
        for c_num in range(pc_theta.shape[1]):
            ps = torch.cat((ps, ps[c_num:,:]),0)
            if (c_num % 2) == 0:
                ps[-1,1] = pc_theta[0,c_num]
            else:
                ps[-1,0] = pc_theta[0,c_num]
        ps = torch.cat((ps, ps[-1:,:]),0)
        ps[-1,1] = ps[0,1]

    return torch.cat([
        uv2idx(xyz2uv(ps, z=-1), 1024, 512),
        uv2idx(xyz2uv(ps, z=pc_height), 1024, 512),
    ], dim=0)


def project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i_step=None):

    # Sample corner loss
    corid = pc2cor_id(pc, pc_vec, pc_theta, pc_height)
    corid_coordinates = torch.stack([corid[:, 1], corid[:, 0]])
    loss_cor = -map_coordinates(scorecor, corid_coordinates).mean()

    # Sample boundary loss
    if pc_theta.numel()==1:
        p1 = pc + pc_vec
        p2 = pc + rotatevec(pc_vec, pc_theta)
        p3 = pc - pc_vec
        p4 = pc + rotatevec(pc_vec, pc_theta - np.pi)

        segs = [
            pts_linspace(p1, p2),
            pts_linspace(p2, p3),
            pts_linspace(p3, p4),
            pts_linspace(p4, p1),
        ]
    else:
        ps = pc + pc_vec
        ps = ps.view(-1,2)
        for c_num in range(pc_theta.shape[1]):
            ps = torch.cat((ps, ps[c_num:,:]),0)
            if (c_num % 2) == 0:
                ps[-1,1] = pc_theta[0,c_num]
            else:
                ps[-1,0] = pc_theta[0,c_num]
        ps = torch.cat((ps, ps[-1:,:]),0)
        ps[-1,1] = ps[0,1]
        segs = []
        for c_num in range(ps.shape[0]-1):
            segs.append(pts_linspace(ps[c_num,:], ps[c_num+1,:]))
        segs.append(pts_linspace(ps[-1,:], ps[0,:]))

    # ceil-wall
    loss_ceilwall = 0
    for seg in segs:
        ceil_uv = xyz2uv(seg, z=-1)
        ceil_idx = uv2idx(ceil_uv, 1024, 512)
        ceil_coordinates = torch.stack([ceil_idx[:, 1], ceil_idx[:, 0]])
        loss_ceilwall -= map_coordinates(scoreedg[..., 1], ceil_coordinates).mean() / len(segs)

    # floor-wall
    loss_floorwall = 0
    for seg in segs:
        floor_uv = xyz2uv(seg, z=pc_height)
        floor_idx = uv2idx(floor_uv, 1024, 512)
        floor_coordinates = torch.stack([floor_idx[:, 1], floor_idx[:, 0]])
        loss_floorwall -= map_coordinates(scoreedg[..., 2], floor_coordinates).mean() / len(segs)

    #losses = 1.0 * loss_cor + 0.1 * loss_wallwall + 0.5 * loss_ceilwall + 1.0 * loss_floorwall
    losses = 1.0 * loss_cor + 1.0 * loss_ceilwall + 1.0 * loss_floorwall

    if i_step is not None:
        with torch.no_grad():
            print('step %d: %.3f (cor %.3f, wall %.3f, ceil %.3f, floor %.3f)' % (
                i_step, losses,
                loss_cor, loss_wallwall,
                loss_ceilwall, loss_floorwall))

    return losses


def optimize_cor_id(cor_id, scoreedg, scorecor, num_iters=100, verbose=False):
    assert scoreedg.shape == (512, 1024, 3)
    assert scorecor.shape == (512, 1024)

    Z = -1
    ceil_cor_id = cor_id[0::2]
    floor_cor_id = cor_id[1::2]
    
    ceil_cor_id, ceil_cor_id_xy = pano.constraint_cor_id_same_z(ceil_cor_id, scorecor, Z)
    #ceil_cor_id_xyz = np.hstack([ceil_cor_id_xy, np.zeros(4).reshape(-1, 1) + Z])
    ceil_cor_id_xyz = np.hstack([ceil_cor_id_xy, np.zeros(ceil_cor_id.shape[0]).reshape(-1, 1) + Z])

    # TODO: revise here to general layout
    #pc = (ceil_cor_id_xy[0] + ceil_cor_id_xy[2]) / 2
    #print(ceil_cor_id_xy)
    if abs(ceil_cor_id_xy[0,0]-ceil_cor_id_xy[1,0])>abs(ceil_cor_id_xy[0,1]-ceil_cor_id_xy[1,1]):
        ceil_cor_id_xy = np.concatenate((ceil_cor_id_xy[1:,:],ceil_cor_id_xy[:1,:]), axis=0)
    #print(cor_id)
    #print(ceil_cor_id_xy)
    pc = np.mean(ceil_cor_id_xy, axis=0)
    pc_vec = ceil_cor_id_xy[0] - pc
    pc_theta = vecang(pc_vec, ceil_cor_id_xy[1] - pc)
    pc_height = pano.fit_avg_z(floor_cor_id, ceil_cor_id_xy, scorecor)
    
    if ceil_cor_id_xy.shape[0] > 4:
        pc_theta = np.array([ceil_cor_id_xy[1,1]])
        for c_num in range(2, ceil_cor_id_xy.shape[0]-1):
            if (c_num % 2) == 0:
                pc_theta = np.append(pc_theta, ceil_cor_id_xy[c_num,0])
            else:
                pc_theta = np.append(pc_theta, ceil_cor_id_xy[c_num,1])

    scoreedg = torch.FloatTensor(scoreedg)
    scorecor = torch.FloatTensor(scorecor)
    pc = torch.FloatTensor(pc)
    pc_vec = torch.FloatTensor(pc_vec)
    pc_theta = torch.FloatTensor([pc_theta])
    pc_height = torch.FloatTensor([pc_height])
    pc.requires_grad = True
    pc_vec.requires_grad = True
    pc_theta.requires_grad = True
    pc_height.requires_grad = True

    #print(pc_theta)
    #time.sleep(2)
    #return cor_id
    optimizer = optim.SGD([
        pc, pc_vec, pc_theta, pc_height
    ], lr=1e-3, momentum=0.9)

    best = {'score': 1e9}

    for i_step in range(num_iters):
        i = i_step if verbose else None
        optimizer.zero_grad()
        score = project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i)
        if score.item() < best['score']:
            best['score'] = score.item()
            best['pc'] = pc.clone()
            best['pc_vec'] = pc_vec.clone()
            best['pc_theta'] = pc_theta.clone()
            best['pc_height'] = pc_height.clone()
        score.backward()
        optimizer.step()

    pc = best['pc']
    pc_vec = best['pc_vec']
    pc_theta = best['pc_theta']
    pc_height = best['pc_height']
    opt_cor_id = pc2cor_id(pc, pc_vec, pc_theta, pc_height).detach().numpy()
    split_num = int(opt_cor_id.shape[0]//2)
    opt_cor_id = np.stack([opt_cor_id[:split_num], opt_cor_id[split_num:]], axis=1).reshape(split_num*2, 2)

    #print(opt_cor_id)
    #print(cor_id)
    #time.sleep(500)
    return opt_cor_id
