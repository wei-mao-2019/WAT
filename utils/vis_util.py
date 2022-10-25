import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import smplx
import smplx
import smplx
from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config
import cv2

import os

def draw_skeleton_old(ax, kpts, parents=[], c='r', marker='o', line_style='-'):
    """

    :param kpts: joint_n*(3 or 2)
    :param parents:
    :return:
    """
    # ax = plt.subplot(111)
    joint_n, dims = kpts.shape
    # by default it is human 3.6m joints
    # [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]
    if len(parents) == 0:
        parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    if parents == 'op':
        parents = [1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if dims > 2:
        ax.view_init(75, 90)
        ax.set_zlabel('Z Label')
    if dims == 2:
        idx_choosed = np.intersect1d(np.where(kpts[:, 0] > 0)[0], np.where(kpts[:, 1] > 0)[0])
        ax.scatter(kpts[idx_choosed, 0], kpts[idx_choosed, 1], c=c, marker=marker, s=2,
                   alpha=0.7)
        # for i in idx_choosed:
        #     ax.text(kpts[i, 0], kpts[i, 1], "{:d}".format(i), color=c)
    else:
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c=c, marker=marker, s=2,
                   alpha=0.7)
        # for i in range(kpts.shape[0]):
        #     ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], "{:d}".format(i), color=c)

    for i in range(len(parents)):
        if parents[i] < 0:
            continue
        if dims == 2:
            if not (parents[i] in idx_choosed and i in idx_choosed):
                continue

        if dims == 2:
            ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=c, linestyle=line_style,
                    alpha=0.7)
        else:
            ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                    [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=c,
                    alpha=0.7)

    return None


def draw_skeleton(ax, kpts, parents=[], is_right=[], cols=["#3498db", "#e74c3c"], marker='o', line_style='-',
                  label=None):
    """

    :param kpts: joint_n*(3 or 2)
    :param parents:
    :return:
    """
    # ax = plt.subplot(111)
    joint_n, dims = kpts.shape
    # by default it is human 3.6m joints
    # [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]
    # if len(parents) == 0:
    #     parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    #     is_right = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    #     if cols == []:
    #         cols = ["#3498db", "#e74c3c"]
    # if parents == 'op':
    #     parents = [1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]
    # if parents == 'smpl':
    #     # parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    #     parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    #     # is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    #     is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    #     if cols == []:
    #         cols = ["#3498db", "#e74c3c"]
    # if parents == 'smpl_add':
    #     parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15]
    #     is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    #     cols = ["#3498db", "#e74c3c"]
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    if dims > 2:
        ax.view_init(75, 90)
        ax.set_zlabel('Z Label')
    # if dims == 2:
    #     # idx_choosed = np.intersect1d(np.where(kpts[:, 0] > 0)[0], np.where(kpts[:, 1] > 0)[0])
    #     # ax.scatter(kpts[idx_choosed, 0], kpts[idx_choosed, 1], c=c, marker=marker, s=10)
    #     ax.scatter(kpts[:, 0], kpts[:, 1], c=cols[0], marker=marker, s=10)
    #     # for i in idx_choosed:
    #     #     ax.text(kpts[i, 0], kpts[i, 1], "{:d}".format(i), color=c)
    # else:
    #     ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c=cols[0], marker=marker, s=10)
    #     for i in range(kpts.shape[0]):
    #         ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], "{:d}".format(i), color=cols[0])
    is_label = True
    for i in range(len(parents)):
        if parents[i] < 0:
            continue
        # if dims == 2:
        #     if not (parents[i] in idx_choosed and i in idx_choosed):
        #         continue

        if dims == 2:
            # ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
            #         linestyle=line_style,
            #         alpha=0.5 if is_right[i] else 1, linewidth=3)
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6)
        else:
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3)
                ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], f"J{i}", color=cols[0])

    return None


def draw_skeleton_smpl(ax, kpts, parents=[], c='r', marker='o', line_style='-'):
    """

    :param kpts: joint_n*(3 or 2)
    :param parents:
    :return:
    """
    # ax = plt.subplot(111)
    joint_n, dims = kpts.shape
    # by default it is human 3.6m joints
    # [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]
    if len(parents) == 0:
        parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        is_right = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        cols = ["#3498db", "#e74c3c"]
    if parents == 'op':
        parents = [1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]
    if parents == 'smpl':
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
        cols = ["#3498db", "#e74c3c"]
    if parents == 'smpl_add':
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15]
        is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        cols = ["#3498db", "#e74c3c"]
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if dims > 2:
        ax.view_init(75, 90)
        ax.set_zlabel('Z Label')
    if dims == 2:
        idx_choosed = np.intersect1d(np.where(kpts[:, 0] > 0)[0], np.where(kpts[:, 1] > 0)[0])
        ax.scatter(kpts[idx_choosed, 0], kpts[idx_choosed, 1], c=c, marker=marker, s=10)
        # for i in idx_choosed:
        #     ax.text(kpts[i, 0], kpts[i, 1], "{:d}".format(i), color=c)
    else:
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c=c, marker=marker, s=10)
        # for i in range(kpts.shape[0]):
        #     ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], "{:d}".format(i), color=c)

    for i in range(len(parents)):
        if parents[i] < 0:
            continue
        if dims == 2:
            if not (parents[i] in idx_choosed and i in idx_choosed):
                continue

        if dims == 2:
            ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                    linestyle=line_style,
                    alpha=0.7)
        else:
            ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                    [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]])

    return None


def render_videos(sequence, device, save_path, key,
                  model_path='./SMPL_models/',
                  w_golbalrot=True, his_frame=10):
    mv = MeshViewer(offscreen=True, width=600, height=900,bg_color=[1.0, 1.0, 1.0, 1.0])
    # mv.viewer.viewer_flags['record'] = True
    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    # camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    camera_pose[:3, 3] = np.array([-.6, -2.4, .3])
    mv.update_camera_pose(camera_pose)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seq_data = sequence['poses']
    gender = sequence.get('gender','male') #str(sequence['gender'])
    T = seq_data.shape[0]

    sbj_m = smplx.create(model_path=model_path,
                         model_type='smplh',
                         gender=gender,
                         # num_pca_comps=24,
                         # v_template=sbj_vtemp,
                         use_pca=False,
                         batch_size=T,
                         ext='pkl').to(device)
    sbj_m.pose_mean[:] = 0
    sbj_m.use_pca = False
    sbj_params = {}
    if w_golbalrot:
        global_rot = np.zeros_like(seq_data[:, :3])
        global_rot[:] = np.array([1.5,0,0])
        sbj_params['global_orient'] = global_rot
        # sbj_params['global_orient'] = seq_data[:, :3]
        sbj_params['body_pose'] = seq_data[:, 3:66]
        # sbj_params['jaw_pose'] = seq_data[:, 66:69]
        # sbj_params['leye_pose'] = seq_data[:, 69:72]
        # sbj_params['reye_pose'] = seq_data[:, 72:75]
        sbj_params['left_hand_pose'] = seq_data[:, 66:111]
        sbj_params['right_hand_pose'] = seq_data[:, 111:156]
        # sbj_params['transl'] = sequence['trans']
    else:
        global_rot = np.zeros_like(seq_data[:, :3])
        global_rot[:] = np.array([1.5,0,0])
        sbj_params['global_orient'] = global_rot
        sbj_params['body_pose'] = seq_data[:, :63]
        # sbj_params['jaw_pose'] = seq_data[:, 66:69]
        # sbj_params['leye_pose'] = seq_data[:, 69:72]
        # sbj_params['reye_pose'] = seq_data[:, 72:75]
        sbj_params['left_hand_pose'] = seq_data[:, 63:108]
        sbj_params['right_hand_pose'] = seq_data[:, 108:153]
        # sbj_params['transl'] = sequence['trans']

    sbj_params['betas'] = sequence.get('betas', np.random.randn(10))[None, :10]

    sbj_parms = params2torch(sbj_params, device=device)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    skip_frame = 1
    imgs = []
    for frame in range(0, T, skip_frame):
        # o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)
        plt.cla()
        if frame < his_frame:
            col = colors['pink']
        else:
            col = colors['orange']
        s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=col, smooth=True)
        # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

        # t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        # mv.set_static_meshes([o_mesh, s_mesh, t_mesh])
        mv.set_static_meshes([s_mesh])
        col, _ = mv.viewer.render(mv.scene)
        imgs.append(col[:, :, [2, 1, 0]])
    import cv2
    video_name = f'{save_path}/{key}.avi'
    # images = [img for img in os.listdir(path_tmp) if img.endswith(".jpg")]
    # frame = cv2.imread(os.path.join(path_tmp, images[0]))
    height, width, layers = imgs[0].shape

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, frameSize=(width, height), fps=30)
    for image in imgs:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()



def render_videos_new(sequence, device, save_path, key,
                  model_path='./SMPL_models/',
                  w_golbalrot=True, his_frame=10, smpl_model='smpl'):
    mv = MeshViewer(offscreen=True, width=600, height=900,bg_color=[1.0, 1.0, 1.0, 1.0])
    # mv.viewer.viewer_flags['record'] = True
    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    # camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    camera_pose[:3, 3] = np.array([-.6, -2.4, .3])
    mv.update_camera_pose(camera_pose)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seq_data = sequence['poses']
    gender = sequence.get('gender','male') #str(sequence['gender'])
    T = seq_data.shape[0]

    sbj_m = smplx.create(model_path=model_path,
                         model_type=smpl_model,
                         gender=gender,
                         # num_pca_comps=24,
                         # v_template=sbj_vtemp,
                         use_pca=False,
                         batch_size=T,
                         ext='pkl').to(device)
    if smpl_model != 'smpl':
        sbj_m.pose_mean[:] = 0
    sbj_m.use_pca = False
    sbj_params = {}

    global_rot = np.zeros_like(seq_data[:, :3])
    global_rot[:] = np.array([1.5, 0, 0])
    sbj_params['global_orient'] = global_rot
    # sbj_params['global_orient'] = seq_data[:, :3]

    # if w_golbalrot:
    if smpl_model == 'smplh':
        sbj_params['body_pose'] = seq_data[:, 3:66]
        # sbj_params['jaw_pose'] = seq_data[:, 66:69]
        # sbj_params['leye_pose'] = seq_data[:, 69:72]
        # sbj_params['reye_pose'] = seq_data[:, 72:75]
        sbj_params['left_hand_pose'] = seq_data[:, 66:111]
        sbj_params['right_hand_pose'] = seq_data[:, 111:156]
        # sbj_params['transl'] = sequence['trans']
    elif smpl_model == 'smplx':
        sbj_params['body_pose'] = seq_data[:, 3:66]
        sbj_params['jaw_pose'] = seq_data[:, 66:69]
        sbj_params['leye_pose'] = seq_data[:, 69:72]
        sbj_params['reye_pose'] = seq_data[:, 72:75]
        sbj_params['left_hand_pose'] = seq_data[:, 75:120]
        sbj_params['right_hand_pose'] = seq_data[:, 120:165]
        sbj_params['expression'] = np.zeros_like(seq_data[:,:10])
    elif smpl_model == 'smpl':
        sbj_params['body_pose'] = seq_data[:, 3:72]

    # else:
    #     global_rot = np.zeros_like(seq_data[:, :3])
    #     global_rot[:] = np.array([1.5,0,0])
    #     sbj_params['global_orient'] = global_rot
    #     sbj_params['body_pose'] = seq_data[:, :63]
    #     # sbj_params['jaw_pose'] = seq_data[:, 66:69]
    #     # sbj_params['leye_pose'] = seq_data[:, 69:72]
    #     # sbj_params['reye_pose'] = seq_data[:, 72:75]
    #     sbj_params['left_hand_pose'] = seq_data[:, 63:108]
    #     sbj_params['right_hand_pose'] = seq_data[:, 108:153]
    #     # sbj_params['transl'] = sequence['trans']

    sbj_params['betas'] = sequence.get('betas', np.random.randn(10))[None, :10]

    sbj_parms = params2torch(sbj_params, device=device)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    skip_frame = 1
    imgs = []
    for frame in range(0, T, skip_frame):
        # o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)
        plt.cla()
        if frame < his_frame:
            col = colors['pink']
        else:
            col = colors['orange']
        s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=col, smooth=True)
        # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

        # t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        # mv.set_static_meshes([o_mesh, s_mesh, t_mesh])
        mv.set_static_meshes([s_mesh])
        col, _ = mv.viewer.render(mv.scene)
        imgs.append(col[:, :, [2, 1, 0]])
    import cv2
    video_name = f'{save_path}/{key}.avi'
    # images = [img for img in os.listdir(path_tmp) if img.endswith(".jpg")]
    # frame = cv2.imread(os.path.join(path_tmp, images[0]))
    height, width, layers = imgs[0].shape

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, frameSize=(width, height), fps=30)
    for image in imgs:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()
