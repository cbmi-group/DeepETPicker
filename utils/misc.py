import torch
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import dilation
from scipy.spatial import distance
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print, stat_print
from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
import scikitplot as skplt
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# combine block batches to whole images
def combine(data, shape, block_size=72, pad_size=18, reverse=False):
    if reverse:
        shape = shape[::-1]
    try:
        union_data = np.zeros(shape, dtype=np.float)
    except:
        union_data = np.zeros(shape, dtype=np.float32)

    step_size = block_size - 2 * pad_size
    block_size = step_size

    for i in range(shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0)):
        for j in range(shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0)):
            for k in range(shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0)):
                if i == shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0) - 1:
                    x = shape[0] - block_size // 2
                else:
                    x = i * step_size + block_size // 2

                if j == shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0) - 1:
                    y = shape[1] - block_size // 2
                else:
                    y = j * step_size + block_size // 2

                if k == shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0) - 1:
                    z = shape[2] - block_size // 2
                else:
                    z = k * step_size + block_size // 2

                union_data[x - block_size // 2: x + block_size // 2,
                y - block_size // 2: y + block_size // 2,
                z - block_size // 2: z + block_size // 2] = data[
                    i * (shape[1] // block_size + (1 if shape[1] % step_size > 0 else 0)) *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + j *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + k]

    return union_data


def combine_torch(data, shape, block_size=72, pad_size=18, reverse=False):
    if reverse:
        shape = shape[::-1]
    union_data = torch.zeros(shape, device=data.device)
    step_size = block_size - 2 * pad_size
    block_size = step_size

    for i in range(shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0)):
        for j in range(shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0)):
            for k in range(shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0)):
                if i == shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0) - 1:
                    x = shape[0] - block_size // 2
                else:
                    x = i * step_size + block_size // 2

                if j == shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0) - 1:
                    y = shape[1] - block_size // 2
                else:
                    y = j * step_size + block_size // 2

                if k == shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0) - 1:
                    z = shape[2] - block_size // 2
                else:
                    z = k * step_size + block_size // 2

                union_data[x - block_size // 2: x + block_size // 2,
                y - block_size // 2: y + block_size // 2,
                z - block_size // 2: z + block_size // 2] = data[
                    i * (shape[1] // block_size + (1 if shape[1] % step_size > 0 else 0)) *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + j *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + k]

    return union_data


def cal_metrics_OneCls(pred, ocp, gt_coords, threshold, border_value, particle_volume=0):
    pred_mask = np.where(pred < threshold, 0, 1)
    z_max, y_max, x_max = pred.shape
    pred_mask = dilation(pred_mask)
    pred_mask = label(pred_mask, connectivity=2)
    pred_props = regionprops(pred_mask)

    print(len(pred_props))
    if len(pred_props) == 0:
        return 0, 0, 0, 0
    else:
        min_vol = particle_volume * 0.05
        # max_vol = particle_volume * 5
        max_vol = 1e10
        centroids = []
        b_value = border_value
        for j in range(len(pred_props)):
            if max_vol >= pred_props[j].area >= min_vol:
                if ((z_max - b_value) > pred_props[j].centroid[0] > b_value
                        and (y_max - b_value) > pred_props[j].centroid[1] > b_value
                        and (x_max - b_value) > pred_props[j].centroid[2] > b_value):
                    # psi = np.pi ** (1/3.) * (6 * pred_props[j].area) ** (2/3.)/ (pred_props[j].perimeter)
                    centroids.append([pred_props[j].centroid[2],
                                      pred_props[j].centroid[1],
                                      pred_props[j].centroid[0],
                                      pred_props[j].area,
                                      pred_props[j].equivalent_diameter,
                                      pred_props[j].label,
                                      pred_props[j].major_axis_length,
                                      pred_props[j].minor_axis_length,
                                      pred_props[j].extent])
        centroids = np.array(centroids)

        gt_particles = [(0, 0, 0)]  # start with a "background" particle
        for i in range(len(gt_coords)):
            x, y, z = gt_coords[i]
            gt_particles.append((int(x), int(y), int(z)))
        n_gt_particles = len(gt_particles) - 1
        gt_particles = np.array(gt_particles)


        dist = 0
        k = 0
        k_dist = []
        for i in range(centroids.shape[0]):
            p_gt_id = int(ocp[int(centroids[i][2]),
                              int(centroids[i][1]),
                              int(centroids[i][0])])
            if p_gt_id != 0:
                p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]
                p_distance = np.abs(
                    distance.euclidean((centroids[i][0], centroids[i][1], centroids[i][2]), (p_gt_x, p_gt_y, p_gt_z)))
                k = k + 1
                dist = dist + p_distance
                k_dist.append([1, p_distance])
            else:
                k_dist.append([0, 0])

        # centroids = np.concatenate((centroids, np.array(k_dist)), axis=1)
        avg_dist = dist / (k + 1e-7)
        precision = k / (centroids.shape[0] + 1e-7)
        recall = k / (gt_particles.shape[0] - 1 + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return precision, recall, f1, avg_dist


def cal_metrics_NMS_OneCls(pred_coords, gt_coords, occupancy_map, cfg):
    pred_coords = coord_duplication(pred_coords[:, 1:], cfg["ocp_diameter"])
    centroids = np.array(pred_coords)

    gt_particles = [(0, 0, 0)]  # start with a "background" particle
    for i in range(len(gt_coords)):
        x, y, z = gt_coords[i]
        gt_particles.append((int(x), int(y), int(z)))
    n_gt_particles = len(gt_particles) - 1
    gt_particles = np.array(gt_particles)


    dist = 0
    k = 0
    k_dist = []
    for i in range(centroids.shape[0]):
        p_gt_id = int(occupancy_map[int(centroids[i][2]),
                          int(centroids[i][1]),
                          int(centroids[i][0])])
        if p_gt_id != 0:
            p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]
            p_distance = np.abs(
                distance.euclidean((centroids[i][0], centroids[i][1], centroids[i][2]), (p_gt_x, p_gt_y, p_gt_z)))
            k = k + 1
            dist = dist + p_distance
            k_dist.append([1, p_distance])
        else:
            k_dist.append([0, 0])

    # centroids = np.concatenate((centroids, np.array(k_dist)), axis=1)
    avg_dist = dist / (k + 1e-7)
    precision = k / (centroids.shape[0] + 1e-7)
    recall = k / (gt_particles.shape[0] - 1 + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return precision, recall, f1, avg_dist


def get_centroids(pred, threshold, border_value, particle_volume=0, cls_idx=0):
    pred_mask = np.where(pred < threshold, 0, 1)
    z_max, y_max, x_max = pred.shape

    pred_mask = label(pred_mask, connectivity=1)
    pred_props = regionprops(pred_mask)

    # print(len(pred_props))
    if len(pred_props) == 0 or len(pred_props) > 10000:
        return [[0, 0, 0, 0, 0]]
    else:
        min_vol = particle_volume * 0.2
        max_vol = particle_volume * 5
        # print(min_vol, max_vol, len(pred_props))
        centroids = []
        b_value = border_value
        for j in range(len(pred_props)):
            if pred_props[j].area >= min_vol:
                if ((z_max - b_value) > pred_props[j].centroid[0] > b_value
                        and (y_max - b_value) > pred_props[j].centroid[1] > b_value
                        and (x_max - b_value) > pred_props[j].centroid[2] > b_value):
                    # psi = np.pi ** (1/3.) * (6 * pred_props[j].area) ** (2/3.)/ (pred_props[j].perimeter)
                    centroids.append([cls_idx,
                                      pred_props[j].centroid[2],
                                      pred_props[j].centroid[1],
                                      pred_props[j].centroid[0],
                                      pred_props[j].area])
        return centroids


def cal_metrics_MultiCls(pred, gt, occupancy_map, cfg, args, pad_size, dir_name, particle_volume):
    # Conversion dicts
    pred = np.array(pred)

    if 'classes' in cfg.keys():
        classes = cfg["classes"]
    else:
        classes = range(args.num_classes)

    num2pdb = {k: v for k, v in enumerate(classes)}
    pdb2num = {v: k for k, v in num2pdb.items()}

    # Loading ground truth particles
    gt_particles = [('0', 0, 0, 0)]  # start with a "background" particle
    for i in range(len(gt)):
        pdb_id, x, y, z, *_ = gt[i]
        gt_particles.append((num2pdb[pdb_id], int(x), int(y), int(z)))

    n_gt_particles = len(gt_particles) - 1

    # print(gt)
    """
    fmt1: d_max = 5, 同类别间去重 仅删除面积最小的一个
    fmt2: d_max = 3, 同类别间去重，已被去掉的忽略计算 仅删除面积最小的一个
    fmt3: d_max = 3, 同类别间去重 仅保留最大面积粒子，删除其他粒子
    fmt4: d_max = 5, 全类别间去重 仅保留最大面积粒子，删除其他粒子
    """
    mini_dist = sorted([int(i)//2+1 for i in cfg["ocp_diameter"].split(',')])[0]
    if args.de_duplication:
        print('de_duplication')
        indexs = []
        areas = np.array(pred)[:, 4]
        pred = np.array(pred)
        try:
            pred_final_ = pred[:, 1:4].astype(np.float)
        except:
            pred_final_ = pred[:, 1:4].astype(np.float32)

        for idx, item in enumerate(pred_final_):
            if idx in indexs:
                continue
            d2 = np.linalg.norm(pred_final_ - item, ord=2, axis=1)
            tmp = (d2 <= mini_dist)
            tmp[idx] = False
            if len(np.nonzero(tmp)[0]) >= 1:
                temp_idx = np.nonzero(tmp)[0].tolist()
                temp_idx.append(idx)
                for idx_, item in enumerate(temp_idx):
                    if item in indexs:
                        del temp_idx[idx_]
                max_idx = np.argmax(areas[temp_idx])
                del temp_idx[max_idx]
                indexs.extend(temp_idx)

        # print('delete indexs:', indexs)
        pred = np.delete(pred, indexs, axis=0)
        # print(pred)

    # Loading occupancy map (voxel -> particle) and morphologically dilate it as a way to close holes
    predicted_particles = []
    for i in range(len(pred)):
        pdb, x, y, z, *_ = pred[i]
        if pdb in num2pdb.keys():
            predicted_particles.append((num2pdb[pdb], int(round(float(x))), int(round(float(y))), int(round(float(z)))))
    n_predicted_particles = len(predicted_particles)

    # Init of some vars for statistics
    # number of particles that were predicted to be outside of tomogram
    n_clipped_predicted_particles = 0
    # reported classes and distances for each GT particle
    found_particles = [[] for _ in range(len(gt_particles))]

    occupancy_map = dilation(occupancy_map)
    border_value = 3
    max_x, max_y, max_z = occupancy_map.shape[::-1]
    for p_i, (p_pdb, *coordinates) in enumerate(predicted_particles):
        if coordinates[0] < border_value or coordinates[1] < border_value or coordinates[2] < border_value or \
            coordinates[0] > max_x - border_value or \
            coordinates[1] > max_y - border_value or \
            coordinates[2] > max_z - border_value:
            continue
        p_x, p_y, p_z = coordinates

        if [p_x, p_y, p_z] != coordinates:
            n_clipped_predicted_particles += 1

        # Find ground truth particle at the predicted location
        # print(p_pdb, (p_x, p_y, p_z))
        # print('shape:', occupancy_map.shape[::-1])
        p_gt_id = int(occupancy_map[p_z, p_y, p_x])
        p_gt_pdb, p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]

        # Compute distance from predicted center to real center
        p_distance = np.abs(distance.euclidean((p_x, p_y, p_z), (p_gt_x, p_gt_y, p_gt_z)))

        # Register found particle, a class it is predicted to be and distance from predicted center to real center
        found_particles[p_gt_id].append((p_pdb, p_distance))

    gt_particles2 = gt_particles.copy()
    n_gt_particles2 = n_gt_particles

    # Compute localization statistics
    n_prediction_missed = len(found_particles[0])
    n_prediction_hit = sum([len(p) for p in found_particles[1:]])
    n_unique_particles_found = sum([int(p >= 1) for p in [len(p) for p in found_particles[1:]]])
    n_unique_particles_not_found = sum([int(p == 0) for p in [len(p) for p in found_particles[1:]]])
    n_unique_particle_with_multiple_hits = sum([int(p > 1) for p in [len(p) for p in found_particles[1:]]])

    n_unique_particles_not_found_idxs = [idx for (idx, p) in [(idx_, len(p)) for (idx_, p) in enumerate(found_particles[1:])] if p ==0]

    localization_recall = n_unique_particles_found / n_gt_particles2
    localization_precision = n_unique_particles_found / (n_predicted_particles + 1e-10)
    localization_f1 = 1 / ((1 / (localization_recall + 1e-10) + 1 / (localization_precision + 1e-10)) / 2)
    # localization_miss_rate = n_unique_particles_not_found / n_gt_particles2
    localization_miss_rate = 1 - localization_recall
    localization_avg_distance = sum([p[0][1] for p in found_particles[1:] if len(p) > 0]) / (n_unique_particles_found + 1e-10)

    # Compute classification statistics and confusion matrix
    gt_particle_classes = np.asarray([pdb2num[p[0]] for p in gt_particles2[1:]], dtype=int)
    predicted_particle_classes = np.asarray([pdb2num[p[0][0]] if p else 0 for p in found_particles[1:]], dtype=int)

    confusion_matrix = ConfusionMatrix(actual_vector=gt_particle_classes, predict_vector=predicted_particle_classes)
    f1_scores = confusion_matrix.class_stat["F1"]
    print(f1_scores)
    f1_res = []
    for i in range(1, args.num_classes):
        f1_res.append(f1_scores[i])
    mean_f1 = np.mean(np.array(f1_res))

    # tmp = "|"
    # f1_sum = 0
    # cls_temp = []
    # for class_name in cfg['classes_sort'][1:]:
    #     tmp += "%0.3f|" % float(f1_scores[class_name])
    #     f1_sum += float(f1_scores[class_name])
    #     cls_temp.append(float(f1_scores[class_name]))
    # f1_mean = (f1_sum / len(cfg['classes_sort'][1:]))
    # tmp += "%0.3f|" % (f1_mean)
    # cls_temp.append(f1_mean)
    # print(f'de_dup={args.de_dup_fmt}, thresh={args.threshold}, cls metrics:', tmp)

    # print(gt_particle_classes)
    # print('haha')
    # print(predicted_particle_classes)
    # print('haha')
    if args.checkpoints != None and len(predicted_particle_classes) > 0:
        model_name = args.checkpoints.split('/')[-4] + '_' + args.checkpoints.split('/')[-1].split('-')[0]
        dataset = cfg["dset_name"]
        save_dir = f"result/{dataset}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/{dir_name}/loc_cls', exist_ok=True)
        os.makedirs(f'{save_dir}/{dir_name}/particle_locations', exist_ok=True)

        np.savetxt(
            f'{save_dir}/{dir_name}/particle_locations/coords_padsize{pad_size}_thresh{args.threshold}.txt',
            pred, delimiter="\t", newline="\n", fmt="%s")

        dif_elements = predicted_particle_classes.tolist()
        dif_elements.extend(gt_particle_classes.tolist())
        num2pdb_v1 = {k: v for k, v in num2pdb.items() if k in set(dif_elements)}

        confusion_matrix = ConfusionMatrix(actual_vector=gt_particle_classes, predict_vector=predicted_particle_classes)
        confusion_matrix.relabel(num2pdb_v1)

        lut_classes = np.asarray(classes)
        skplt.metrics.plot_confusion_matrix(lut_classes[gt_particle_classes],
                                            lut_classes[predicted_particle_classes],
                                            labels=cfg["classes_sort"],
                                            figsize=(20, 20), text_fontsize=18, hide_zeros=True,
                                            hide_counts=False)
        plt.savefig(f'{save_dir}/{dir_name}/loc_cls//plain_cm_padsize{pad_size}_segThresh{args.threshold}.png')

        # Prepare confusion matrix prints
        confusion_matrix_table = table_print(confusion_matrix.classes, confusion_matrix.table)
        confusion_matrix_stats = stat_print(confusion_matrix.classes, confusion_matrix.class_stat,
                                            confusion_matrix.overall_stat, confusion_matrix.digit,
                                            SUMMARY_OVERALL, SUMMARY_CLASS)

        # print(type(confusion_matrix_stats))
        # Format confusion matrix and stats
        confusion_matrix_table = '\t'.join(confusion_matrix_table.splitlines(True))
        confusion_matrix_stats = '\t'.join(confusion_matrix_stats.splitlines(True))

        # Construct a report and write it
        report = f'\n\t### Localization\n' \
                 f'\tSubmission has {n_predicted_particles} predicted particles\n' \
                 f'\tTomogram has {n_gt_particles2} particles\n' \
                 f'\tTP: {n_unique_particles_found} unique particles found\n' \
                 f'\tFP: {n_prediction_missed} predicted particles are false positive\n' \
                 f'\tFN: {n_unique_particles_not_found} unique particles not found\n' \
                 f'\tThere was {n_unique_particle_with_multiple_hits} particles that had more than one prediction\n' \
                 f'\tThere was {n_clipped_predicted_particles} predicted particles that were outside of tomo bounds\n' \
                 f'\tAverage euclidean distance from predicted center to ground truth center: {localization_avg_distance}\n' \
                 f'\tRecall: {localization_recall:.5f}\n' \
                 f'\tPrecision: {localization_precision:.5f}\n' \
                 f'\tMiss rate: {localization_miss_rate:.5f}\n' \
                 f'\tF1 score: {localization_f1:.5f}\n' \
                 f'\n\t### Classification\n' \
                 f'\t{confusion_matrix_table}\n' \
                 f'\t{confusion_matrix_stats}\n\n\n'

        print(report)
        confusion_matrix.save_html(f'{save_dir}/{dir_name}/loc_cls/confusion_matrix_padsize{pad_size}_segThresh{args.threshold}')

        with open(f'{save_dir}/{dir_name}/loc_cls/confusion_matrix_table_padsize{pad_size}_segThresh{args.threshold}.txt', 'w') as f:  # 设置文件对象
            f.write(confusion_matrix_table)  # 将字符串写入文件中

        with open(f'{save_dir}/{dir_name}/loc_cls/confusion_matrix_stats_padsize{pad_size}_segThresh{args.threshold}.txt', 'w') as f:  # 设置文件对象
            f.write(confusion_matrix_stats)  # 将字符串写入文件中

        # print(type(confusion_matrix_stats))
        # print(confusion_matrix_stats)

        localization_res = {'Submission': ['Ours_%d' % pad_size],
                            'RR': [n_predicted_particles],
                            'TP': [n_unique_particles_found],
                            'FP': [n_prediction_missed],
                            'FN': [n_unique_particles_not_found],
                            'MH': [n_unique_particle_with_multiple_hits],
                            'RO': [n_clipped_predicted_particles],
                            'AD': [localization_avg_distance],
                            'Recall': [localization_recall],
                            'Precision': [localization_precision],
                            'Miss rate': [localization_miss_rate],
                            'F1 Score': [localization_f1]}

        df = pd.DataFrame(localization_res)
        df.to_csv(f'{save_dir}/{dir_name}/loc_cls/localization_res_padsize{pad_size}_segThresh{args.threshold}.csv')

        try:
            temp = np.array(df.to_numpy()[0, 1:]).astype(np.float)
        except:
            temp = np.array(df.to_numpy()[0, 1:]).astype(np.float32)

        tmp = "|"
        for idx, item in enumerate(temp):
            if idx <= (5 if not args.skip_vesicles else 4):
                tmp += "%d|" % item
            elif idx > 5:
                tmp += "%0.3f|" % item
            else:
                pass


        print('*' * 100)
        print(f'de_dup={args.de_dup_fmt}, thresh={args.threshold}, localization metrics:', tmp)
        print('*' * 100)
        loc_temp = temp
        loc_tmp = tmp

        # 提取classification_stats数据并保存
        # f1_scores = np.array(list(confusion_matrix.class_stat["F1"].values())[1:]).astype(float)
        f1_scores = confusion_matrix.class_stat["F1"]
        tmp = "|"
        f1_sum = 0
        cls_temp = []
        for class_name in cfg['classes_sort'][1:]:
            tmp += "%0.3f|" % float(f1_scores[class_name])
            f1_sum += float(f1_scores[class_name])
            cls_temp.append(float(f1_scores[class_name]))
        f1_mean = (f1_sum/len(cfg['classes_sort'][1:]))
        tmp += "%0.3f|" % (f1_mean)
        cls_temp.append(f1_mean)
        print(f'de_dup={args.de_dup_fmt}, thresh={args.threshold}, cls metrics:', tmp)

        # 将统计数据保存至excel
        csv_res = np.zeros([2, len(cls_temp)])
        if args.skip_vesicles:
            loc_temp = np.concatenate((loc_temp[:5], loc_temp[6:]))
        csv_res[0, :len(loc_temp)] = loc_temp
        csv_res[1, :len(cls_temp)] = cls_temp
        csv_pd = pd.DataFrame(csv_res)
        writer = pd.ExcelWriter(f'{save_dir}/{dir_name}/loc_cls_official_res_Dup{int(args.de_duplication)}_{args.de_dup_fmt}_padsize{args.pad_size[0]}_meanPool{int(args.meanPool_NMS)}.xlsx')
        csv_pd.to_excel(writer, 'sheet1', float_format='%.3f')
        writer.save()
        writer.close()

        # 将统计数据保存为markdown格式的文本文件
        print('*' * 100)
        with open(f'{save_dir}/{dir_name}/loc_cls_official_res_Dup{int(args.de_duplication)}_{args.de_dup_fmt}_padsize{args.pad_size[0]}_meanPool{int(args.meanPool_NMS)}.txt', 'w') as f:
            f.write(f'de_dup={args.de_dup_fmt}, thresh={args.threshold}, localization metrics:\n{loc_tmp}\n\n')
            f.write(f'de_dup={args.de_dup_fmt}, thresh={args.threshold}, cls metrics:\n{tmp}')

        # print(f"{f1_mean:.3f}({f1_scores[0]:.3f}/{f1_scores[1]:.3f})|--|{args.threshold}{loc_tmp[:-1]}" )

        # plot P, R, F1 for different particles
        classes = cfg["classes_sort"]

        # print(confusion_matrix.class_stat)

        plt.figure(num=2, figsize=(11, 5))
        width = 0.2
        xticks = np.arange(1, args.num_classes+1 if args.use_paf else args.num_classes)
        xticks1 = xticks - width

        if len(xticks) == len(classes):
            y1 = np.array([0 if i is 'None' else i for i in list(confusion_matrix.class_stat["PPV"].values())[1:]]).astype(float)  # Precision
            plt.bar(xticks1, y1, width=width, color='r', label='Precision')
            plt.grid(alpha=0.4)

            xticks2 = xticks
            y2 = np.array([0 if i is 'None' else i for i in list(confusion_matrix.class_stat["TPR"].values())[1:]]).astype(float)  # Recall
            plt.bar(xticks2, y2, width=width, color='g', label='Recall')
            plt.grid(alpha=0.4)

            xticks3 = xticks + width
            y3 = np.array([0 if i is 'None' else i for i in list(confusion_matrix.class_stat["F1"].values())[1:]]).astype(float)
            plt.bar(xticks3, y3, width=width, color='b', label='F1-score')
            plt.xticks(xticks, classes)
            # plt.ylim([0, 1.15])
            plt.grid(alpha=0.4)
            plt.title('classification_metrics_padsize{}_segThresh{}'.format(pad_size, args.threshold))
            # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
            plt.legend()

            plt.savefig(f'{save_dir}/{dir_name}/loc_cls/classification_metrics_padsize{pad_size}_segThresh{args.threshold}.png')

        # calculate the metrics based on different area ratio
        ratios = np.arange(0.0, 2.0, 0.1)
        particle_volume = 0.5 if args.use_paf else particle_volume
        res = []
        for ratio in ratios:
            area_thresh = particle_volume * ratio
            pred_sel = pred[pred[:, 4] > area_thresh]
            pred_sel = np.insert(pred_sel, 5, values=0, axis=1)

            predicted_particles = []
            for i in range(len(pred_sel)):
                pdb, x, y, z, *_ = pred_sel[i]
                if int(pdb) in num2pdb.keys():
                    predicted_particles.append(
                        (num2pdb[int(pdb)], int(round(float(x))), int(round(float(y))), int(round(float(z)))))

            # Init of some vars for statistics
            # number of particles that were predicted to be outside of tomogram
            n_clipped_predicted_particles = 0
            # reported classes and distances for each GT particle
            found_particles = [[] for _ in range(len(gt_particles))]

            for p_i, (p_pdb, *coordinates) in enumerate(predicted_particles):
                if coordinates[0] < border_value or coordinates[1] < border_value or coordinates[2] < border_value or \
                        coordinates[0] > max_x - border_value or \
                        coordinates[1] > max_y - border_value or \
                        coordinates[2] > max_z - border_value:
                    continue
                p_x, p_y, p_z = coordinates
                # p_x, p_y, p_z = np.clip(coordinates, (0, 0, 0), occupancy_map.shape[::-1])

                if [p_x, p_y, p_z] != coordinates:
                    n_clipped_predicted_particles += 1

                # Find ground truth particle at the predicted location
                p_gt_id = int(occupancy_map[p_z, p_y, p_x])
                p_gt_pdb, p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]

                # Compute distance from predicted center to real center
                p_distance = np.abs(distance.euclidean((p_x, p_y, p_z), (p_gt_x, p_gt_y, p_gt_z)))

                if p_gt_pdb == p_pdb:
                    # print(p_gt_pdb, p_pdb)
                    pred_sel[p_i, 5] = 1

                # Register found particle, a class it is predicted to be and distance from predicted center to real center
                found_particles[p_gt_id].append((p_pdb, p_distance))
                # Compute classification statistics and confusion matrix

            gt_particles2 = gt_particles.copy()
            n_gt_particles2 = n_gt_particles
            # if skip 4V94, remove 4V94s from both GT and predicted
            if args.skip_4v94:
                for i in range(len(gt_particles2) - 1, 0, -1):
                    if gt_particles2[i][0] == '4V94':
                        del gt_particles2[i]
                        del found_particles[i]

            if args.skip_vesicles:
                for i in range(len(gt_particles2) - 1, -1, -1):
                    if gt_particles2[i][0].lower() == 'vesicle':
                        del gt_particles2[i]
                        del found_particles[i]

            gt_classes = np.asarray([pdb2num[p[0]] for p in gt_particles2[1:]], dtype=int)
            pred_classes = np.asarray([pdb2num[p[0][0]] if (p and p[0][0] != '4V94') else 0 for p in found_particles[1:]],
                                                        dtype=int)
            precision_m, recall_m, f1_score_m, _ = precision_recall_fscore_support(gt_classes, pred_classes,
                                                                                   average='macro')
            rr = (args.num_classes + 1) / args.num_classes
            precision_m = precision_m * rr
            recall_m = recall_m * rr
            f1_score_m = f1_score_m * rr

            res.append([ratio, area_thresh, precision_m, recall_m, f1_score_m])

            pred_sel = pred_sel[pred_sel[:, 0] > 0]
            np.savetxt(
                f'{save_dir}/{dir_name}/particle_locations/coords_padsize{pad_size}_thresh{args.threshold}_area{ratio:.2f}.txt',
                pred_sel, delimiter="\t", newline="\n", fmt="%s")
            np.savetxt(
                f'{save_dir}/{dir_name}/particle_locations/coords_padsize{pad_size}_thresh{args.threshold}_area{ratio:.2f}.coords',
                pred_sel[:, :4].astype(int), delimiter="\t", newline="\n", fmt="%s")
            np.savetxt(
                f'{save_dir}/{dir_name}/particle_locations/coords_padsize{pad_size}_thresh{args.threshold}_area{ratio:.2f}_eval.txt',
                pred_sel[:, :4].astype(int), delimiter="\t", newline="\n", fmt="%s")

        res = np.array(res)
        np.savetxt(
            f'{save_dir}/{dir_name}/particle_locations/PRF1_DifferentArea_padsize{pad_size}_thresh{args.threshold}.txt',
            res, delimiter="\t", newline="\n", fmt="%s")
        plt.figure(num=3, figsize=(10, 7))
        plt.plot(res[:, 0], res[:, 2], label='precision')
        plt.plot(res[:, 0], res[:, 3], label='recall')
        plt.plot(res[:, 0], res[:, 4], label='f1_score')
        plt.xlabel('area threshold (%)')
        plt.ylabel('preformance')

        plt.xticks(res[:, 0], [int(i) for i in res[:, 0] * 100])
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dir_name}/particle_locations/PRF1_DifferentArea_padsize{pad_size}_thresh{args.threshold}.png')

    return localization_precision, \
           localization_recall, \
           localization_f1, \
           localization_miss_rate, \
           localization_avg_distance, \
           gt_particle_classes, \
           predicted_particle_classes, \
           num2pdb, \
           mean_f1


#
def coord_duplication(pred, mini_dist):
    mini_dist = int(float(mini_dist))
    pred_ = pred[:, :3].astype(float)
    scores = pred[:, 3]
    indexs = []
    if pred_.shape[0] > 0:
        # de_duplication
        # print(pred_, mini_dist)
        for idx, item in enumerate(pred_):
            d2 = np.linalg.norm(pred_ - item, ord=2, axis=1)
            tmp = (d2 < mini_dist)
            tmp[idx] = False
            if len(np.nonzero(tmp)[0]) >= 1:
                temp_idx = np.nonzero(tmp)[0].tolist()
                temp_idx.append(idx)
                del temp_idx[np.argmax(scores[temp_idx])]
                indexs.extend(temp_idx)
        pred = np.delete(pred, indexs, axis=0)
    return pred


def de_dup(pred, args):
    mini_dist = args.mini_dist
    print('de_duplication')
    indexs = []
    areas = np.array(pred)[:, 4]
    pred = np.array(pred)
    pred_final_ = pred[:, 1:4].astype(float)
    for idx, item in enumerate(pred_final_):
        if idx in indexs:
            continue
        d2 = np.linalg.norm(pred_final_ - item, ord=2, axis=1)
        tmp = (d2 < mini_dist)
        tmp[idx] = False
        if len(np.nonzero(tmp)[0]) >= 1:
            temp_idx = np.nonzero(tmp)[0].tolist()
            temp_idx.append(idx)
            for idx_, item in enumerate(temp_idx):
                if item in indexs:
                    del temp_idx[idx_]
            max_idx = np.argmax(areas[temp_idx])
            del temp_idx[max_idx]
            indexs.extend(temp_idx)
    # print('delete indexs:', indexs)
    print('Before De_Dup:', pred.shape[0])
    pred = np.delete(pred, indexs, axis=0)
    print('After De_Dup:', pred.shape[0])
    return pred


