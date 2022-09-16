import argparse
import copy

import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2

import datasets.mvtec as mvtec

# device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def load_model():
    """加载模型，并在网络中加入钩子，用于后续钩出指定层的输出"""
    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()

    # set model's intermediate outputs
    layer_outputs = []  # 用于保存与训练网网络特定层的输出，特征提取

    def hook(module, input, output):
        """钩子函数，用于输出网络中特定层的信息，这里输出指定层的输出"""
        layer_outputs.append(output)

    # 输出网络的特定层的输出
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    return model, layer_outputs


def make_dataload(class_name):
    """返回dataload"""
    train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=False)
    test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=False)

    return train_dataloader, test_dataloader


def extract_train_feature(args, train_dataloader, model, train_feature_outputs, class_name, layer_outputs):
    """提取训练数据的特征，并保存"""
    train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)

    if not os.path.exists(train_feature_filepath):
        # 遍历训练集图片（每次遍历batch张），钩出网络特定层的输出，保存在字典中
        for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # 模型forward才能调用钩子方法
            with torch.no_grad():
                pred = model(x.to(device))
            # 将网络中间（指定）层的输出保存到字典中
            for k, v in zip(train_feature_outputs.keys(), layer_outputs):
                train_feature_outputs[k].append(v)
            # 清空，用于下一次保存
            layer_outputs = []
        # 对于每个指定层保存的输出，将其多次的输出在axis=0上合并
        for k, v in train_feature_outputs.items():
            train_feature_outputs[k] = torch.cat(v, 0)
        # 保存提取的特征
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_feature_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_feature_outputs = pickle.load(f)

    return train_feature_outputs


def extract_test_feature(test_dataloader, model, test_feature_outputs, class_name, layer_outputs):
    """提取测试图片特征"""
    gt_list = []
    gt_mask_list = []
    test_imgs = []
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # 模型前向传播才能调用钩子方法
        with torch.no_grad():
            pred = model(x.to(device))
        # 用钩子函数，钩出特定层的输出，并保存在字典中
        for k, v in zip(test_feature_outputs.keys(), layer_outputs):
            test_feature_outputs[k].append(v)
        # 清空列表，进行下一次迭代（测试图片）保存钩出的特征
        layer_outputs = []
    # 对上述多次迭代后钩出的特征，在axis=0方向上合并，即迭代次数方向上
    for k, v in test_feature_outputs.items():
        test_feature_outputs[k] = torch.cat(v, 0)


def main():
    args = parse_args()

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)
    model, hook_layer_outputs = load_model()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES[:1]:

        train_dataloader, test_dataloader = make_dataload(class_name)
        # 从预训练模型中提取的特征
        train_feature_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_feature_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        # --------------------------------------------------------------------------
        # 提取训练数据的特征

        train_feature_outputs = copy.deepcopy(train_feature_outputs)  # 深拷贝，防止原始列表被修改
        hook_layer_outputs = copy.deepcopy(hook_layer_outputs)
        train_feature_outputs = extract_train_feature(args, train_dataloader, model, train_feature_outputs, class_name,
                                                      hook_layer_outputs)

        # --------------------------------------------------------------------------
        # 提取测试集中的特征
        test_feature_outputs = copy.deepcopy(test_feature_outputs)  # 深拷贝，防止原始列表被修改
        hook_layer_outputs = copy.deepcopy(hook_layer_outputs)
        extract_test_feature(test_dataloader, model, test_feature_outputs, class_name, hook_layer_outputs)

        # calculate distance matrix
        dist_matrix = calc_dist_matrix(torch.flatten(test_feature_outputs['avgpool'], 1),
                                       torch.flatten(train_feature_outputs['avgpool'], 1))

        # select K nearest neighbor and take average
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()

        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        score_map_list = []
        for t_idx in tqdm(range(test_feature_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = train_feature_outputs[layer_name][topk_indexes[t_idx]]
                test_feat_map = test_feature_outputs[layer_name][t_idx:t_idx + 1]
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                # calculate distance matrix
                dist_matrix_list = []

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = torch.cat(dist_matrix_list, 0)

                # k nearest features from the gallery (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                          mode='bilinear', align_corners=False)
                score_maps.append(score_map)

            # average distance between the features
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(score_map)

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=5)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):
    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
