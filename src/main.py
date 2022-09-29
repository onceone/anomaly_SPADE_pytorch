import argparse
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from tqdm import tqdm

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

    global layer_outputs
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


def extract_train_feature(args, train_dataloader, model, class_name):
    """提取训练数据的特征，并保存，字典做参数，内容别修改，不用返回 , layer_outputs"""
    global layer_outputs
    train_feature_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

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
        # 对于每个指定层保存的输出，将其多次的输出在axis=0上合并，
        # n * [1,channel,height,width] -> [n,channel,height,width]
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


def extract_test_feature(test_dataloader, model, class_name):
    """提取测试图片特征,"""
    global layer_outputs
    test_feature_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

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

    return test_feature_outputs, test_imgs, gt_list, gt_mask_list


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
        # --------------------------------------------------------------------------------------------------
        # 说明：如下注释中的“每张”概念，实际是batch张图片，即，以一个batch为单位！！！！！！
        # 提取训练数据的特征,[n,channel,height,width] , n为迭代次数
        train_feature_outputs = extract_train_feature(args, train_dataloader, model, class_name)
        # --------------------------------------------------
        # 提取测试集中的特征
        test_feature_outputs, test_img, gt_list, gt_mask_list = extract_test_feature(test_dataloader, model, class_name)
        # ---------------------------------------------------------------------------------------------------
        """
        计算测试图片和训练图片之间的距离(用最后一个卷积后的avgpool代表图片---------->>>img-level)
        [n,c,h,w] -> [n,c*h*w] ； [m,c,h,w] -> [m,c*h*w];n、m分别为图片“个数” ，每“个”图片为batch张图片
        计算 [n,G]&[m,G] 之间的距离 ； [83,2048]&[209,2048]-> [83,209]
        test_feature是x在前，train_feature是y在后，返回的是[83,209],即83张测试图片和209张训练图片的距离
        """
        x = torch.flatten(test_feature_outputs['avgpool'], 1)
        y = torch.flatten(train_feature_outputs['avgpool'], 1)
        dist_matrix = calc_dist_matrix(x, y)

        # KNN,在上一步计算的dist_matrix([83,209])中，选取5个最小的值，即对每“个”测试选取5个最近的训练图片
        # topk_values:=> [83,5] ； topk_indexes: [83,5]距离最近的5个训练图片的索引
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
        # [83,5] -> [83],5个值取平均 ,即每个测试图片与（5个）训练图片的距离，把距离当做得分
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()

        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))
        # ----------------------------------------------------------------------------------------------------
        score_map_list = []  # 保存所有处理后的测试图片
        for t_idx in tqdm(range(test_feature_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
            score_maps_all_layers = []  # 保存 所有层(layer1,layer2,layer3) 中测试集与对应的层中的训练集特征图最近个score_map
            for layer_name in ['layer1', 'layer2', 'layer3']:  # 图像通道数分别为：256、512、1024

                # -----------------------------------------------------------------------------------------------
                # 遍历每“张”（batch张），“特定层”中训练集的特征图
                """ construct a gallery of features at all pixel locations of the K nearest neighbors
                train_feature_outputs['layer1']:[209,256,56,56] ,topk_indexs:[83,5],topk_indexs[0]: (5,)
                依次找每张测试图片距离最近的的5张(训练图片在网络中指定层的)特征图,eg: top_feat_map:[5,256,56,56],5张layer1层的特征"""
                topk_feat_map = train_feature_outputs[layer_name][topk_indexes[t_idx]]
                # [83,256,56,56]->[1,256,56,56],选择指定索引的测试图片指定层的特征图，为保持shape,使用索引切片
                test_feat_map = test_feature_outputs[layer_name][t_idx:t_idx + 1]
                # [5,256,56,56],->[5,56,56,256] ->[15680,256]->[15680,256,1] -> [15680,256,1,1]
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                # -----------------------------------------------------------------------------------------------
                # 计算每"张"(batch张）测试特征图和最近的5“张”(batch张)训练特征图的距离
                # test,test_feat_map: [1,256,56,56]  ,逐层、逐个遍历
                # train,feat_gallery: [156800,256,1,1]
                dist_matrix_list = []
                for d_idx in range(feat_gallery.shape[0] // 100):
                    # feat_gallery:[15680,256,1,1];
                    train_feat = feat_gallery[d_idx * 100:d_idx * 100 + 100]
                    # train_feat: [100,256,1,1]  test_feat_map:[1,256,56,56]  -> [100,256,56]
                    # train_feat和test_feat_map的batch要相等，或其中一个为1
                    """这里实际是对最后一维进行距离计算，可以理解为特征图对应行计算距离"""
                    dist = torch.pairwise_distance(train_feat, test_feat_map)
                    dist_matrix_list.append(dist)
                # 156*[100,256,56]-> [15600,256,56]
                dist_matrix = torch.cat(dist_matrix_list, 0)
                # ------------------------------------------------------------------------------------------------
                # k nearest features from the gallery (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]  # 最小值 [15600,256,56] -> [256,56]
                # [256,56] -> [1,1,256,56] -> [1,1,224,224]
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224, mode='bilinear',
                                          align_corners=False)
                score_maps_all_layers.append(score_map)
                # ------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------
            # 对三个层中的特征图求平均, 3*[1,1,224,224] -> [3,1,224,224] -> [1,224,224]
            score_map_layer_mean = torch.mean(torch.cat(score_maps_all_layers, 0), dim=0)

            # apply gaussian smoothing on the score map [1,224,224] -> [224,224]
            score_map_layer_smooth = gaussian_filter(score_map_layer_mean.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(score_map_layer_smooth)
        # ----------------------------------------------------------------------------------------------------------
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        # ----------------------------------------------------------------------------------------------------------
        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # ----------------------------------------------------------------------------------------------------------
        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # ----------------------------------------------------------------------------------------------------------
        # visualize localization result
        visualize_loc_result(test_img, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=5)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor
        x: [83,2048] ，2048为 channel * height * width
        y: [209,2048]
        return: dist_matrix [83,209]
    """

    n = x.size(0)  # 83
    m = y.size(0)  # 209
    d = x.size(1)  # 2048
    x = x.unsqueeze(1).expand(n, m, d)  # [83,2048] -> [83,1,2048] -> [83,209,2048]
    y = y.unsqueeze(0).expand(n, m, d)  # [209,2048] -> [1,209,2048] -> [83,209,2048]
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))  # [83,209,2048] -> [83,209]
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
