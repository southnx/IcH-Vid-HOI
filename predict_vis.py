import argparse
from collections import defaultdict
import json
import os
import hydra

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch

from pyrutils.metrics import f1_at_k, f1_at_k_single_example
from pyrutils.utils import read_dictionary, cleanup_directory
from vhoi.data_loading import load_testing_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.data_loading import determine_num_classes
from vhoi.losses import extract_value, decide_num_main_losses
from vhoi.models import select_model, query_model
from vhoi.visualisation import plot_segmentation
import platform
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt

SYSTEM = platform.system()

@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    # Data
    model_name, model_input_type = cfg.metadata.model_name, cfg.metadata.input_type
    
    if SYSTEM == 'Linux':
        test_loader, data_info, segmentations, test_ids = load_testing_data(cfg.data, model_name, model_input_type,
                                                                      batch_size=1)
    else:
        test_loader, data_info, segmentations, test_ids = load_testing_data(cfg.data, model_name, model_input_type,
                                                                      batch_size=1)

    if SYSTEM == 'Linux':
        device = 'cuda:0' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'

    data_path = cfg.data.path
    if 'BimanualActions' in data_path:
        label_num = 14
        if SYSTEM == 'Linux':
            init_label = torch.load('/Datasets/BimanualActions/label.pt')
        elif SYSTEM == 'Windows':
            init_label = torch.load('/2G-GCN/data/BimanualActions/label.pt')
    elif 'MPHOI' in data_path:
        label_num = 13
        if SYSTEM == 'Linux':
            init_label = torch.load('/Datasets/MPHOI/label.pt')
        elif SYSTEM == 'Windows':
            init_label = torch.load('/2G-GCN/data/MPHOI/label.pt')
    else:
        label_num = 10
    # sim = torch.nn.functional.cosine_similarity(init_label.unsqueeze(0), init_label.unsqueeze(1), dim=-1)
    sim = None

    dataset_name = cfg.data.name
    if dataset_name == 'mphoi':
        feature_size = 768
        ffn_shape = 2048
    elif dataset_name == 'bimanual':
        feature_size = 256
        ffn_shape = 2048
    else:
        feature_size = 512
        ffn_shape = 2048
    
    human_size, object_size = data_info['input_size']

    
    USE_GRAPH = False
    if dataset_name == 'bimanual':
        language = False
        cross = False
        affordance = False
        isr = True
    elif dataset_name == 'mphoi':
        language = True
        cross = True
        affordance = False
        isr = True
    else:
        language = False
        cross = False
        one = True
        affordance = False
        isr = True

    if isr:
        isr_feature = torch.load('/Datasets/vhoi_model/isr.pt')
    else:
        isr_feature = None

    feature_clip_init = True
    print('feature clip: ', feature_clip_init)
    if feature_clip_init:
        feature = torch.load('/Datasets/vhoi_model/' + dataset_name + '_label.pt')
    else:
        feature = None

    if dataset_name != 'cad120':
        model = query_model(label_num, feature_size, human_size, object_size, device, feature=feature, use_graph=USE_GRAPH, sim=sim,\
                             language=language, ffn_shape=ffn_shape, cross=cross, affordance=affordance, isr=isr, isr_feature=isr_feature).to(device)
    else:
        model = query_model(label_num, feature_size, human_size, object_size, device, feature=feature, use_graph=USE_GRAPH, sim=sim,\
                             language=language, ffn_shape=ffn_shape, cross=cross, one=one, affordance=affordance, isr=isr, isr_feature=isr_feature).to(device)
    test_subject_id = cfg.data.cross_validation_test_subject
    if isinstance(test_subject_id, int):
        test_subject_id = str(test_subject_id)
    print('test subject id: ', test_subject_id)
    if SYSTEM == 'Linux':
        map_location = device
        model.load_state_dict(torch.load('/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id +'_model.pt', map_location=map_location))
    elif SYSTEM == 'Windows':
        model.load_state_dict(torch.load('/2G-GCN/'+ dataset_name +'_model.pt'))

    
    misc_dict = cfg.get('misc', default_value={})
    fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                 dataset_name=dataset_name, **{**misc_dict, **cfg.parameters})
    
    targets = []
    preds = []
    targets_obj = []
    preds_obj = []

    targets_list = []
    preds_list = []
    targets_list_obj = []
    preds_list_obj = []
    downsampling = cfg.data.get('downsampling', default_value=1)
    cnt = 0
    model.eval()

    feature_list = []

    for batch in test_loader:
        input, target = fetch_model_data(batch, device)

        x_human, x_object, x_object_mask, x_steps, x_language = input[0], input[1], input[2], input[7], input[-1]
        if not language:
            x_language = None
        if dataset_name == 'cad120':
            target_obj = target[6]
            target = target[4]
        else:
            target = target[2]

        if not affordance:
            pred, _, feature = model(x_human, x_object, x_object_mask, x_steps, x_language)
        else:
            pred, _, pred_obj = model(x_human, x_object, x_object_mask, x_steps, x_language)
            pred_obj = torch.argmax(pred_obj, dim=-1)
            pred_obj = torch.repeat_interleave(pred_obj, repeats=downsampling, dim=1)

        feature = feature.detach().cpu()
        pred = torch.argmax(pred, dim=-1)
        pred = torch.repeat_interleave(pred, repeats=downsampling, dim=1)
        feature = torch.repeat_interleave(feature, repeats=downsampling, dim=1)


        if pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1]]
            feature = feature[:, :target.shape[1]]
        elif pred.shape[1] < target.shape[1]:
            diff = target.shape[1] - pred.shape[1]
            pad = pred[:, -1:]
            feature_pad = feature[:, -1:]
            pad = torch.repeat_interleave(pad, repeats=diff, dim=1)
            feature_pad = torch.repeat_interleave(feature_pad, repeats=diff, dim=1)
            pred = torch.cat([pred, pad], dim=1)
            feature = torch.cat([feature, feature_pad], dim=1)
        
        if affordance:
            if pred_obj.shape[1] > target_obj.shape[1]:
                pred_obj = pred_obj[:, :target_obj.shape[1]]
            elif pred_obj.shape[1] < target_obj.shape[1]:
                diff = target_obj.shape[1] - pred_obj.shape[1]
                pad = pred_obj[:, -1:]
                pad = torch.repeat_interleave(pad, repeats=diff, dim=1)
                pred_obj = torch.cat([pred_obj, pad], dim=1)
        

        for b_ix, (out, label, feat) in enumerate(zip(pred, target, feature)):
            out = out.cpu().numpy()
            label = label.cpu().numpy()
            targets_list.append(label)
            preds_list.append(out)
            feature_list.append(feat)

            masks = label == -1
            label = label[~masks]
            out = out[~masks]
            assert label.shape[0] == out.shape[0]
            targets += label.tolist()
            preds += out.tolist()
            cnt+=1

        if affordance:
            for b_ix, (out, label) in enumerate(zip(pred_obj, target_obj)):
                out = out.cpu().numpy()
                label = label.cpu().numpy()
                targets_list_obj.append(label)
                preds_list_obj.append(out)
                masks = label == -1
                label = label[~masks]
                out = out[~masks]
                assert label.shape[0] == out.shape[0]
                targets_obj += label.tolist()
                preds_obj += out.tolist()

    def downsample_bad_bimanual_videos(outputs, targets, features, test_ids, video_id_to_video_fps):
        for video_index, video_id in enumerate(test_ids):
            video_fps = video_id_to_video_fps[video_id]
            if video_fps != 15:
                continue
            y_pred, y_true, y_feat = outputs[video_index], targets[video_index], features[video_index]
            y_pred, y_true, y_feat = y_pred[1::2, :], y_true[1::2, :], y_feat[1::2, :]
            outputs[video_index] = y_pred
            targets[video_index] = y_true
            features[video_index] = y_feat
        return outputs, targets, features


    if dataset_name == 'bimanual':
        with open(cfg.data.video_id_to_video_fps, mode='r') as f:
            video_id_to_video_fps = json.load(f)
        preds_list_, targets_list_, feature_list_ = downsample_bad_bimanual_videos(preds_list, targets_list, feature_list, test_ids, video_id_to_video_fps)
        targets, preds = [], []
        preds_list, targets_list, feature_list = [], [], []
        for p, t, f in zip(preds_list_, targets_list_, feature_list_):
            p1, p2 = p[:, 0], p[:, 1]
            t1, t2 = t[:, 0], t[:, 1]
            f1, f2 = f[:, 0], f[:, 1]
            preds_list.append(p1[t1 != -1])
            preds_list.append(p2[t2 != -1])
            targets_list.append(t1[t1 != -1])
            targets_list.append(t2[t2 != -1])
            feature_list.append(f1[t1 != -1])
            feature_list.append(f2[t2 != -1])
            mask = t == -1
            t = t[~mask]
            p = p[~mask]
            targets += t.tolist()
            preds += p.tolist()
    else:
        num_human = targets_list[0].shape[-1]
        if num_human==2:
            targets_list = sum([[item[:, 0], item[:, 1]]for item in targets_list], [])
            preds_list = sum([[item[:, 0], item[:, 1]] for item in preds_list], [])
            feature_list = sum([[item[:, 0], item[:, 1]] for item in feature_list], [])
        else:
            targets_list = sum([[item[:, 0]]for item in targets_list], [])
            preds_list = sum([[item[:, 0]] for item in preds_list], [])

    if feature_clip_init:
        preds_feature = model.label_map(model.label_feature).cpu().detach().numpy()
    else:
        preds_feature = model.label_feature.cpu().detach().numpy()
    print(preds_feature.shape)
    preds_feature = [[item] for item in preds_feature]
    
    preds_frame = [[(-1, -1, -1)] for i in range(label_num)] 

    for v1, (t, p, f) in enumerate(zip(targets_list, preds_list, feature_list)):
        
        for v2, (tt, pp, ff) in enumerate(zip(t, p, f)):
            if tt==-1:
                continue
            preds_feature[pp].append(ff.numpy())
            # print(pp, v1//2)
            preds_frame[pp].append((test_ids[v1//2], v2, v1%2))
            # print(pp, v1//2, test_ids[v1//2], v2, v1%2)
        # print(preds_frame[0])
        # print(preds_frame[1])
        # exit()
    
    # print(preds_frame[1][:300])
    # print(preds_frame[2][:300])

    max_num = 100
    preds_feature = [item[:max_num] if len(item) >max_num else item for item in preds_feature]
    preds_frame = [item[:max_num] if len(item) >max_num else item for item in preds_frame]
    # preds_feature = [preds_feature[1], preds_feature[6]] # 5
    # preds_frame = [preds_frame[1], preds_frame[6]]

    # for idx, (fet, frm) in enumerate(zip(preds_feature, preds_frame)):
    #     rand_idx = np.random.randint(0, fet.shape[0], size=max_num)
    #     fet = fet[rand_idx]
    #     frm = frm[rand_idx]

    with open('/2G-GCN/frame.json', 'w') as f:
        json.dump(preds_frame, f, indent=4)

    X = sum(preds_feature, [])
    X = np.array([i.tolist() for i in X]).astype(np.float)
    print(X.shape)
    print(X.dtype)
    
    torch.save(X, '/2G-GCN/embed.pt')
    exit()

    embed = TSNE(n_components=3,
                   init='random', perplexity=30).fit_transform(X)
    embed = torch.tensor(embed)
    torch.save(embed, '/2G-GCN/embed.pt')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c=y, s=s, cmap='rainbow')
    ax.axis('off')
    ax.grid(None)
    ax.view_init(elev=46, azim=33)

    # plt.scatter(embed[:, 0], embed[:, 1], c=y, s=s)
    # plt.colorbar()

    plt.savefig('/2G-GCN/vis.png')

if __name__ == '__main__':
    main()