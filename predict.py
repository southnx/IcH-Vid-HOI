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
from vhoi.data_loading import load_testing_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.models import query_model
import platform

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
    sim = None
    label_num = cfg.data.label_num

    dataset_name = cfg.data.name

    feature_size = cfg.data.feature_size
    
    human_size, object_size = data_info['input_size']

    
    USE_GRAPH = False

    language = cfg.hyperparam.language
    affordance = cfg.hyperparam.affordance

    if cfg.hyperparam.isr:
        isr_feature = torch.load('/mnt/data0/Datasets/vhoi_model/isr.pt')
    else:
        isr_feature = None

    feature_clip_init = True
    print('feature clip: ', feature_clip_init)
    if feature_clip_init:
        feature = torch.load('/mnt/data0/Datasets/vhoi_model/' + dataset_name + '_label.pt')
    else:
        feature = None
    
    model = query_model(label_num, feature_size, human_size, object_size, device, feature = feature, use_graph=USE_GRAPH, sim=sim, isr_feature = isr_feature, **cfg.hyperparam).to(device)
    
    test_subject_id = cfg.data.cross_validation_test_subject
    if isinstance(test_subject_id, int):
        test_subject_id = str(test_subject_id)
    print('test subject id: ', test_subject_id)
    if SYSTEM == 'Linux':
        map_location = device
        model.load_state_dict(torch.load('/mnt/data0/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id +'_model.pt', map_location=map_location))
    elif SYSTEM == 'Windows':
        model.load_state_dict(torch.load('d:/intern/2G-GCN/'+ dataset_name +'_model.pt'))

    
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
            pred, _ = model(x_human, x_object, x_object_mask, x_steps, x_language)
        else:
            pred, _, pred_obj = model(x_human, x_object, x_object_mask, x_steps, x_language)
            pred_obj = torch.argmax(pred_obj, dim=-1)
            pred_obj = torch.repeat_interleave(pred_obj, repeats=downsampling, dim=1)
        pred = torch.argmax(pred, dim=-1)
        pred = torch.repeat_interleave(pred, repeats=downsampling, dim=1)


        if pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1]]
        elif pred.shape[1] < target.shape[1]:
            diff = target.shape[1] - pred.shape[1]
            pad = pred[:, -1:]
            pad = torch.repeat_interleave(pad, repeats=diff, dim=1)
            pred = torch.cat([pred, pad], dim=1)
        
        if affordance:
            if pred_obj.shape[1] > target_obj.shape[1]:
                pred_obj = pred_obj[:, :target_obj.shape[1]]
            elif pred_obj.shape[1] < target_obj.shape[1]:
                diff = target_obj.shape[1] - pred_obj.shape[1]
                pad = pred_obj[:, -1:]
                pad = torch.repeat_interleave(pad, repeats=diff, dim=1)
                pred_obj = torch.cat([pred_obj, pad], dim=1)
        

        for b_ix, (out, label) in enumerate(zip(pred, target)):
            out = out.cpu().numpy()
            label = label.cpu().numpy()
            targets_list.append(label)
            preds_list.append(out)

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

    def downsample_bad_bimanual_videos(outputs, targets, test_ids, video_id_to_video_fps):
        for video_index, video_id in enumerate(test_ids):
            video_fps = video_id_to_video_fps[video_id]
            if video_fps != 15:
                continue
            y_pred, y_true = outputs[video_index], targets[video_index]
            y_pred, y_true = y_pred[1::2, :], y_true[1::2, :]
            outputs[video_index] = y_pred
            targets[video_index] = y_true
        return outputs, targets


    if dataset_name == 'bimanual':
        with open(cfg.data.video_id_to_video_fps, mode='r') as f:
            video_id_to_video_fps = json.load(f)
        preds_list_, targets_list_ = downsample_bad_bimanual_videos(preds_list, targets_list, test_ids, video_id_to_video_fps)
        targets, preds = [], []
        preds_list, targets_list = [], []
        for p, t in zip(preds_list_, targets_list_):
            p1, p2 = p[:, 0], p[:, 1]
            t1, t2 = t[:, 0], t[:, 1]
            preds_list.append(p1[t1 != -1])
            preds_list.append(p2[t2 != -1])
            targets_list.append(t1[t1 != -1])
            targets_list.append(t2[t2 != -1])
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
        else:
            targets_list = sum([[item[:, 0]]for item in targets_list], [])
            preds_list = sum([[item[:, 0]] for item in preds_list], [])

    precisions, recalls, fss, _ = precision_recall_fscore_support(targets, preds, labels=range(label_num))
    
    print(precisions, sum(precisions)/len(precisions))
    print(recalls, sum(recalls)/len(recalls))
    print(fss, sum(fss)/len(fss))

    
    for k in [0.1, 0.25, 0.5]:
        f1 = f1_at_k(targets_list, preds_list, label_num, overlap=k, ignore_value=-1)
        print(f'f1 at {k}: {f1}')

    if affordance:
        precisions, recalls, fss, _ = precision_recall_fscore_support(targets_obj, preds_obj, labels=range(12)) 
        print(precisions, sum(precisions)/len(precisions))
        print(recalls, sum(recalls)/len(recalls))
        print(fss, sum(fss)/len(fss))
        targets_list_obj = np.array([item.tolist() for item in targets_list_obj]).swapaxes(1, 2)
        preds_list_obj = np.array([item.tolist() for item in preds_list_obj]).swapaxes(1, 2)
        b, n, f = preds_list_obj.shape
        preds_list_obj = preds_list_obj.reshape(b*n, f)
        targets_list_obj = targets_list_obj.reshape(b*n, f)
        for k in [0.1, 0.25, 0.5]:
            f1 = f1_at_k(targets_list_obj, preds_list_obj, 12, overlap=k, ignore_value=-1)
            print(f'f1 at {k}: {f1}')



if __name__ == '__main__':
    main()
