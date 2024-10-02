import hydra
from omegaconf import DictConfig
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from vhoi.data_loading import load_training_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.models import query_model
import json
from sklearn.metrics import precision_recall_fscore_support
import os
from pyrutils.metrics import f1_at_k_my
import platform
import random
import numpy as np

SYSTEM = platform.system()
PRETRAINED = False
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig):
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    # Data
    model_name, model_input_type = cfg.metadata.model_name, cfg.metadata.input_type
    batch_size, val_fraction = cfg.optimization.batch_size, cfg.optimization.val_fraction
    misc_dict = cfg.get('misc', default_value={})
    sigma = misc_dict.get('segmentation_loss', {}).get('sigma', 0.0)

    dataset_name = cfg.data.name
    print("dataset: ", dataset_name)
    Base = 0
    if dataset_name != 'mphoi':
        Base = 0

    if local_rank != -1:
        local_rank += Base
        print('local rank:', local_rank)
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else:
        device = 'cuda:3' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg.resources.num_threads)
    
    
    data_path = cfg.data.path
    label_num = cfg.data.label_num
    
    sim = None

    feature_size = cfg.data.feature_size
    batch_size = cfg.data.batch_size

    train_loader, val_loader, data_info, scalers, sampler = load_training_data(cfg.data, model_name, model_input_type,
                                                                      batch_size=batch_size,
                                                                      val_fraction=val_fraction,
                                                                      seed=seed, debug=False, sigma=sigma)
    
    human_size, object_size = data_info['input_size']
    epoch = cfg.data.epoch

    USE_GRAPH = False
    language = cfg.hyperparam.language
    affordance = cfg.hyperparam.affordance
    TEMPORAL_BETA = cfg.data.temporal_beta

    if cfg.hyperparam.isr:
        isr_feature = torch.load('/Datasets/vhoi_model/isr.pt')
    else:
        isr_feature = None
    print('isr: ', cfg.hyperparam.isr)
    print('temporal beta: ', TEMPORAL_BETA)

    feature_clip_init = True
    print('feature clip: ', feature_clip_init)
    if feature_clip_init:
        feature = torch.load('/Datasets/vhoi_model/' + dataset_name + '_label.pt')
    else:
        feature = None
    
    model = query_model(label_num, feature_size, human_size, object_size, device, feature = feature, use_graph=USE_GRAPH, sim=sim, isr_feature = isr_feature, **cfg.hyperparam).to(device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank)
    
    test_subject_id = cfg.data.cross_validation_test_subject
    if isinstance(test_subject_id, int):
        test_subject_id = str(test_subject_id)
    print('test subject id :', test_subject_id)
    if PRETRAINED:
        if local_rank == -1:
            model.load_state_dict(torch.load('/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id +'_model.pt'))
        else:
            map_location = torch.device(f'cuda:{local_rank}')
            model.module.load_state_dict(torch.load('/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id +'_model.pt', map_location=map_location))
        print('Load from saved model')

    data_path = cfg.data.path
    if 'BimanualActions' in data_path:
        init_learning_rate = 2e-4 # no clip: 2e-4
        fin_learning_rate = 2e-4
        step = 200
    elif 'MPHOI' in data_path:
        init_learning_rate = 1e-4 # 1 or 2 !!!!!!!!!!!!!!!!!!!!
        fin_learning_rate = 1e-4
        step = 100
    else:
        init_learning_rate = 2e-4
        fin_learning_rate = 2e-4
        step = 50
    warm_step = 0
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=init_learning_rate)
    
    scheduler = StepLR(optimizer, step, gamma=0.5)

    thres = 0
    if dataset_name == 'mphoi':
        thres = 0


    fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                 dataset_name=dataset_name, **{**misc_dict, **cfg.parameters})
    
    def loss_new(pred, target, temporal_diff, seg, pred_obj=None, target_obj=None, gamma=2, eps=1e-8):
        temporal = torch.multiply(temporal_diff, 1 - seg).pow(2)/2 + torch.multiply(torch.nn.functional.relu(10 - temporal_diff), seg).pow(2)/2
        temporal *= TEMPORAL_BETA # 2e-4
        obj_loss = None
        if pred_obj is not None:
            obj_mask = target_obj == -1
            target_obj[obj_mask] = 0
            obj_one_hot = torch.nn.functional.one_hot(target_obj, num_classes=12).to(torch.float)
            p = pred_obj
            q = 1 - p
            p = torch.max(p, torch.tensor(eps))
            q = torch.max(q, torch.tensor(eps))
            pos_loss = -(q ** gamma) * torch.log(p)
            neg_loss = -(p ** gamma) * torch.log(q)
            obj_loss = pos_loss * obj_one_hot + neg_loss * (1 - obj_one_hot)
            obj_loss[obj_mask] = 0
            obj_loss = obj_loss.sum(-1).mean(dim=-1).unsqueeze(-1)

        target_mask = target == -1
        target[target_mask] = 0
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=label_num)
        p = pred
        q = 1 - p
        p = torch.max(p, torch.tensor(eps))
        q = torch.max(q, torch.tensor(eps))
        pos_loss = -(q ** gamma) * torch.log(p)
        neg_loss = -(p ** gamma) * torch.log(q)
        label = target_one_hot.to(torch.float64)
        loss = pos_loss * label + neg_loss * (1 - label)
        loss = loss.sum(-1)
        loss += temporal
        if obj_loss is not None:
            loss += obj_loss
        loss[target_mask] = 0
        return loss.mean()


    best = 0.
    val_cnt = 0
    torch.autograd.set_detect_anomaly(True)
    for e in range(1, epoch+1):
        
        losses = torch.tensor([0.], dtype=torch.float32, device=device)
        model.train()
        if local_rank != -1:
            # if e % 50 == 0:
            #     model.module.update_range()
            if local_rank == Base:
                print(f'[Epoch {e}/{epoch}]')
            train_loader.sampler.set_epoch(e) #shuffle
            
            with model.join():
                for batch in train_loader:
                    input, target = fetch_model_data(batch, device)
                    x_human, x_object, x_object_mask, x_steps = input[0], input[1], input[2], input[7]
                    if language:
                        x_language = input[-1]
                    else:
                        x_language = None
                    seg = target[0]
                    if dataset_name == 'cad120':
                        target_obj = target[6]
                        target = target[4]
                    else:
                        target_obj = None
                        target = target[2]

                    pred_obj = None
                    if not affordance:
                        pred, temporal_diff = model(x_human, x_object, x_object_mask, x_steps, x_language)
                    else:
                        pred, temporal_diff, pred_obj = model(x_human, x_object, x_object_mask, x_steps, x_language)
                    loss = loss_new(pred, target, temporal_diff, seg, pred_obj, target_obj)

                    label = model.module.label_feature
                    label1, label2 = label.unsqueeze(0), label.unsqueeze(1)
                    label_norm = 20 - torch.norm(label1 - label2, dim=-1)
                    label_norm = label_norm.reshape(-1)[:-1].reshape(label_num - 1, label_num + 1)[:, 1:]
                    dis = torch.nn.functional.relu(label_norm, inplace=False).pow(2)/2
                    loss = loss + dis.sum()/(2 * (label_num - 1) * label_num)

                    losses += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            torch.cuda.synchronize(device=local_rank)
        else:
            print(f'[Epoch {e}/{epoch}]')
            for batch in train_loader:
                input, target = fetch_model_data(batch, device)
                x_human, x_object, x_object_mask, x_steps = \
                                    input[0], input[1], input[2], input[7]
                if language:
                    x_language = input[-1]
                else:
                    x_language = None
                seg = target[0]
                if dataset_name == 'cad120':
                    target_obj = target[6]
                    target = target[4]
                else:
                    target_obj = None
                    target = target[2]
                
                pred_obj = None
                if not affordance:
                    pred, temporal_diff = model(x_human, x_object, x_object_mask, x_steps, x_language)
                else:
                    pred, temporal_diff, pred_obj = model(x_human, x_object, x_object_mask, x_steps, x_language)
                loss = loss_new(pred, target, temporal_diff, seg, pred_obj, target_obj)

                label = model.label_feature
                label1, label2 = label.unsqueeze(0), label.unsqueeze(1)
                label_norm = 20 - torch.norm(label1 - label2, dim=-1)
                label_norm = label_norm.reshape(-1)[:-1].reshape(label_num - 1, label_num + 1)[:, 1:]
                dis = torch.nn.functional.relu(label_norm, inplace=False).pow(2)/2
                loss = loss + dis.sum()/(2 * (label_num - 1) * label_num)
                
                losses += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                


        loss = losses / len(train_loader)
        if local_rank!=-1:
            torch.distributed.all_reduce(loss)
            loss /= torch.distributed.get_world_size()
            if local_rank==Base:
                print(loss.item())
        else:
            print(loss.item())
        
        if e <= warm_step:
            dif = (fin_learning_rate - init_learning_rate)/warm_step
            lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr + dif
        else:
            scheduler.step()

        if e % 50 != 0:
            continue
    
        if local_rank == -1 or local_rank == Base:
            print('[Evaluate]')

        targets = []
        preds = []
        targets_obj = []
        preds_obj = []

        targets_list = []
        preds_list = []
        targets_list_obj = []
        preds_list_obj = []

        model.eval()
        if local_rank == -1:
            for batch in val_loader:
                input, target = fetch_model_data(batch, device)

                x_human, x_object, x_object_mask, x_steps, x_language = input[0], input[1], input[2], input[7], input[-1]
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
                pred = torch.argmax(pred, dim=-1)
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
                    
        else:
            with model.join():
                for batch in val_loader:
                    input, target = fetch_model_data(batch, device)
                    x_human, x_object, x_object_mask, x_steps, x_language = input[0], input[1], input[2], input[7], input[-1]
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
                    pred = torch.argmax(pred, dim=-1)
                    for b_ix, (out, label) in enumerate(zip(pred, target)):
                        out = out.cpu().numpy()
                        label = label.cpu().numpy()
                        targets_list.append(label)
                        preds_list.append(out)
                        targets += label.tolist()
                        preds += out.tolist()
                    if affordance:
                        for b_ix, (out, label) in enumerate(zip(pred_obj, target_obj)):
                            out = out.cpu().numpy()
                            label = label.cpu().numpy()
                            targets_list_obj.append(label)
                            preds_list_obj.append(out)
                            targets_obj += label.tolist()
                            preds_obj += out.tolist()
                        
            torch.cuda.synchronize(device=local_rank)

        
        if local_rank != -1:
            preds = torch.tensor(preds, device=device)
            targets = torch.tensor(targets, device=device)
            world_size = torch.distributed.get_world_size()
            all_preds = [torch.zeros_like(preds) for _ in range(world_size)]
            all_targets = [torch.zeros_like(targets) for _ in range(world_size)]
            torch.distributed.all_gather(all_preds, preds)
            torch.distributed.all_gather(all_targets, targets)
            preds = torch.tensor([item.tolist() for item in all_preds])
            targets = torch.tensor([item.tolist() for item in all_targets])
            mask = targets == -1
            preds = preds[~mask].tolist()
            targets = targets[~mask].tolist()
        
        temp = 0.
        if local_rank != -1:
            precisions, recalls, fss, _ = precision_recall_fscore_support(targets, preds, labels=range(label_num))
            if local_rank == Base:
                print(precisions, sum(precisions)/len(precisions))
                print(recalls, sum(recalls)/len(recalls))
                print(fss, sum(fss)/len(fss))
            # temp = sum(fss)/len(fss)
            num_human = targets_list[0].shape[-1]
            if num_human==2:
                targets_list = sum([[item[:, 0], item[:, 1]]for item in targets_list], [])
                preds_list = sum([[item[:, 0], item[:, 1]] for item in preds_list], [])
            else:
                targets_list = sum([[item[:, 0]]for item in targets_list], [])
                preds_list = sum([[item[:, 0]] for item in preds_list], [])
            for k in [0.1, 0.25, 0.5]:
                f1, num_examples = f1_at_k_my(targets_list, preds_list, label_num, overlap=k, ignore_value=-1)
                f1 = torch.tensor(f1, dtype=torch.float32, device=device)
                num_examples = torch.tensor(num_examples, dtype=torch.float32, device=device)
                torch.distributed.all_reduce(f1)
                torch.distributed.all_reduce(num_examples)
                f1 = f1.item()
                num_examples = num_examples.item()
                f1 = f1/num_examples
                temp += f1
                if local_rank==Base:
                    print(f'f1 at {k}: {f1}')
        else:
            precisions, recalls, fss, _ = precision_recall_fscore_support(targets, preds, labels=range(label_num))
            print(precisions, sum(precisions)/len(precisions))
            print(recalls, sum(recalls)/len(recalls))
            print(fss, sum(fss)/len(fss))
            temp = sum(fss)/len(fss)
            if affordance:
                precisions, recalls, fss, _ = precision_recall_fscore_support(targets_obj, preds_obj, labels=range(12))
                print(precisions, sum(precisions)/len(precisions))
                print(recalls, sum(recalls)/len(recalls))
                print(fss, sum(fss)/len(fss))
            num_human = targets_list[0].shape[-1]
            if num_human==2:
                targets_list = sum([[item[:, 0], item[:, 1]]for item in targets_list], [])
                preds_list = sum([[item[:, 0], item[:, 1]] for item in preds_list], [])
            else:
                targets_list = sum([[item[:, 0]]for item in targets_list], [])
                preds_list = sum([[item[:, 0]] for item in preds_list], [])
            for k in [0.1, 0.25, 0.5]:
                f1, num_examples = f1_at_k_my(targets_list, preds_list, label_num, overlap=k, ignore_value=-1)
                f1 = f1/num_examples
                temp += f1
                print(f'f1 at {k}: {f1}')
            if affordance:
                preds_list_obj = np.array([item.tolist() for item in preds_list_obj]).swapaxes(1, 2)
                targets_list_obj = np.array([item.tolist() for item in targets_list_obj]).swapaxes(1, 2)
                b, n, f = preds_list_obj.shape
                preds_list_obj = preds_list_obj.reshape(b*n, f)
                targets_list_obj = targets_list_obj.reshape(b*n, f)
                for k in [0.1, 0.25, 0.5]:
                    f1, num_examples = f1_at_k_my(targets_list_obj, preds_list_obj, 12, overlap=k, ignore_value=-1)
                    f1 = f1/num_examples
                    temp += f1
                    print(f'f1 at {k}: {f1}')

            

        if e < thres:
            continue
        
        temp /= 3.
        if temp <= best:
            val_cnt +=1
        else:
            best = temp
            val_cnt = 0

        if local_rank == -1 or local_rank == Base:
            print('val count: ', val_cnt)

        if val_cnt == 2:
            val_cnt = 0
            if local_rank == -1 or local_rank == Base:
                print('Load old model')
            if local_rank == -1:
                model.load_state_dict(torch.load('/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id  +'_model.pt'))
            else:
                map_location = torch.device(f'cuda:{local_rank}')
                model.module.load_state_dict(torch.load('/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id  +'_model.pt', map_location=map_location))
                torch.distributed.barrier()
            continue

        if val_cnt != 0:
            continue

        if local_rank == -1 or local_rank == Base:
            print('Save new model')
        if local_rank == -1:
            torch.save(model.state_dict(), '/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id  +'_model.pt')
        else:
            if local_rank == Base:
                torch.save(model.module.state_dict(), '/Datasets/vhoi_model/'+ dataset_name + '_' + test_subject_id  +'_model.pt')
            torch.distributed.barrier()



if __name__ == '__main__':
    main()