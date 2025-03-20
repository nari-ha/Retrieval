import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import matplotlib.pyplot as plt
import numpy as np

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    loss_history = []
    accuracy_history = []
    map_history = []
    r1_history = []
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
                
        loss_history.append(loss_meter.avg)
        accuracy_history.append(acc_meter.avg)
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            save_model(cfg, model, epoch)

        if epoch % eval_period == 0:
            mAP, r1 = evaluate_model(cfg, model, val_loader, evaluator, device, epoch, logger, text_features)  # 모델 평가
            map_history.append(round(mAP * 100, 1))
            r1_history.append(round(r1 * 100, 1))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    
    # learning curve graph 저장
    loss_history = np.array(loss_history)
    accuracy_history = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in accuracy_history])
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='blue')
    ax1.plot(range(1, epochs + 1), loss_history, label="Loss", color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='orange')
    ax2.plot(range(1, epochs + 1), accuracy_history, label="Accuracy", color='orange', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle("Stage2 Loss&Accuracy")
    fig.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "stage2.png"))
    
    # evaluation graph 저장
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel("eval steps")
    ax1.set_ylabel("mAP", color='red')
    ax1.plot(range(1, len(map_history) + 1), map_history, label="mAP", linewidth=2, marker='o', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    for i, v in enumerate(map_history):
        ax1.annotate(v, (i + 1, v), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel("R1", color='green')
    ax2.plot(range(1, len(r1_history) + 1), r1_history, label="R1", linewidth=2, marker='o', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    for i, v in enumerate(r1_history):
        ax2.annotate(v, (i + 1, v), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='green')


    fig.suptitle("Stage2 Evaluation")
    fig.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "eval.png"))

def save_model(cfg, model, epoch):
    if cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0:
        return  # 분산 훈련 시 rank 0만 저장
    
    torch.save(model.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth"))

def evaluate_model(cfg, model, val_loader, evaluator, device, epoch, logger, text_features):
    # 모델 평가하기
    model.eval()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid, text_features))
    
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info(f"Validation Results - Epoch: {epoch}")
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    
    torch.cuda.empty_cache()
    return mAP, cmc[0]

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]