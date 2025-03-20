import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def draw_curve(cfg, loss_history, accuracy_history, map_history, r1_history):
        # learning curve graph 저장
    loss_history = np.array(loss_history)
    accuracy_history = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in accuracy_history])
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='blue')
    ax1.plot(range(1, cfg.SOLVER.STAGE2.MAX_EPOCHS + 1), loss_history, label="Loss", color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='orange')
    ax2.plot(range(1, cfg.SOLVER.STAGE2.MAX_EPOCHS + 1), accuracy_history, label="Accuracy", color='orange', linewidth=2)
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