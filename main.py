import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime, timedelta
import time
import shutil
import importlib
import argparse
import random
from dataloader import MiniImageNet, CifarFs, Cub200, TieredImageNet, MetaDataset
from dgcm import DGCM

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloader(config, mode):

    if mode == 'train':
        loader_config = config['train_config']
    else:  # 'val' or 'test'
        loader_config = config['eval_config']

    dataset_args = {
        'mode': mode,
        'root': config['dataset']['root'],
        'num_ways': loader_config['num_ways'],
        'num_shots': loader_config['num_shots'],
    }

    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImageNet(**dataset_args)
    # elif config['dataset_name'] == 'cifar-fs':
    #     dataset = CifarFs(**dataset_args)
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImageNet(**dataset_args)
    elif config['dataset_name'] == 'cub200':

        if mode != 'train':
            dataset_args['num_query'] = config['eval_config'].get('num_query', 15)
        dataset = Cub200(**dataset_args)
    elif config['dataset_name'] == 'cifar-fs':
        dataset = CifarFs(**dataset_args)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset_name']}")


    dataloader = DataLoader(dataset, batch_size=loader_config['batch_size'], shuffle=(mode == 'train'), num_workers=4,
                            pin_memory=True)
    return dataloader


def run_evaluation(config, model, loader, device):

    model.eval()
    accs = []


    eval_iterations = config['eval_config']['iteration']
    eval_batch_size = config['eval_config']['batch_size']
    num_batches_to_run = eval_iterations // eval_batch_size

    print(f"Running validation for {eval_iterations} tasks ({num_batches_to_run} batches of size {eval_batch_size})...")


    val_iter = iter(loader)
    for _ in range(num_batches_to_run):
        try:
            batch = next(val_iter)
        except StopIteration:

            val_iter = iter(loader)
            batch = next(val_iter)


        support_x, support_y, query_x, query_y, class_names = batch
        support_x, support_y, query_x, query_y = [x.to(device) for x in [support_x, support_y, query_x, query_y]]

        with torch.no_grad():
            if config.get('model_name') == 'dgcm':
                _, acc = model(support_x, support_y, query_x, query_y, class_names)
            else:
                _, acc = model(support_x, support_y, query_x, query_y)
        accs.append(acc.item())

    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    ci95 = 1.96 * std_acc / np.sqrt(len(accs))

    return avg_acc, ci95


def main(args):


    config_module = importlib.import_module(f"config.{args.config}")
    config = config_module.config

    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Using fixed random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    backbone_name_safe = config.get('backbone', 'default_backbone').replace('/', '-')

    model_name = config.get('model_name', 'dgcm')
    experiment_name = f"{model_name}_{config['train_config']['num_ways']}way_{config['train_config']['num_shots']}shot_{backbone_name_safe}_{config['dataset_name']}"

    checkpoint_dir = os.path.join('checkpoints', experiment_name)


    print("Instantiating model...")


    model_name = config.get('model_name', 'dgcm')

    if model_name == 'dgcm':
        print("Model: DGCM")
        model = DGCM(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)

    if args.mode == 'train':

        log_dir = os.path.join('logs', experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy(f"config/{args.config}.py", os.path.join(log_dir, 'config.py'))
        log_file = os.path.join(log_dir, 'train_log.txt')
        with open(log_file, 'a') as f:
            f.write(f"Experiment started at {datetime.now()}\n=========================================\n")

        print("Getting dataloaders...")
        train_loader = get_dataloader(config, 'train')
        val_loader = get_dataloader(config, 'val')

        if config.get('model_name') == 'dgcm':
            print("--- Optimizing DGCM's trainable parameters only ---")
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            trainable_params = model.parameters()
        optimizer = optim.Adam(trainable_params, lr=config['train_config']['lr'],
                               weight_decay=config['train_config']['weight_decay'])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train_config']['dec_lr'],
                                                 gamma=config['train_config']['lr_adj_base'])

        best_val_acc = 0.0
        train_iter = iter(train_loader)

        print("Starting training...")

        log_interval_start_time = time.time()
        loss_accumulator = 0.0
        acc_accumulator = 0.0
        log_interval = 100

        for i in range(1, config['train_config']['iteration'] + 1):
            model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            support_x, support_y, query_x, query_y, class_names = batch
            support_x, support_y, query_x, query_y = [x.to(device) for x in [support_x, support_y, query_x, query_y]]

            optimizer.zero_grad()
            if config.get('model_name') == 'dgcm':
                loss, acc = model(support_x, support_y, query_x, query_y, class_names)
            else:
                loss, acc = model(support_x, support_y, query_x, query_y)
            if 'protonet' not in config.get('model_name', ''):
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            loss_accumulator += loss.item()
            acc_accumulator += acc.item()

            if i % log_interval == 0:
                # --- 日志生成 ---
                timestamp_str = datetime.now().strftime('%H:%M:%S')
                duration_for_interval = time.time() - log_interval_start_time
                iter_speed = duration_for_interval / log_interval

                iterations_left = config['train_config']['iteration'] - i
                eta_seconds = iterations_left * iter_speed
                eta_str = str(timedelta(seconds=int(eta_seconds)))

                avg_loss = loss_accumulator / log_interval
                avg_acc = acc_accumulator / log_interval
                current_lr = lr_scheduler.get_last_lr()[0]

                log_str = (
                    f"Time: {timestamp_str} | "
                    f"Iter: {i}/{config['train_config']['iteration']} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time/{log_interval}iter: {duration_for_interval:.2f}s | "
                    f"ETA: {eta_str}"
                )
                print(log_str)
                with open(log_file, 'a') as f: f.write(log_str + '\n')

                loss_accumulator = 0.0
                acc_accumulator = 0.0
                log_interval_start_time = time.time()

            if i % config['eval_config']['interval'] == 0:
                print(f"\n--- Starting Validation at Iteration {i} ---")
                avg_val_acc, _ = run_evaluation(config, model, val_loader, device)
                log_str = f"--- Validation at iteration {i} | Avg Acc: {avg_val_acc:.4f} ---"
                print(log_str)
                with open(log_file, 'a') as f:
                    f.write(log_str + '\n')

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                    print(f"*** Best validation accuracy updated: {best_val_acc:.4f}. Model saved. ***\n")

        print("--- Training finished. ---")

        print("Loading best model for final test...")
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
        test_loader = get_dataloader(config, 'test')
        avg_test_acc, ci95 = run_evaluation(config, model, test_loader, device)
        log_str = f"Final Test Results: Avg Acc: {avg_test_acc:.4f}, 95% CI: {ci95:.4f}"
        print(log_str)
        with open(log_file, 'a') as f:
            f.write("=========================================\n" + log_str + '\n')

    elif args.mode == 'eval':

        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

        print(f"Loading best model from {checkpoint_path} for evaluation...")
        model.load_state_dict(torch.load(checkpoint_path))

        print("Getting test dataloader...")
        test_loader = get_dataloader(config, 'test')

        print("Running evaluation on the test set...")
        avg_test_acc, ci95 = run_evaluation(config, model, test_loader, device)

        print("\n" + "=" * 30)
        print("      Final Evaluation Results")
        print("=" * 30)
        print(f"  Config: {args.config}")
        print(f"  Dataset: {config['dataset_name']}")
        print(f"  Model: {config.get('model_name', 'dgcm')}")
        print(f"  Avg Test Accuracy: {avg_test_acc:.4f}")
        print(f"  95% Confidence Interval: +/- {ci95:.4f}")
        print("=" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a Few-Shot Learning model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Name of the config file to use (without .py extension)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Execution mode: "train" or "eval"')
    args = parser.parse_args()
    main(args)

