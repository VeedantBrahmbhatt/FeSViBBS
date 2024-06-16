import os
import numpy as np
import models
import random
from dataset import skinCancer, bloodmnisit, isic2019
from utils import weight_dec_global, weight_vec
import argparse
import torch as torch
from torch import nn
from scipy.stats import wasserstein_distance
import math

# def store_intermediate_features(model, input_data, client_id):
#     # Function to store intermediate features from each block for a specific client
#     features = []
#
#     def hook(module, input, output):
#         features.append(output.detach().cpu().numpy())
#
#     hooks = []
#     for block in model.clients[client_id].vit.blocks:
#         hooks.append(block.register_forward_hook(hook))
#
#     with torch.no_grad():
#         _ = model(input_data.to(next(model.parameters()).device))
#
#     for hook in hooks:
#         hook.remove()
#
#     return features

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
def store_intermediate_features(model, input_data, client_id):
    """Stores intermediate features for a specific client."""
    intermediate_features = []
    intermediate_features_tuple=[]
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for block_num in range(model.initial_block, model.final_block + 1):
            # Use model directly
            output = model(input_data, chosen_block=block_num, client_idx=client_id)
            intermediate_features.append(output.detach().cpu())
            intermediate_features_tuple.append((block_num, output.detach().cpu()))
    return intermediate_features,intermediate_features_tuple
def calculate_wasserstein_distance(original, features):
    distances = []
    for feature in features:
        distance = wasserstein_distance(original.flatten(), feature.flatten())
        distances.append(distance)
    return distances


def fesvibs(
        dataset_name, lr, batch_size, Epochs, input_size, num_workers, save_every_epochs,
        model_name, pretrained, opt_name, seed, base_dir, root_dir, csv_file_path, num_clients, DP,
        epsilon, delta, resnet_dropout, initial_block, final_block, fesvibs_arg, local_round
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    if fesvibs_arg:
        method_flag = 'FeSViBS'
    else:
        method_flag = 'SViBS'

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("Using CPU CUDA=FALSE recieved")

    if DP:
        std = np.sqrt(2 * np.math.log(1.25 / delta)) / epsilon
        mean = 0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch__{epsilon, delta}DP_{method_flag}"
    else:
        mean = 0
        std = 0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch_{method_flag}"

    save_dir = f'{dir_name}'
    os.mkdir(save_dir)

    print(f"Logging to: {dir_name}")

    print('Getting the Dataset and Dataloader!')
    if dataset_name == 'HAM':
        num_classes = 7
        _, _, traindataset, testdataset = skinCancer(input_size=input_size, batch_size=batch_size, base_dir=base_dir,
                                                     num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'bloodmnist':
        num_classes = 8
        _, _, traindataset, testdataset = bloodmnisit(input_size=input_size, batch_size=batch_size, download=True,
                                                      num_workers=num_workers)
        num_channels = 3

    elif dataset_name == 'isic2019':
        num_classes = 8
        DATALOADERS, _, _, _, _, test_loader = isic2019(input_size=input_size, batch_size=batch_size, root_dir=root_dir,
                                                        csv_file_path=csv_file_path, num_workers=num_workers)
        num_channels = 3

    # criterion = ContrastiveLoss()
    criterion = nn.CrossEntropyLoss()


    fesvibs_network = models.FeSVBiS(
        ViT_name=model_name, num_classes=num_classes,
        num_clients=num_clients, in_channels=num_channels,
        ViT_pretrained=pretrained,
        initial_block=initial_block, final_block=final_block,
        resnet_dropout=resnet_dropout, DP=DP, mean=mean, std=std
    ).to(device)

    Split = models.SplitFeSViBS(
        num_clients=num_clients, device=device, network=fesvibs_network,
        criterion=criterion, base_dir=save_dir,
        initial_block=initial_block, final_block=final_block,
    )

    if dataset_name != 'isic2019':
        print('Distribute Images Among Clients')
        Split.distribute_images(dataset_name=dataset_name, train_data=traindataset, test_data=testdataset,
                                batch_size=batch_size)
    else:
        Split.CLIENTS_DATALOADERS = DATALOADERS
        Split.testloader = test_loader

    Split.set_optimizer(opt_name, lr=lr)
    Split.init_logs()

    print('Start Training! \n')

    client_blocks = {}  # Dictionary to store selected block for each client
    client_blocks_list = []  # List to store tuples of (client_name, block_number)

    for r in range(Epochs):
        print(f"Round {r + 1} / {Epochs}")
        agg_weights = None
        for client_i in range(num_clients):
            client_name = f'Client_{client_i}'
            if r == 0:
                # Store intermediate features in the first epoch
                input_data, _ = next(iter(Split.CLIENTS_DATALOADERS[client_i]))
                input_data = input_data.to(device)
                original_features = input_data.cpu().numpy()
                intermediate_features,tuple_info = store_intermediate_features(Split.network, input_data, client_i)
                distances = calculate_wasserstein_distance(original_features, intermediate_features)
                selected_block = distances.index(min(distances)) + initial_block
                client_blocks[client_i] = selected_block
                client_blocks_list.append((client_name, selected_block))
                print(f"Selected block for client {client_i}: {selected_block}")
            else:
                # Use the selected block for training
                Split.network.clients[client_i].update_final_block(
                    client_blocks[client_i])  # Assuming a method to update the final block

            weight_dict = Split.train_round(client_i)
            if client_i == 0:
                agg_weights = weight_dict
            else:
                agg_weights['blocks'] += weight_dict['blocks']
                agg_weights['cls'] += weight_dict['cls']
                agg_weights['pos_embed'] += weight_dict['pos_embed']

        agg_weights['blocks'] /= num_clients
        agg_weights['cls'] /= num_clients
        agg_weights['pos_embed'] /= num_clients

        Split.network.vit.blocks = weight_dec_global(
            Split.network.vit.blocks,
            agg_weights['blocks'].to(device)
        )

        Split.network.vit.cls_token.data = agg_weights['cls'].to(device) + 0.0
        Split.network.vit.pos_embed.data = agg_weights['pos_embed'].to(device) + 0.0

        if fesvibs_arg and ((r + 1) % local_round == 0 and r != 0):
            print('========================== \t \t Federation \t \t ==========================')
            tails_weights = []
            head_weights = []
            for head, tail in zip(Split.network.resnet50_clients, Split.network.mlp_clients_tail):
                head_weights.append(weight_vec(head).detach().cpu())
                tails_weights.append(weight_vec(tail).detach().cpu())

            mean_avg_tail = torch.mean(torch.stack(tails_weights), axis=0)
            mean_avg_head = torch.mean(torch.stack(head_weights), axis=0)

            for i in range(num_clients):
                Split.network.mlp_clients_tail[i] = weight_dec_global(Split.network.mlp_clients_tail[i],
                                                                      mean_avg_tail.to(device))
                Split.network.resnet50_clients[i] = weight_dec_global(Split.network.resnet50_clients[i],
                                                                      mean_avg_head.to(device))

        for client_i in range(num_clients):
            Split.eval_round(client_i)

        print('---------')

        if (r + 1) % save_every_epochs == 0 and r != 0:
            Split.save_pickles(save_dir)
        print('============================================')

    # Print the client blocks list at the end of training
    print("Client blocks list:", client_blocks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Centralized Experiments')
    parser.add_argument('--dataset_name', type=str, choices=['HAM', 'bloodmnist', 'isic2019'], help='Dataset Name')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--local_round', type=int, default=2,
                        help='Local round before federation in FeSViBS, default : 2')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--initial_block', type=int, default=1, help='Initial Block, default : 1')
    parser.add_argument('--final_block', type=int, default=6, help='Final Block, default : 6')
    parser.add_argument('--num_clients', type=int, default=6, help='Number of Clients, default: 6')
    parser.add_argument('--model_name', type=str, default='vit_base_r50_s16_224',
                        help='Model Name, default: vit_base_r50_s16_224')
    parser.add_argument('--fesvibs_arg', type=bool, default=True, help='FeSViBS argument flag, default: True')
    parser.add_argument('--pretrained', type=bool, default=True, help='Pretrained or not, default: True')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size, default: 8')
    parser.add_argument('--Epochs', type=int, default=20, help='Number of Epochs, default: 20')
    parser.add_argument('--opt_name', type=str, default='Adam', help='Optimizer Name, default: Adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate, default : 0.0001')
    parser.add_argument('--save_every_epochs', type=int, default=1, help='Save logs after every N epoch')
    parser.add_argument('--resnet_dropout', type=float, default=0.5, help='Dropout rate in resnet tail, default: 0.5')
    parser.add_argument('--base_dir', type=str, default='data', help='Base Directory for datasets')
    parser.add_argument('--root_dir', type=str, default='data/ISIC_2019_Training_Input',
                        help='Root Directory for ISIC 2019 Dataset')
    parser.add_argument('--csv_file_path', type=str, default='data/ISIC_2019_Training_GroundTruth.csv',
                        help='CSV file path for ISIC 2019 Dataset')
    parser.add_argument('--DP', type=bool, default=False, help='Differential Privacy flag, default: False')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon value for DP, default: 1.0')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta value for DP, default: 1e-5')
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')

    args = parser.parse_args()

    fesvibs(
        args.dataset_name, args.lr, args.batch_size, args.Epochs, args.input_size, args.num_workers,
        args.save_every_epochs, args.model_name, args.pretrained, args.opt_name, args.seed, args.base_dir,
        args.root_dir, args.csv_file_path, args.num_clients, args.DP, args.epsilon, args.delta,
        args.resnet_dropout, args.initial_block, args.final_block, args.fesvibs_arg, args.local_round
    )
