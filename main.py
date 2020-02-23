import argparse 
import torch.utils.data
from torchvision import datasets, transforms

from train import TrainerVaDE
from preprocess import get_mnist, get_webcam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of iterations")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='learning rate')
    parser.add_argument('--pretrained_path', type=str, default='weights/pretrained_parameter.pth',
                        help='Output path')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'webcam'],
                        help='Dataset to be used for training')
    parser.add_argument('--sup_mul', type=float, default=0.9,
                        help='Hyperparameters that control the supervised importance')
    parser.add_argument('--n_shots', type=int, default=1,
                        help='Number of supervised points to be used')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        dataloader_sup, dataloader_unsup = get_mnist(args)
        n_classes = 10
    else:
        dataloader_sup, dataloader_unsup = get_webcam(args)
        n_classes = 31
    
    vade = TrainerVaDE(args, device, dataloader_sup, dataloader_unsup, n_classes)
    if args.pretrain==True:
        vade.pretrain()
    vade.train()

