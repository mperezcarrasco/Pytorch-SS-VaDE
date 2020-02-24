import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.linear_assignment_ import linear_assignment

class ComputeLosses:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.alpha = args.sup_mul

    def forward(self, mode, x_sup, y_sup, x_unsup=None):
        if mode =='train':
            reconst_loss, kl_div = self.unsupervised_loss(x_unsup)
            unsupervised_loss = reconst_loss + kl_div

            reconst_loss, kl_div, probs = self.supervised_loss(x_sup, y_sup)
            supervised_loss = reconst_loss + kl_div
            loss = self.alpha * supervised_loss + (1 - self.alpha) * unsupervised_loss
            acc = self.compute_metrics(y_sup, probs)

        elif mode=='test':
            reconst_loss, kl_div, probs = self.supervised_loss(x_sup, y_sup)
            loss = reconst_loss + kl_div
            acc = self.compute_metrics(y_sup, probs)

        return loss, reconst_loss, kl_div, acc*100

    def supervised_loss(self, x, y):
        x_hat, mu, log_var, z = self.model(x)

        means_batch = self.model.mu_prior[y, :]
        covs_batch = self.model.log_var_prior[y,:].exp()
        p_c = self.model.pi_prior

        kl_div = torch.mean((torch.sum(torch.log(2*np.pi*covs_batch) + \
                            torch.exp(log_var)/covs_batch + \
                            torch.pow(mu-means_batch,2)/covs_batch - \
                            (1+log_var),dim=1)*0.5 - z.size(1)*0.5*np.log(2*np.pi)))

        reconst_loss = F.mse_loss(x_hat, x)

        probs = self.compute_pcz(z, p_c)

        return reconst_loss, kl_div, probs

    def unsupervised_loss(self, x):
        x_hat, mu, log_var, z = self.model(x)

        means = self.model.mu_prior
        covs = self.model.log_var_prior.exp()
        p_c = self.model.pi_prior
        
        gamma = self.compute_pcz(z, p_c)

        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - means).pow(2)
        h = torch.sum(torch.log(covs) + h / covs, dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h)
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-20))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-20))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

        kl_div = log_p_z_given_c - log_p_c +  log_q_c_given_x - log_q_z_given_x
        kl_div /= x.size(0)

        reconst_loss = F.mse_loss(x_hat, x)
        return reconst_loss, kl_div
    
    def compute_pcz(self, z, p_c):
        covs = self.model.log_var_prior.exp()
        means = self.model.mu_prior

        h = (z.unsqueeze(1) - means).pow(2) / covs
        h += torch.log(2*np.pi*covs)
        p_z_c = torch.exp(torch.log(p_c + 1e-20).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-20
        p_z_given_c = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return p_z_given_c
    
    def compute_metrics(self, y, probs):
        y_pred = np.argmax(probs.cpu().detach().numpy(), axis=1)
        y_true = y.cpu().detach().numpy()

        acc = accuracy_score(y_pred, y_true)
        #f1_macro = f1_score(y_pred, y_true, average='macro')

        return acc #, f1_macro

    def cluster_acc(self, real, pred):
        D = max(pred.max(), real.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(pred.size):
            w[pred[i], real[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/pred.size*100, w
