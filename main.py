# coding=utf-8
import os
import torch.backends.cudnn as cudnn
from GCN_model import GCN,Attention
from encode_model import GCNModelVAE,InnerProductDecoder
from utils import *
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import random
import torch.utils.data
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--n_epoch', type=int, default=2000)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--gfeat', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--nfeat', type=float, default=6775,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_src', type=str, default='citationv1',
                    help='source dataset name')
parser.add_argument('--data_trg', type=str, default='acmv9',
                    help='target dataset name')
parser.add_argument('--classes', type=int, default=5,
                    help='classes number')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--lambda_d', type=float, default=0.5,
                    help='hyperparameter for domain loss')
parser.add_argument('--lambda_r', type=float, default=1,
                    help='hyperparameter for reconstruction loss')
parser.add_argument('--lambda_f', type=float, default=0.0001,
                    help='hyperparameter for different loss')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
cuda = True
cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

''' Load data '''
adj_s, features_s, labels_s, idx_s,X_n_s = load_data_citation(dataset=args.data_src)
adj_t, features_t, labels_t, idx_t,X_n_t = load_data_citation(dataset=args.data_trg)

''' Load adj labels for reconstruction '''
adj_label_s,pos_weight_s,norm_s = load_adj_label_for_reconstruction(dataset_name=args.data_src)
adj_label_t,pos_weight_t,norm_t = load_adj_label_for_reconstruction(dataset_name=args.data_trg)

def predict(feature,adj,ppmi):
    _,basic_encoded_output,_ = shared_encoder_l(feature,adj)
    _,ppmi_encoded_output,_ = shared_encoder_g(feature,ppmi)
    encoded_output = att_model([basic_encoded_output,ppmi_encoded_output])
    logits = cls_model(encoded_output)
    return logits

def evaluate(preds, labels):
    accuracy1 = accuracy(preds, labels)
    return accuracy1

def test(feature,adj,ppmi,label):
    for model in models:
        model.eval()
    logits = predict(feature,adj,ppmi)
    labels = label
    accuracy = evaluate(logits, labels)
    return accuracy

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

def recon_loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

''' set loss function '''
loss_diff = DiffLoss()
cls_loss = nn.CrossEntropyLoss().to(device)
domain_loss = torch.nn.NLLLoss()

''' load model '''
''' private encoder/encoder for S/T (including Local GCN and Global GCN) '''
private_encoder_s_l = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat, dropout=args.dropout).to(device)
private_encoder_t_l = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat, dropout=args.dropout).to(device)
private_encoder_s_g = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat, dropout=args.dropout).to(device)
private_encoder_t_g = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat, dropout=args.dropout).to(device)
decoder_s = InnerProductDecoder(dropout=args.dropout, act=lambda x: x)
decoder_t = InnerProductDecoder(dropout=args.dropout, act=lambda x: x)

''' shared encoder (including Local GCN and Global GCN) '''
shared_encoder_l = GCN(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)
shared_encoder_g = GCN(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)

''' node classifier model '''
cls_model = nn.Sequential(
    nn.Linear(args.gfeat, args.classes),
).to(device)

''' domain discriminator model '''
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(args.gfeat, 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(10, 2),
).to(device)

''' attention layer for local and global features '''
att_model = Attention(args.gfeat).cuda()
att_model_self_s = Attention(args.gfeat).cuda()
att_model_self_t = Attention(args.gfeat).cuda()

''' the set of models used in ASN '''
models = [private_encoder_s_l,private_encoder_s_g,private_encoder_t_l,private_encoder_t_g, shared_encoder_g,shared_encoder_l,cls_model,domain_model,decoder_s,decoder_t,att_model,att_model_self_s,att_model_self_t]
params = itertools.chain(*[model.parameters() for model in models])

''' setup optimizer '''
optimizer = torch.optim.Adam(params, lr=args.lr,weight_decay=5e-4)

''' training '''
best_acc = 0

for epoch in range(args.n_epoch):
    len_dataloader = min(labels_s.shape[0],labels_t.shape[0])
    global rate
    rate = min((epoch + 1) / args.n_epoch, 0.05)

    for model in models:
        model.train()
    optimizer.zero_grad()

    if cuda:
        adj_s = adj_s.cuda()
        adj_t = adj_t.cuda()
        labels_s = labels_s.cuda()
        labels_t = labels_t.cuda()
        features_s = features_s.cuda()
        features_t = features_t.cuda()
        X_n_s = X_n_s.cuda()
        X_n_t = X_n_t.cuda()
        adj_label_s = adj_label_s.cuda()
        adj_label_t = adj_label_t.cuda()
        pos_weight_s = pos_weight_s.cuda()
        pos_weight_t = pos_weight_t.cuda()

    recovered_s, mu_s, logvar_s = private_encoder_s_l(features_s, adj_s)
    recovered_t, mu_t, logvar_t = private_encoder_t_l(features_t, adj_t)

    recovered_s_p, mu_s_p, logvar_s_p = private_encoder_s_g(features_s, X_n_s)
    recovered_t_p, mu_t_p, logvar_t_p = private_encoder_t_g(features_t, X_n_t)

    z_s, shared_encoded_source1, shared_encoded_source2 = shared_encoder_l(features_s, adj_s)
    z_t, shared_encoded_target1, shared_encoded_target2 = shared_encoder_l(features_t, adj_t)

    z_s_p,ppmi_encoded_source,ppmi_encoded_source2 = shared_encoder_g(features_s, X_n_s)
    z_t_p,ppmi_encoded_target,ppmi_encoded_target2 = shared_encoder_g(features_t, X_n_t)

    ''' the node representations after shared encoder for S and T '''
    encoded_source = att_model([shared_encoded_source1,ppmi_encoded_source])
    encoded_target = att_model([shared_encoded_target1,ppmi_encoded_target])

    ''' compute encoder difference loss for S and T '''
    diff_loss_s = loss_diff(mu_s,shared_encoded_source1)
    diff_loss_t = loss_diff(mu_t, shared_encoded_target1)
    diff_loss_all =  diff_loss_s + diff_loss_t

    ''' compute decoder reconstruction loss for S and T '''
    z_cat_s = torch.cat((att_model_self_s([recovered_s,recovered_s_p]),att_model_self_s([z_s,z_s_p])),1)
    z_cat_t = torch.cat((att_model_self_t([recovered_t,recovered_t_p]),att_model_self_t([z_t,z_t_p])),1)
    recovered_cat_s = decoder_s(z_cat_s)
    recovered_cat_t = decoder_t(z_cat_t)
    mu_cat_s = torch.cat((mu_s, mu_s_p, shared_encoded_source1, ppmi_encoded_source), 1)
    mu_cat_t = torch.cat((mu_t, mu_t_p, shared_encoded_target1, ppmi_encoded_target), 1)
    logvar_cat_s = torch.cat((logvar_s, logvar_s_p, shared_encoded_source2, ppmi_encoded_source2), 1)
    logvar_cat_t = torch.cat((logvar_t, logvar_t_p, shared_encoded_target2, ppmi_encoded_target2), 1)
    recon_loss_s = recon_loss(preds=recovered_cat_s, labels=adj_label_s,
                         mu=mu_cat_s, logvar=logvar_cat_s, n_nodes=features_s.shape[0],
                         norm=norm_s, pos_weight=pos_weight_s)
    recon_loss_t = recon_loss(preds=recovered_cat_t, labels=adj_label_t,
                              mu=mu_cat_t, logvar=logvar_cat_t, n_nodes=features_t.shape[0]*2,
                              norm=norm_t, pos_weight=pos_weight_t)
    recon_loss_all =  recon_loss_s + recon_loss_t

    ''' compute node classification loss for S '''
    source_logits = cls_model(encoded_source)
    cls_loss_source = cls_loss(source_logits, labels_s)
    source_acc = evaluate(source_logits, labels_s)

    ''' compute domain classifier loss for both S and T '''
    domain_output_s = domain_model(encoded_source)
    domain_output_t = domain_model(encoded_target)
    err_s_domain = cls_loss(domain_output_s,
        torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(device))
    err_t_domain = cls_loss(domain_output_t,
        torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(device))
    loss_grl = err_s_domain + err_t_domain

    ''' compute entropy loss for T '''
    target_logits = cls_model(encoded_target)
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

    ''' compute overall loss '''
    loss = cls_loss_source + args.lambda_d * loss_grl + args.lambda_r * recon_loss_all + args.lambda_f * diff_loss_all + loss_entropy * (epoch / args.n_epoch * 0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        acc_trg = test(features_t,adj_t,X_n_t,labels_t)
        if acc_trg > best_acc:
            best_acc = acc_trg
        print('epoch: {}, acc_test_trg: {},loss_class:{},loss_domain:{},loss_recon:{},loss_diff:{}'.format(epoch,acc_trg.item(),cls_loss_source.item(),args.lambda_d * loss_grl.item(), args.lambda_r * recon_loss_all.item(),args.lambda_f * diff_loss_all.item()))

print('best acc :{}'.format(best_acc))
print('done')
print('lr:{},d:{},r:{},f:{}'.format(args.lr,args.lambda_d,args.lambda_r,args.lambda_f))