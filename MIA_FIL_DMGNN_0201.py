import sys
import numpy as np
import argparse
from tqdm import tqdm

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import dgl.function as fn
from dgl.utils import check_eq_shape, expand_as_pair

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .... import train_Dataset,valid_Dataset

class FILConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm=None,
        activation=None,
    ):
        super(FILConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.activation = activation
        #matrix P
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.reset_parameters()
    def reset_parameters(self):
        """
        Initialize linear weights
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        """
        graph : DGLGraph
        feat : Node characteristics
        edge_weight : Pearson coefficient between nodes
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")
            h_self = feat_dst
            # The case of boundless graphs
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)
            # (PX)W
            graph.srcdata["h"] = self.fc_neigh(feat_src)
            graph.update_all(msg_fn, fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            FIL_out = self.fc_self(h_self) + h_neigh
            if self.activation is not None:
                FIL_out = self.activation(FIL_out)
            if self.norm is not None:
                FIL_out = self.norm(FIL_out)
            return FIL_out


class SparseMHA(nn.Module):
    """
    Sparse Multi-head Attention Module
    """
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))
        attn = attn.softmax()
        out = dglsp.bspmm(attn, v)

        return self.out_proj(out.reshape(N, -1))

class GTLayer(nn.Module):
    """
    Graph Transformer Layer
    """
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(0.5)
    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)
        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h
        return self.batchnorm2(h)

class FILmodule(nn.Module):
    def __init__(self,
                 in_dim,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation):
        super(FILmodule, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.layers.append(FILConv(in_dim, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(FILConv(n_hidden, n_hidden))
        self.layers.append(FILConv(n_hidden, out_dim))

    def forward(self, graph, inputs, edge_features):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(graph, h, edge_features)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h


class ApplyNodeFunc(nn.Module):
    """
    Update the node feature hv with MLP, BN and ReLU.
    """
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(args.hidden_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, args.hidden_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((args.hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = h.to(torch.float32)
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class DMGNN(nn.Module):
    """
    DMGNN model
    """
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(DMGNN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.FIL = FILmodule(1,32,args.hidden_dim,2,F.relu)
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        self.linears_prediction_ts = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(args.hidden_dim, output_dim))
                self.linears_prediction_ts.append(
                    nn.Linear(args.hidden_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))
                self.linears_prediction_ts.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)
        hidden_size =64
        num_heads = 4
        num_layers_gt =num_layers+1
        self.gt_layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers_gt)]
        )
        self.pooler = dglnn.SumPooling()

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):

        edge_features = g.edata['weight']
        #FIL
        h = self.FIL(g,h,edge_features)
        #DMGNN
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices,edge_features, shape=(N, N))
        h_ts = self.gt_layers[0](A, h)
        hidden_rep = [h]
        hidden_h_ts_rep = [h_ts]
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h_ts = self.gt_layers[i+1](A, h)
            hidden_rep.append(h)
            hidden_h_ts_rep.append(h_ts)
        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            pooled_h_ts = self.pool(g, hidden_h_ts_rep[i])
            pooled_h = pooled_h.to(torch.float32)
            pooled_h_ts = pooled_h_ts.to(torch.float32)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            score_over_layer += self.drop(self.linears_prediction_ts[i](pooled_h_ts))
        return score_over_layer


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)


def train(args, net,train_loader, optimizer, criterion , epoch):
    net.train()
    running_loss = 0
    total_iters = len(train_loader)
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)
    for pos, (graphs, labels) in zip(bar,train_loader):
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('feat')
        outputs = net(graphs, feat)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    running_loss = running_loss / total_iters
    return running_loss


def eval_net(args, net, dataloader, criterion):
    net.eval()
    total = 0
    total_loss = 0
    total_correct = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('feat')
        total += len(labels)
        outputs = net(graphs,feat)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        labels_all = np.append(labels_all, labels.cpu().numpy())
        predict_all = np.append(predict_all, predicted.cpu().numpy())
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    return loss, acc

class Parser():

    def __init__(self, description):
        """
           arguments parser
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        self.parser.add_argument('-f')
        #data
        self.parser.add_argument(
            '--fold_ID', type=int, default=4,
            help='fold_ID in range[1,5] (min: 1, max:5)')
        self.parser.add_argument(
            '--batch_size', type=int, default=8,
            help='batch size for training and validation (default: 32)')

        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')


        ##MLP
        self.parser.add_argument(
            '--hidden_dim', type=int, default=32,
            help='number of hidden units (default: 16)')
        self.parser.add_argument(
            '--num_layers', type=int, default=5,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=4,
            help='number of MLP layers(default: 4). 1 means linear model.')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=42,
            help='random seed (default: 42)')
        self.parser.add_argument(
            '--epochs', type=int, default=600,
            help='number of epochs to train (default: 600)')
        self.parser.add_argument(
            '--lr', type=float, default=0.01,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.5,
            help='final layer dropout (default: 0.5)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="max", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="mean", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')
        # done
        self.args = self.parser.parse_args()


def main(args):
    # set up seeds, args.seed supported
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    is_cuda = not args.disable_cuda and torch.cuda.is_available()
    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")
    # 网格调参
    for a in range(1, 6):
        args.fold_ID = a
        print('fold_ID:', args.fold_ID)
        print('***********************************************')
        for b_s in [8]:
            args.batch_size = b_s
            print('batch_size:', args.batch_size)
            print('----------------------------------------------')
            for c in [64]:
                args.hidden_dim = c
                print('hidden_dim:', args.hidden_dim)
                print('----------------')
                for d in [0.01]:
                    args.lr = d
                    print('lr:', args.lr)
                    print('----------------')
                    for e in [0.5]:
                        args.final_dropout = e
                        print('final_dropout:', args.final_dropout)
                        fold_ID = args.fold_ID
                        # imput data
                        train_set = train_Dataset(fold_ID)
                        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                  collate_fn=collate)
                        valid_set = valid_Dataset(fold_ID)
                        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True,
                                                  collate_fn=collate)
                        train_set.dim_nfeats = 1
                        train_set.gclasses = 2

                        model = DMGNN(
                            args.num_layers, args.num_mlp_layers,
                            train_set.dim_nfeats, args.hidden_dim, train_set.gclasses,
                            args.final_dropout, args.learn_eps,
                            args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)
                        print(model)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                        tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
                        vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)

                        for epoch, _ in zip(tbar, vbar):
                            train(args, model, train_loader, optimizer, criterion, epoch)
                            scheduler.step()
                            train_loss, train_acc = eval_net(
                                args, model, train_loader, criterion)
                            tbar.set_description(
                                'train set - average loss: {:.4f}, accuracy: {:.3f}%'
                                    .format(train_loss, 100. * train_acc))

                            valid_loss, valid_acc = eval_net(
                                args, model, valid_loader, criterion)
                            vbar.set_description(
                                'valid set - average loss: {:.4f}, accuracy: {:.3f}%'
                                    .format(valid_loss, 100. * valid_acc))

                        tbar.close()
                        vbar.close()


if __name__ == '__main__':
    args = Parser(description='FIL_DMGNN').args
    print('show all arguments configuration...')
    print(args)
    main(args)