import torch
import torch.nn as nn

class PAN(nn.Module):
    def __init__(self, input_dim, num_supports, args):
        super(PAN, self).__init__()
        self.ge = args.ge
        if self.ge:
            self.num_units = [input_dim] + args.hidden
            self.enc_layers = nn.ModuleList([])
            for i in range(len(self.num_units)-1):
                self.enc_layers.append(GCN(
                    input_dim=self.num_units[i],
                    output_dim=self.num_units[i+1],
                    num_supports=num_supports,
                    dropout=args.dropout
                ))
        else:
            self.num_units = [input_dim]

        # decoder below is the same as the similarity module
        self.decoder = MLPDecoder(args, input_dim=self.num_units[-1],
                                  output_dim=args.nout)
    
    def forward(self, inp_feats, supports, row_idxs, col_idxs):
        x = inp_feats
        if self.ge:
            for layer in self.enc_layers:
                x = layer(x, supports)
        node_feats = x
        out, attrs, imps = self.decoder(node_feats, row_idxs, col_idxs)
        return out, attrs, imps # edge prediction scores

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_supports, dropout=0.):
        # NOTE : dropout is drop probability
        super(GCN, self).__init__()
        self.wts = nn.ParameterList([nn.Parameter(
            torch.FloatTensor(input_dim, output_dim))
            for i in range(num_supports)])
        # initialize wts
        for wt in self.wts:
            nn.init.kaiming_normal_(wt)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, inp_feats, supports):
        """
        Receives as input all the node features in a single tensor
        :param inp_feats: node features
        :param supports: supports (connectivity in the graph)
        :return: output node features
        """
        inp_feats = self.dropout(inp_feats)
        out = None
        for i, support in enumerate(supports):
            # support : sparse tensor (num_nodes x num_nodes)
            # x : num_nodes x input_dim
            # weights : input_dim x output_dim
            curr_term = torch.matmul(
                support, torch.matmul(inp_feats, self.wts[i]))

            if i == 0:
                out = curr_term
            else:
                out += curr_term
        out = self.act(out)
        out = self.bn(out)

        return out

class MLPDecoder(nn.Module):
    def __init__(self, args, input_dim, output_dim, dropout=0.):
        super(MLPDecoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats, row_idxs, col_idxs):
        # Returns a tensor of size len(row_idxs) x 1
        node_feats = self.dropout(node_feats)
        feats1 = node_feats[row_idxs]
        feats2 = node_feats[col_idxs]

        jfeat = torch.abs(feats1 - feats2)

        attrs = torch.sigmoid(self.fc1(jfeat))
        attrs = attrs.clamp(0, 1) # clamp for safety; used in BCELoss
        imps = torch.softmax(self.fc2(jfeat), dim=1) # importance scores

        out = (attrs * imps).sum(dim=1)
        out = out.squeeze()
        out = out.clamp(0, 1)
        return out, attrs, imps


