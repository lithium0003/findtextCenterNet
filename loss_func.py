import torch

from util_func import modulo_list

# Multi-Loss Weighting with Coefficient of Variations
# https://arxiv.org/abs/2009.01717
# https://github.com/rickgroen/cov-weighting
class CoVWeightingLoss(torch.nn.modules.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.momentum = kwargs.pop('momentum', 1e-3)
        self.device = kwargs.pop('device', 'cpu')
        self.losses = kwargs.pop('losses', [])
        self.num_losses = len(self.losses)
        super().__init__(*args, **kwargs)

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False, dtype=torch.float32, device=self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False, dtype=torch.float32, device=self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False, dtype=torch.float32, device=self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False, dtype=torch.float32, device=self.device)
        self.running_std_l = None

    def forward(self, losses):
        L = torch.stack([losses[key].clone().detach().to(torch.float32) for key in self.losses])
        L = torch.nan_to_num(L)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0
        l = torch.nan_to_num(l)

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False, dtype=torch.float32, device=self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            ls = torch.nan_to_num(ls)
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        else:
            mean_param = (1. - max(self.momentum, 1 / (self.current_iter + 1)))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l.clamp_min(1e-16))

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * losses[key].to(torch.float32) for i,key in enumerate(self.losses)]
        loss = sum(weighted_losses)
        return loss

def heatmap_loss(true, logits):
    alpha = 2
    beta = 4
    pos_th = 1.0

    logits32 = logits.to(torch.float32)
    predict = torch.sigmoid(logits32)

    pos_mask = (true >= pos_th).to(torch.float32)
    neg_mask = (true < pos_th).to(torch.float32)

    neg_weights = torch.pow(1. - true, beta)

    pos_loss = - torch.nn.functional.logsigmoid(logits32) * torch.pow(1 - predict, alpha) * pos_mask
    neg_loss = (logits32 + torch.nn.functional.softplus(-logits32)) * torch.pow(predict, alpha) * neg_weights * neg_mask

    loss = (pos_loss + neg_loss).mean()

    return loss.to(logits.dtype)

def loss_function(fmask, labelmap, idmap, heatmap, decoder_outputs):
    key_th1 = 0.75
    key_th2 = 0.75
    key_th3 = 0.99

    keylabel = labelmap[:,0,:,:]
    mask1 = keylabel > key_th1
    mask3 = keylabel.flatten()[fmask] > key_th3
    mask4 = keylabel.flatten()[fmask] == 1.0

    weight1 = torch.maximum(keylabel - key_th1, torch.tensor(0.)) / (1 - key_th1)
    weight1 = torch.masked_select(weight1, mask1)
    weight1_count = torch.maximum(torch.tensor(1.), weight1.sum())
    weight2 = torch.maximum(keylabel - key_th2, torch.tensor(0.)) / (1 - key_th2)
    weight3 = torch.maximum(keylabel - key_th3, torch.tensor(0.)) / (1 - key_th3)
    weight3 = torch.masked_select(weight3.flatten()[fmask], mask3)
    weight3_count = torch.maximum(torch.tensor(1.), weight3.sum())

    keymap_loss = heatmap_loss(true=keylabel, logits=heatmap[:,0,:,:]) * 1e3

    huber = torch.nn.HuberLoss(reduction='none')
    xsize_loss = huber(torch.masked_select(heatmap[:,1,:,:], mask1), torch.masked_select(labelmap[:,1,:,:], mask1))
    ysize_loss = huber(torch.masked_select(heatmap[:,2,:,:], mask1), torch.masked_select(labelmap[:,2,:,:], mask1))
    size_loss = (xsize_loss + ysize_loss) * weight1
    size_loss = size_loss.sum() / weight1_count

    textline_loss = torch.nn.functional.binary_cross_entropy_with_logits(heatmap[:,3,:,:], labelmap[:,3,:,:], pos_weight=torch.tensor([3.], dtype=heatmap.dtype, device=heatmap.device))
    separator_loss = torch.nn.functional.binary_cross_entropy_with_logits(heatmap[:,4,:,:], labelmap[:,4,:,:], pos_weight=torch.tensor([3.], dtype=heatmap.dtype, device=heatmap.device))

    code_losses = {}
    for i in range(4):
        label_map = ((idmap[:,1,:,:] & 2**(i)) > 0).to(torch.float32)
        predict_map = heatmap[:,5+i,:,:]
        weight = torch.ones_like(label_map) + label_map * 30 + weight2 * 10
        code_loss = torch.nn.functional.binary_cross_entropy_with_logits(predict_map, label_map, weight=weight)
        code_losses['code%d_loss'%2**(i)] = code_loss * 10
    
    target_id = idmap[:,0,:,:].flatten()[fmask]
    target_ids = []
    for modulo in modulo_list:
        target_id1 = target_id % modulo
        target_ids.append(target_id1)

    id_loss = 0.
    for target_id1, decoder_id1 in zip(target_ids, decoder_outputs):
        target_id1 = torch.masked_select(target_id1, mask3)
        decoder_id1 = decoder_id1[mask3,:]
        id1_loss = torch.nn.functional.cross_entropy(decoder_id1, target_id1, reduction='none')
        id1_loss = (id1_loss * weight3).sum() / weight3_count
        id_loss += id1_loss

    pred_ids = []
    for decoder_id1 in decoder_outputs:
        pred_id1 = torch.argmax(decoder_id1[mask4,:], dim=-1)
        pred_ids.append(pred_id1)

    target_id = torch.masked_select(target_id, mask4)
    target_ids = []
    for modulo in modulo_list:
        target_id1 = target_id % modulo
        target_ids.append(target_id1)

    correct = torch.zeros_like(pred_ids[0])
    for p,t in zip(pred_ids,target_ids):
        correct += p == t

    total = torch.ones_like(correct).sum()
    correct = (correct == 3).sum()

    loss = keymap_loss + size_loss + textline_loss + separator_loss + id_loss
    for c_loss in code_losses.values():
        loss += c_loss

    return {
        'loss': loss,
        'keymap_loss': keymap_loss,
        'size_loss': size_loss,
        'textline_loss': textline_loss,
        'separator_loss': separator_loss,
        'id_loss': id_loss,
        **code_losses,
        'correct': correct,
        'total': total,
    }