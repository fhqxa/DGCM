import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def compute_sparse_adj(features, k=5, tau=0.1):
    sim = torch.bmm(features, features.transpose(1, 2))
    topk_values, topk_indices = torch.topk(sim, k=k, dim=-1)
    mask_sim = torch.full_like(sim, float('-inf'))
    mask_sim.scatter_(dim=-1, index=topk_indices, src=topk_values)
    adj = F.softmax(mask_sim / tau, dim=-1)
    return adj


class FullGNNLayer(nn.Module):

    def __init__(self, emb_size, dropout_rate=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.linear = nn.Linear(emb_size, emb_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.scale = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        message = torch.bmm(adj, x)
        update = self.linear(message)
        update = self.act(update)
        update = self.dropout(update)
        x_new = x + self.scale * update
        return x_new


class DGCM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        print(f"INFO: Loading CLIP backbone: {config['backbone']} ...")
        self.clip_model, self.preprocess = clip.load(config['backbone'], device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.emb_size = self.clip_model.visual.output_dim
        dsp_config = config['dsp_config']


        dropout_rate = config['train_config'].get('dropout', 0.1)
        self.dsp_iterations = dsp_config['dsp_iterations']


        self.k_neighbors = dsp_config.get('k_neighbors', 5)
        print(f"INFO: DGCM initialized. Depth={self.dsp_iterations}, Sparse-K={self.k_neighbors}")


        self.igb_gnn = nn.ModuleList([FullGNNLayer(self.emb_size, dropout_rate) for _ in range(self.dsp_iterations)])
        self.pgb_gnn = nn.ModuleList([FullGNNLayer(self.emb_size, dropout_rate) for _ in range(self.dsp_iterations)])


        self.dataset_name = config.get('dataset_name', 'mini-imagenet')
        self.alpha = nn.Parameter(torch.tensor(dsp_config['alpha_init']))
        self.beta = nn.Parameter(torch.tensor(dsp_config['beta_init']))


        self.tau_instance = dsp_config.get('tau_instance', 0.1)
        self.tau_proto = dsp_config.get('tau_proto', 0.1)
        self.tau_guidance = dsp_config.get('tau_guidance', 0.1)


        self.alpha_res = 0.2

    def forward(self, support_x, support_y, query_x, query_y, class_names):
        B, NK, C, H, W = support_x.shape
        Q = query_x.shape[1]
        N = self.config['train_config']['num_ways']
        K = self.config['train_config']['num_shots']


        all_images = torch.cat([support_x, query_x], dim=1).view(B * (NK + Q), C, H, W)
        with torch.no_grad():
            V = self.clip_model.encode_image(all_images).float()
        V = F.normalize(V, p=2, dim=-1).view(B, NK + Q, -1)


        if isinstance(class_names[0], (list, tuple)):
            class_names_batch = list(zip(*class_names))
        else:
            class_names_batch = class_names

        # 2. 展平处理
        flat_names = [name for episode in class_names_batch for name in episode]

        if self.dataset_name == 'cub200':
            # CUB 专用 Prompt (针对细粒度鸟类)
            prompts = [f"a photo of a {name}, a type of bird." for name in flat_names]
        else:
            # 其他数据集 (MiniImageNet, Tiered, CIFAR-FS 等) 使用通用 Prompt
            prompts = [f"a photo of a {name}." for name in flat_names]

        with torch.no_grad():
            tokenized = clip.tokenize(prompts).to(self.device)
            P = self.clip_model.encode_text(tokenized).float()


        P = F.normalize(P, p=2, dim=-1).view(B, N, -1)


        V_init = V.clone()
        P_init = P.clone()

        accumulated_loss = 0.0
        label_smoothing_value = self.config['train_config'].get('label_smoothing', 0.0)
        final_logits = None


        for i in range(self.dsp_iterations):

            adj_I = compute_sparse_adj(V, k=self.k_neighbors, tau=self.tau_instance)

            V = self.igb_gnn[i](V, adj_I)
            V = (1 - self.alpha_res) * V + self.alpha_res * V_init
            V = F.normalize(V, p=2, dim=-1)

            adj_P = compute_sparse_adj(P, k=min(self.k_neighbors, N), tau=self.tau_proto)

            P = self.pgb_gnn[i](P, adj_P)
            P = (1 - self.alpha_res) * P + self.alpha_res * P_init
            P = F.normalize(P, p=2, dim=-1)

            support_features = V[:, :NK, :].view(B, N, K, -1)
            VC = F.normalize(support_features.mean(dim=2), p=2, dim=-1)

            P_adapted = self.alpha * P + (1 - self.alpha) * VC
            P_adapted = F.normalize(P_adapted, p=2, dim=-1)

            similarities = torch.bmm(V, P_adapted.transpose(1, 2))
            soft_assignments = F.softmax(similarities / self.tau_guidance, dim=-1)
            guidance_vectors = torch.bmm(soft_assignments, P_adapted)
            V = self.beta * V + (1 - self.beta) * guidance_vectors
            V = F.normalize(V, p=2, dim=-1)

            curr_query_features = V[:, NK:, :]

            scale = 5.0
            curr_logits = scale * torch.bmm(curr_query_features, P_adapted.transpose(1, 2))

            step_loss = F.cross_entropy(curr_logits.view(B * Q, N), query_y.view(B * Q),
                                        label_smoothing=label_smoothing_value)
            accumulated_loss += step_loss
            final_logits = curr_logits

        final_loss = accumulated_loss / self.dsp_iterations

        with torch.no_grad():
            pred = torch.argmax(final_logits.view(B * Q, N), dim=-1)
            accuracy = (pred == query_y.view(B * Q)).float().mean()

        return final_loss, accuracy
