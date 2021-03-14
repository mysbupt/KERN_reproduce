import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class KERN(nn.Module):
    def __init__(self, conf, adj=None):
        super(KERN, self).__init__()
        self.conf = conf
        feat_size = conf["feat_size"]
        if self.conf["use_grp_embed"] is True:
            self.enc_input_size = feat_size * 3 + 1 # grp_emb + ele_emb + time_emb + last_v
            self.dec_input_size = feat_size * 3 # grp_emb + ele_emb + time_emb
        else:
            self.enc_input_size = feat_size * 2 + 1 # ele_emb + time_emb + last_v
            self.dec_input_size = feat_size * 2 # ele_emb + time_emb
        device = torch.device(conf["device"])
        self.device = device

        hidden_size = conf["rnn_hidden_size"]
        self.encoder = nn.LSTM(self.enc_input_size, hidden_size)
        self.decoder = nn.LSTM(self.dec_input_size, hidden_size, bidirectional=True)

        self.grp_embeds = nn.Embedding(conf["grp_num"], feat_size)
        self.ele_embeds = nn.Embedding(conf["ele_num"], feat_size)
        self.time_embeds = nn.Embedding(conf["time_num"], feat_size)
        if self.conf["dataset"] == "FIT":
            self.city_embeds = nn.Embedding(conf["city_num"], feat_size)
            self.gender_embeds = nn.Embedding(conf["gender_num"], feat_size)
            self.age_embeds = nn.Embedding(conf["age_num"], feat_size, padding_idx=0)
            self.city_gender_age_agg = nn.Linear(feat_size * 3, feat_size)

        self.enc_linear = nn.Linear(hidden_size, 1)
        self.dec_linear = nn.Linear(hidden_size*2, 1)
        self.loss_function = nn.L1Loss()

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
            #self.adj_2 = torch.matmul(self.adj, self.adj) # [grp_num, ele_num, ele_num]
            #self.adj_2 = F.normalize(self.adj_2, p=1, dim=1)
            #self.agg_prop = nn.Linear(feat_size * 3, feat_size)
            #self.agg_prog_exc = nn.Linear(feat_size * 2, feat_size)


    def predict(self, each_cont): 
        each_trend, each_ori_trend, metadata = each_cont
        input_seq, output_seq = each_trend
        
        if self.conf["dataset"] == "FIT":
            seq_id, each_grp, each_ele, each_norm, each_city, each_gender, each_age = metadata
            city_id = each_city.squeeze(1) # [batch_size]
            gender_id = each_gender.squeeze(1) # [batch_size]
            age_id = each_age.squeeze(1) # [batch_size]
            city_embed = self.city_embeds(city_id) #[batch_size, feat_size]
            gender_embed = self.gender_embeds(gender_id) #[batch_size, feat_size]
            age_embed = self.age_embeds(age_id) #[batch_size, feat_size]
            grp_embed = self.city_gender_age_agg(torch.cat([city_embed, gender_embed, age_embed], dim=1))
        else:
            seq_id, each_grp, each_ele, each_norm = metadata
            grp_id = each_grp.squeeze(1) # [batch_size]
            grp_embed = self.grp_embeds(grp_id) #[batch_size, feat_size]

        ele_id = each_ele.squeeze(1) # [batch_size]
        ori_ele_embed = self.ele_embeds(ele_id) #[batch_size, feat_size]

        # affiliation relation
        if self.conf["ext_kg"] is True:
            #ele_embed_prop_1 = torch.matmul(self.adj, self.ele_embeds.weight) # [ele_num, feat_size]
            #ele_embed_prop_2 = torch.matmul(self.adj_2, self.ele_embeds.weight) # [ele_num, feat_size]
            #ele_embed_prop_1 = ele_embed_prop_1[ele_id, :] #[batch_size, feat_size] 
            #ele_embed_prop_2 = ele_embed_prop_2[ele_id, :] #[batch_size, feat_size] 
            #ele_embed = self.agg_prop(torch.cat([ori_ele_embed, ele_embed_prop_1, ele_embed_prop_2], dim=1))
            ele_embed_prop_1 = torch.matmul(self.adj, self.ele_embeds.weight) # [ele_num, feat_size]
            ele_embed_prop_1 = ele_embed_prop_1[ele_id, :] #[batch_size, feat_size] 
            ele_embed = ori_ele_embed + ele_embed_prop_1
        else:
            ele_embed = ori_ele_embed

        enc_time_embed = self.time_embeds(input_seq[:, :, 0].long()) #[batch_size, enc_seq_len, feat_size]
        dec_time_embed = self.time_embeds(output_seq[:, :, 0].long()) #[batch_size, dec_seq_len, feat_size]
        # encode part:
        # input_seq: [batch_size, enc_seq_len, 2]
        enc_seq_len = input_seq.shape[1]
        enc_grp_embed = grp_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]
        enc_ele_embed = ele_embed.unsqueeze(1).expand(-1, enc_seq_len, -1) #[batch_size, enc_seq_len, feat_size]

        if self.conf["use_grp_embed"] is True:
            enc_input_seq = torch.cat([enc_grp_embed, enc_ele_embed, enc_time_embed, input_seq[:, :, 1].unsqueeze(-1)], dim=-1) #[batch_size, enc_seq_len, enc_input_size]
        else:
            enc_input_seq = torch.cat([enc_ele_embed, enc_time_embed, input_seq[:, :, 1].unsqueeze(-1)], dim=-1) #[batch_size, enc_seq_len, enc_input_size]
        enc_input_seq = enc_input_seq.permute(1, 0, 2) #[enc_seq_len, batch_size, enc_input_size]

        enc_outputs, (enc_hidden, enc_c) = self.encoder(enc_input_seq) #outputs: [seq_len, batch_size, hidden_size], hidden: [1, batch_size, hidden_size]

        enc_grd = input_seq[:, 1:, 1] #[batch_size, enc_seq_len-1]
        enc_output_feat = enc_outputs.permute(1, 0, 2)[:, 1:, :] #[batch_size, enc_seq_len-1, hidden_size]
        enc_pred = self.enc_linear(enc_output_feat).squeeze(-1) #[batch_size, enc_seq_len-1]

        # decode part:
        # output_seq: [batch_size, dec_seq_len, 2]
        dec_seq_len = output_seq.shape[1]
        dec_grp_embed = grp_embed.unsqueeze(1).expand(-1, dec_seq_len, -1) #[batch_size, dec_seq_len, feat_size]
        dec_ele_embed = ele_embed.unsqueeze(1).expand(-1, dec_seq_len, -1) #[batch_size, dec_seq_len, feat_size]
 
        if self.conf["use_grp_embed"] is True:
            dec_input_seq = torch.cat([dec_grp_embed, dec_ele_embed, dec_time_embed], dim=-1) #[batch_size, dec_seq_len, dec_input_size]
        else:
            dec_input_seq = torch.cat([dec_ele_embed, dec_time_embed], dim=-1) #[batch_size, dec_seq_len, dec_input_size]
        dec_input_seq = dec_input_seq.permute(1, 0, 2) #[dec_seq_len, batch_size, dec_input_size]

        dec_init_hidden = enc_hidden.expand(2, -1, -1) #[2, batch_size, hidden_size]
        dec_init_c = enc_c.expand(2, -1, -1) #[2, batch_size, hidden_size]
        dec_output_feat, _ = self.decoder(dec_input_seq, (dec_init_hidden.contiguous(), dec_init_c.contiguous())) #outputs: [seq_len, batch_size, hidden_size*2]

        dec_output_feat = dec_output_feat.permute(1, 0, 2) # [batch_size, seq_len, hidden_size*2]
        dec_grd = output_seq[:, :, 1] #[batch_size, dec_seq_len]
        dec_pred = self.dec_linear(dec_output_feat).squeeze(-1) #[batch_size, dec_seq_len]

        enc_loss = self.loss_function(enc_pred, enc_grd)
        dec_loss = self.loss_function(dec_pred, dec_grd)

        return enc_loss, dec_loss, dec_pred, enc_hidden.squeeze(0), enc_output_feat, dec_output_feat # [batch_size, hidden_size], [batch_size, enc_seq_len-1, hidden_size], [batch_size, seq_len, hidden_size*2] 


    def forward(self, self_cont, close_cont, far_cont, close_score, far_score):
        self_enc_loss, self_dec_loss, self_pred, self_enc_hidden, self_enc_output, self_dec_output = self.predict(self_cont)
        if self.conf["int_kg"] is True:
            close_enc_loss, close_dec_loss, close_pred, close_enc_hidden, close_enc_output, close_dec_output = self.predict(close_cont)
            far_enc_loss, far_dec_loss, far_pred, far_enc_hidden, far_enc_output, far_dec_output = self.predict(far_cont)

            close_score = close_score.squeeze(1) # [batch_size]
            far_score = far_score.squeeze(1) # [batch_size]

            def cal_triplet_loss(self_dec_output, close_dec_output, far_dec_output):
                close_dist = (self_dec_output - close_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
                close_dist = torch.sqrt(close_dist + 1e-8*torch.ones(close_dist.shape).to(self.device))
                close_dist = torch.sum(close_dist, dim=2) # [batch_size, seq_len]
                far_dist = (self_dec_output - far_dec_output).pow(2) # [batch_size, seq_len, hidden_size*2]
                far_dist = torch.sqrt(far_dist + 1e-8*torch.ones(far_dist.shape).to(self.device))
                far_dist = torch.sum(far_dist, dim=2) # [batch_size, seq_len]
                residual = close_dist - far_dist
                triplet_loss = torch.max(torch.zeros(residual.shape).to(self.device), residual) # [batch_size, seq_len]
                triplet_loss = torch.mean(torch.mean(triplet_loss, dim=1))
                return triplet_loss

            enc_triplet_loss = cal_triplet_loss(self_enc_output, close_enc_output, far_enc_output)
            dec_triplet_loss = cal_triplet_loss(self_dec_output, close_dec_output, far_dec_output)
            triplet_loss = enc_triplet_loss + dec_triplet_loss

        if self.conf["int_kg"] is False:
            enc_loss = self_enc_loss
            dec_loss = self_dec_loss
            triplet_loss = None
        else:
            enc_loss = (self_enc_loss + close_enc_loss + far_enc_loss) / 3
            dec_loss = (self_dec_loss + close_dec_loss + far_dec_loss) / 3

        #enc_loss = (self_enc_loss + close_enc_loss + far_enc_loss) / 3
        #dec_loss = (self_dec_loss + close_dec_loss + far_dec_loss) / 3

        return enc_loss, dec_loss, triplet_loss
