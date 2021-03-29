import os
import json
import yaml
import math
import numpy as np
from model.KERN import KERN
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utility import TrendData

from torch.utils.tensorboard import SummaryWriter


def test(conf, _print=True):
    # initiate log files
    log_path = "./log/%s/pred_%d/" %(conf["dataset"], conf["output_len"])
    settings = [conf["loss"]]
    if conf["use_grp_embed"] is True:
        settings.append("GrpEmb")
    if conf["lr_decay"] is True:
        settings.append("LRD%d" %(conf["lr_decay_interval"]))
    if conf["ext_kg"] is True:
        settings.append("ExtKG")
    if conf["int_kg"] is True:
        settings.append("IntKG_lambda:%.6f_SampleRange:%d" %(conf["triplet_lambda"], conf["sample_range"]))
    setting = "__".join(settings)
    log_path = log_path + setting + "/"
        
    # initiate dataset
    dataset = TrendData(conf)

    # check/update/print settings
    conf["grp_num"] = len(dataset.grp_id_map)
    conf["ele_num"] = len(dataset.ele_id_map)
    conf["time_num"] = dataset.time_num
    conf["seq_num"] = dataset.seq_num
    if conf["dataset"] == "FIT":
        conf["city_num"] = len(dataset.city_id_map)
        conf["gender_num"] = len(dataset.gender_id_map)
        conf["age_num"] = len(dataset.age_id_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    if _print:
        for k, v in conf.items():
            print(k, v)

    # initiate model, load parameters from the pre-trained model
    model = KERN(conf, adj=dataset.adj)
    best_model_path = log_path + "model.stat_dict"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    mae, mape, all_grd, all_pred, val_mae, val_mape, test_mae, test_mape, test_loss_scalars = evaluate(model, dataset, device, conf)
    each_print = "MAE: %.6f, MAPE: %.6f, VAL_MAE: %.6f, VAL_MAPE: %.6f, TEST_MAE: %.6f, TEST_MAPE: %.6f" %(mae, mape, val_mae, val_mape, test_mae, test_mape)

    if _print:
        print(each_print)
    else:
        return test_mae, test_mape


def evaluate(model, dataset, device, conf):
    model.eval()                                                                                                           
    pbar = tqdm(enumerate(dataset.test_loader), total=len(dataset.test_loader))                                                              
    loss_print, enc_loss_print, dec_loss_print = 0, 0, 0
    all_grd_ori, all_grd, all_pred, all_norm = [], [], [], []
    for batch_i, batch in pbar:
        each_cont = [[x.to(device) for x in each] for each in batch]
        enc_loss, dec_loss, pred, _, _, _ = model.predict(each_cont)

        enc_loss_scalar = enc_loss.detach().cpu()
        dec_loss_scalar = dec_loss.detach().cpu()
        loss_scalar = enc_loss_scalar + dec_loss_scalar
        
        loss_print += loss_scalar
        enc_loss_print += enc_loss_scalar
        dec_loss_print += dec_loss_scalar

        loss_final = loss_print/(batch_i+1)
        enc_loss_final = enc_loss_print/(batch_i+1)
        dec_loss_final = dec_loss_print/(batch_i+1)

        pbar.set_description('L:{:.6f}, EL:{:.6f}, DL:{:.6f}'.format(loss_print/(batch_i+1), enc_loss_print/(batch_i+1), dec_loss_print/(batch_i+1)))

        each_trend, each_ori_trend, metadata = each_cont
        input_seq, output_seq = each_trend
        input_seq_ori, output_seq_ori = each_ori_trend
        each_norm = metadata[3]

        all_grd.append(output_seq[:, :, 1].cpu())
        all_grd_ori.append(output_seq_ori[:, :, 1].cpu())
        all_pred.append(pred.detach().cpu())
        all_norm.append(each_norm.cpu())

    all_grd = torch.cat(all_grd, dim=0).numpy()
    all_grd_ori = torch.cat(all_grd_ori, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_norm = torch.cat(all_norm, dim=0).numpy()

    if conf["denorm"] is True:
        all_grd = all_grd_ori
        all_pred = denorm(all_pred, all_norm)

    val_pred = all_pred[::2]
    val_grd = all_grd[::2]
    test_pred = all_pred[1::2]
    test_grd = all_grd[1::2]
    mae = np.mean(np.abs(all_pred-all_grd))
    mape = np.mean(np.abs(all_pred-all_grd)/all_grd)*100
    val_mae = np.mean(np.abs(val_pred-val_grd))
    val_mape = np.mean(np.abs(val_pred-val_grd)/val_grd)*100
    test_mae = np.mean(np.abs(test_pred-test_grd))
    test_mape = np.mean(np.abs(test_pred-test_grd)/test_grd)*100

    return mae, mape, all_grd, all_pred, val_mae, val_mape, test_mae, test_mape, [loss_final, enc_loss_final, dec_loss_final]


def denorm(seq, norms):
    # seq: [num_samples]
    # norms: [num_samples, 3] 2nd-dim: min, max, eps
    #seq = np.min(seq, 1)
    #seq = np.max(seq, 0)
    seq_len = seq.shape[-1]
    min_v = np.expand_dims(norms[:, 0], axis=1).repeat(seq_len, axis=1)
    max_v = np.expand_dims(norms[:, 1], axis=1).repeat(seq_len, axis=1)
    eps = np.expand_dims(norms[:, 2], axis=1).repeat(seq_len, axis=1)
    denorm_res = seq * (max_v - min_v) + min_v
    return denorm_res


def main():
    conf = yaml.safe_load(open("./config.yaml"))

    dataset = conf["dataset"]
    conf = conf[dataset]
    conf["dataset"] = dataset

    test(conf)


if __name__ == "__main__":
    main()
