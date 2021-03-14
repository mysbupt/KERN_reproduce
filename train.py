import os
import json
import yaml
import math
import numpy as np
from model.KERN import KERN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from utility import TrendData

from torch.utils.tensorboard import SummaryWriter


def train(conf):
    # initiate log files
    run_path = "./runs/%s/pred_%d/" %(conf["dataset"], conf["output_len"])
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
        
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    run = SummaryWriter(run_path + setting)
    result_output = open(log_path + "result_output.txt", "w")

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

    for k, v in conf.items():
        print(k, v)

    # initiate model/optimizer
    model = KERN(conf, adj=dataset.adj)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf["lr_decay_interval"]*len(dataset.train_loader)), gamma=conf["lr_decay_gamma"])

    # start training and evaluation
    best_epoch, best_batch, best_mae, best_mape = 0, 0, 1, 100
    best_val_mae, best_val_mape, best_test_mae, best_test_mape = 1, 100, 1, 100
    ttl_batch = len(dataset.train_loader)
    for epoch in range(int(conf["epoch"])):
        print("\n%s Epoch: %d" %(setting, epoch))
        loss_print, enc_loss_print, dec_loss_print, triplet_loss_print = 0, 0, 0, 0
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        
        for batch_i, batch in pbar:
        #for batch_i, batch in enumerate(dataset.train_loader):
            model.train(True)
            optimizer.zero_grad()

            self_cont, close_cont, far_cont, close_score, far_score = batch

            self_cont = [[x.to(device) for x in each] for each in self_cont]
            close_cont = [[x.to(device) for x in each] for each in close_cont]
            far_cont = [[x.to(device) for x in each] for each in far_cont]
            close_score = close_score.to(device)
            far_score = far_score.to(device)

            enc_loss, dec_loss, triplet_loss = model(self_cont, close_cont, far_cont, close_score, far_score)

            if conf["int_kg"] is True:
                loss = enc_loss + dec_loss + conf["triplet_lambda"] * triplet_loss
            else:
                loss = enc_loss + dec_loss
            loss.backward()                                                                                                
            optimizer.step()
            if conf["lr_decay"] is True:
                exp_lr_scheduler.step() #adjust learning rate
            
            loss_scalar = loss.detach().cpu()
            enc_loss_scalar = enc_loss.detach().cpu()
            dec_loss_scalar = dec_loss.detach().cpu()
            if conf["int_kg"] is True:
                triplet_loss_scalar = triplet_loss.detach().cpu()
            else:
                triplet_loss_scalar = 0

            curr_batch = batch_i + epoch * ttl_batch
            run.add_scalar('Loss/train', loss_scalar, curr_batch)
            run.add_scalar('EncLoss/train', enc_loss_scalar, curr_batch)
            run.add_scalar('DecLoss/train', dec_loss_scalar, curr_batch)
            run.add_scalar('TripletLoss/train', triplet_loss_scalar, curr_batch)

            loss_print += loss_scalar
            enc_loss_print += enc_loss_scalar
            dec_loss_print += dec_loss_scalar
            triplet_loss_print += triplet_loss_scalar
            pbar.set_description('L:{:.6f}, EL:{:.6f}, DL:{:.6f}, ML:{:.6f}'.format(loss_print/(batch_i+1), enc_loss_print/(batch_i+1), dec_loss_print/(batch_i+1), triplet_loss_print/(batch_i+1)))
                                                                                                                           
            if (batch_i + 1) % int(conf["test_interval"]*ttl_batch) == 0:
                mae, mape, all_grd, all_pred, val_mae, val_mape, test_mae, test_mape, test_loss_scalars = evaluate(model, dataset, device, conf)
                run.add_scalar('MAE/all', mae, curr_batch)
                run.add_scalar('MAPE/all', mape, curr_batch)
                run.add_scalar('MAE/val', val_mae, curr_batch)
                run.add_scalar('MAPE/val', val_mape, curr_batch)
                run.add_scalar('MAE/test', test_mae, curr_batch)
                run.add_scalar('MAPE/test', test_mape, curr_batch)
                [loss_test, enc_loss_test, dec_loss_test] = test_loss_scalars
                run.add_scalar('Loss/test', loss_test, curr_batch)
                run.add_scalar('EncLoss/test', enc_loss_test, curr_batch)
                run.add_scalar('DecLoss/test', dec_loss_test, curr_batch)
                if val_mae <= best_val_mae and val_mape <= best_val_mape:
                    best_val_mae = val_mae
                    best_val_mape = val_mape
                    best_test_mae = test_mae
                    best_test_mape = test_mape
                    best_epoch = epoch
                    best_batch = batch_i
                    np.save(log_path + "all_grd", all_grd)
                    np.save(log_path + "all_pred", all_pred)
                    np.save(log_path + "ele_embed", model.ele_embeds.weight.detach().cpu().numpy())
                    # Save the best model parameters in the log path
                    torch.save(model.state_dict(), log_path + "model.stat_dict")
                    if conf["dataset"] == "FIT":
                       np.save(log_path + "city_embed", model.city_embeds.weight.detach().cpu().numpy())
                       np.save(log_path + "gender_embed", model.gender_embeds.weight.detach().cpu().numpy())
                       np.save(log_path + "age_embed", model.age_embeds.weight.detach().cpu().numpy())
                each_print = "MAE: %.6f, MAPE: %.6f, VAL_MAE: %.6f, VAL_MAPE: %.6f, TEST_MAE: %.6f, TEST_MAPE: %.6f" %(mae, mape, val_mae, val_mape, test_mae, test_mape)
                each_print_best = "BEST in epoch %d batch: %d, VAL_MAE: %.6f, VAL_MAPE: %.6f, TEST_MAE: %.6f, TEST_MAPE: %.6f" %(best_epoch, best_batch, best_val_mae, best_val_mape, best_test_mae, best_test_mape)
                print(each_print)
                print(each_print_best)
                result_output.write(each_print + "\n")
                result_output.write(each_print_best + "\n")


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

    train(conf)


if __name__ == "__main__":
    main()
