import os
import json
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')


class TrendTrainDataset(Dataset):
    def __init__(self, conf, trends, ori_trends, ids, norm, shifts, grps, eles, dist_mat=None, city=None, gender=None, age=None):
        self.conf = conf
        self.trends = trends
        self.ori_trends = ori_trends
        self.ids = ids
        self.norm = norm
        self.shifts = shifts
        self.grps = grps
        self.eles = eles
        if dist_mat is not None:
            self.dist_mat = dist_mat
            self.neibors_mat = np.argsort(self.dist_mat, axis=1)

        self.city = city
        self.gender = gender
        self.age = age
        self.seq_num = self.trends.shape[0]

        self.ttl_len = self.conf["input_len"] + self.conf["output_len"]


    def __len__(self):
        return len(self.ids)


    def return_one(self, idx):
        seq_id = self.ids[idx]
        seq_shift = self.shifts[idx]

        each_trend = self.trends[seq_id, seq_shift:seq_shift+self.ttl_len]
        each_input = torch.from_numpy(each_trend[:self.conf["input_len"]]).float()
        each_output = torch.from_numpy(each_trend[self.conf["input_len"]:]).float()
        each_trend = [each_input, each_output]

        each_ori_trend = self.ori_trends[seq_id, seq_shift:seq_shift+self.ttl_len]
        each_ori_input = torch.from_numpy(each_ori_trend[:self.conf["input_len"]]).float()
        each_ori_output = torch.from_numpy(each_ori_trend[self.conf["input_len"]:]).float()
        each_ori_trend = [each_ori_input, each_ori_output]

        each_grp = torch.LongTensor([self.grps[idx]])
        each_ele = torch.LongTensor([self.eles[idx]])
        each_norm = torch.FloatTensor(self.norm[idx])

        metadata = [seq_id, each_grp, each_ele, each_norm]
        if self.conf["dataset"] == "FIT":
            each_city = torch.LongTensor([self.city[idx]])
            each_gender = torch.LongTensor([self.gender[idx]])
            each_age = torch.LongTensor([self.age[idx]])
            metadata += [each_city, each_gender, each_age]

        return [each_trend, each_ori_trend, metadata], seq_id


    def __getitem__(self, idx):
        self_cont, seq_id = self.return_one(idx)
        neibors = self.neibors_mat[seq_id].tolist()

        # sampling methods, which can do hard negtive sample
        start_point = random.sample([x for x in range(self.seq_num-self.conf["sample_range"])], 1)[0]
        end_point = start_point + self.conf["sample_range"]
        sample_neibors = random.sample(neibors[start_point:end_point], 3)
        filtered_neibors = []
        for x in sample_neibors:
            if x != seq_id:
                filtered_neibors.append(x)
        filtered_neibors = sorted(filtered_neibors) 

        close_item = filtered_neibors[0]
        ori_close_score = self.dist_mat[seq_id][close_item]
        close_score = torch.FloatTensor([ori_close_score])
        close_item_new = idx - (idx % self.seq_num) + close_item
        close_cont, _ = self.return_one(close_item_new)

        far_item = filtered_neibors[1] 
        ori_far_score = self.dist_mat[seq_id][far_item]
        far_score = torch.FloatTensor([ori_far_score])
        far_item_new = idx - (idx % self.seq_num) + far_item
        far_cont, _ = self.return_one(far_item_new)

        if far_score >= close_score:
            return self_cont, close_cont, far_cont, close_score, far_score
        else:
            return self_cont, far_cont, close_cont, far_score, close_score


class TrendTestDataset(TrendTrainDataset):
    def __init__(self, conf, trends, ori_trends, ids, norm, shifts, grps, eles, city=None, gender=None, age=None):
        super(TrendTestDataset, self).__init__(conf, trends, ori_trends, ids, norm, shifts, grps, eles, city=city, gender=gender, age=age)


    def __getitem__(self, idx):
        self_cont, seq_id = self.return_one(idx)
        return self_cont


def dump_geostyle_taxonomy(attributes, categories):
    for i in range(len(attributes)):
        attr = attributes[i]
        print(attr)
        for j in range(len(categories[i])):
            value = categories[i][j]
            print("\t%s" %(value))
        print("\n")


def pre_preprocess_geostyle_data(conf, eps=0.01):
    input_file = conf["raw_data_path"]
    if not isfile(input_file):
        print("FileNotFound: Download 'metadata.pkl' and place it in the directory: %s" %(conf["raw_data_path"]))
        exit()
    with open(input_file, 'rb') as fo:
        data = pickle.load(fo)
    cities = sorted(data['cities'].keys())
    attributes = data['attributes']
    categories = data['categories']

    dump_geostyle_taxonomy(attributes, categories)

    new_data, new_data_normed, new_data_norm = {}, {}, {}

    first_iter = True
    for i in range(len(attributes)):
        attr = attributes[i]
        for j in range(len(categories[i])):
            value = categories[i][j]
            attr_value = "__".join([attr, value])
            for cind, city in enumerate(cities):
                pos_tot = []
                datum = data['classifications'][city]
                # remove weeks with small amount of data from start and end
                if first_iter:
                    weeks = sorted(datum.keys())
                    weeks = weeks[5:-5]
                    first_iter = False
                for week in weeks:
                    week_num = int(week.split("_")[1]) - 1
                    pos_tot.append([np.sum(datum[week][:, i] == j), datum[week].shape[0], week_num])

                timestep, trend = [], []
                for k in range(len(pos_tot)):
                    if pos_tot[k][0] == 0:
                        pos_tot[k][0] = 1
                    elif pos_tot[k][0] == pos_tot[k][1]:
                        pos_tot[k][0] = pos_tot[k][0]-1
                    #timestep.append(int(pos_tot[k][2]/2))
                    timestep.append(int(pos_tot[k][2]))
                    trend.append(pos_tot[k][0]/float(pos_tot[k][1]))

                max_v = max(trend)
                min_v = min(trend)
                normed_trend = [max((x-min_v)/(max_v-min_v), eps) for x in trend]
                normed_res, res = [], []
                for time_s, trend_v in zip(timestep, trend):
                    res.append([time_s, trend_v])
                for time_s, trend_v in zip(timestep, normed_trend):
                    normed_res.append([time_s, trend_v])

                if city not in new_data:
                    new_data[city] = {attr_value: res}
                    new_data_normed[city] = {attr_value: normed_res}
                    new_data_norm[city] = {attr_value: [min_v, max_v, eps]}
                else:
                    new_data[city][attr_value] = res
                    new_data_normed[city][attr_value] = normed_res
                    new_data_norm[city][attr_value] = [min_v, max_v, eps]

    json.dump(new_data, open(conf["ori_data_path"], "w"))
    json.dump(new_data_normed, open(conf["normed_data_path"], "w"))
    json.dump(new_data_norm, open(conf["norm_path"], "w"))


class TrendData(Dataset):
    def __init__(self, conf):
        conf["dist_mat_path"] = "%s/dist_mat_%d.npy" %(conf["dist_mat_path"], conf["output_len"])
        self.conf = conf
        if self.conf["dataset"] == "FIT":
            self.adj, self.grp_id_map, self.ele_id_map = self.load_fit_affiliation_adj()
            self.ori_trends, self.trends, self.trend_norm, self.grp_ids, self.ele_ids, self.time_num, self.city_ids, self.gender_ids, self.age_ids = self.get_ori_data()
            train_ids, train_norm, train_shift, train_grps, train_eles, test_ids, test_norm, test_shift, test_grps, test_eles, train_city, train_gender, train_age, test_city, test_gender, test_age = self.preprocess_data(self.trends, self.trend_norm, self.grp_ids, self.ele_ids, city_ids=self.city_ids, gender_ids=self.gender_ids, age_ids=self.age_ids)
            self.dist_mat = self.load_dist_mat(self.trends)
            self.train_set = TrendTrainDataset(conf, self.trends, self.ori_trends, train_ids, train_norm, train_shift, train_grps, train_eles, dist_mat=self.dist_mat, city=train_city, gender=train_gender, age=train_age)
            self.test_set = TrendTestDataset(conf, self.trends, self.ori_trends, test_ids, test_norm, test_shift, test_grps, test_eles, city=test_city, gender=test_gender, age=test_age)
        else:
            if not os.path.exists(self.conf["ori_data_path"]):
                pre_preprocess_geostyle_data(self.conf)
            self.adj = None
            self.ori_trends, self.trends, self.trend_norm, self.grp_ids, self.ele_ids, self.time_num = self.get_ori_data()
            train_ids, train_norm, train_shift, train_grps, train_eles, test_ids, test_norm, test_shift, test_grps, test_eles = self.preprocess_data(self.trends, self.trend_norm, self.grp_ids, self.ele_ids)
            self.dist_mat = self.load_dist_mat(self.trends)
            self.train_set = TrendTrainDataset(conf, self.trends, self.ori_trends, train_ids, train_norm, train_shift, train_grps, train_eles, dist_mat=self.dist_mat)
            self.test_set = TrendTestDataset(conf, self.trends, self.ori_trends, test_ids, test_norm, test_shift, test_grps, test_eles)

        self.train_loader = DataLoader(self.train_set, batch_size=self.conf["batch_size"], shuffle=True, num_workers=self.conf["num_workers"])
        self.test_loader = DataLoader(self.test_set, batch_size=self.conf["batch_size"], shuffle=False, num_workers=self.conf["num_workers"])


    def load_fit_affiliation_adj(self):
        ele_id, grp_id = 0, 0
        ele_id_map, grp_id_map = {}, {}
        ori_adj = json.load(open(self.conf["group_element_adj_path"]))
        for group, adj_data in ori_adj.items():
            grp_id_map[group] = grp_id
            grp_id += 1
            for k, res in adj_data.items():
                if k not in ele_id_map:
                    ele_id_map[k] = ele_id
                    ele_id += 1
                for v, score in res.items():
                    if v not in ele_id_map:
                        ele_id_map[v] = ele_id
                        ele_id += 1

        single_adj = json.load(open(self.conf["element_adj_path"]))
        adj = np.zeros((len(ele_id_map), len(ele_id_map)), dtype=float)
        for k, res in single_adj.items():
            k_id = ele_id_map[k]
            for v, score in res.items():
                v_id = ele_id_map[v]
                adj[v_id][k_id] = score
        return adj, grp_id_map, ele_id_map 


    def load_dist_mat(self, trends):
        if os.path.exists(self.conf["dist_mat_path"]):
            dist_mat = np.load(self.conf["dist_mat_path"])
        else:
            all_train = trends[:, :-self.conf["output_len"], 1] # all_train: [n_len, seq_len]
            n_len = all_train.shape[0]
            dist_mat = []
            for a_id, a in enumerate(all_train):
                a_broad = np.repeat(a[np.newaxis, :], n_len, axis=0) # [n_len, seq_len]
                mape = np.mean(np.abs(a_broad - all_train) / all_train, axis=-1)*100 # [n_len]
                dist_mat.append(mape)
            dist_mat = np.stack(dist_mat, axis=0) # [n_len, n_len]
            np.save(self.conf["dist_mat_path"], dist_mat)

        return dist_mat


    def get_ori_data(self):
        ori_data = json.load(open(self.conf["ori_data_path"]))
        normed_data = json.load(open(self.conf["normed_data_path"]))
        data_norm = json.load(open(self.conf["norm_path"]))
        ori_seq_len = self.conf["seq_len"]

        if self.conf["dataset"] == "FIT":
            city_id_map, gender_id_map, age_id_map = {}, {}, {"null": 0}
            city_ids, gender_ids, age_ids = [], [], []

        if self.conf["dataset"] == "FIT":
            grp_id_map = self.grp_id_map
            ele_id_map = self.ele_id_map
        else:
            # generate the ele_id_map. Note we separate this part just to make sure the fashion element id keeps the same between src and trg datasets
            grp_set, grp_id_map = set(), {}
            ele_set, ele_id_map = set(), {}
            for group_name, res in normed_data.items():
                if group_name not in grp_set:
                    grp_set.add(group_name)
                for fashion_ele, seq in res.items():
                    ele_set.add(fashion_ele)
            for grp in sorted(grp_set):
                grp_id_map[grp] = len(grp_id_map)
            for ele in sorted(ele_set):
                ele_id_map[ele] = len(ele_id_map)

        ori_trends, trends, trends_info, grp_ids, ele_ids, trend_norm = [], [], [], [], [], []
        time_num = 0

        for group_name, res in sorted(normed_data.items(), key=lambda i: i[0]):
            if self.conf["dataset"] == "FIT":
                comps = group_name.split("__")
                city_id, gender_id, age_id = 0, 0, 0
                for each in comps:
                    if "city:" in each:
                        city = each.split(":")[1]
                        if each not in city_id_map:
                            city_id_map[each] = len(city_id_map)
                        city_id = city_id_map[each]
                    if "gender:" in each:
                        if each not in gender_id_map:
                            gender_id_map[each] = len(gender_id_map)
                        gender_id = gender_id_map[each]
                    if "age:" in each:
                        if each not in age_id_map:
                            age_id_map[each] = len(age_id_map)
                        age_id = age_id_map[each]

            curr_grp_id = grp_id_map[group_name]

            #for fashion_ele, seq in sorted(res.items(), key=lambda i: i[0]):
            for fashion_ele, seq in res.items():
                seq_name = "__".join([group_name, fashion_ele])

                time_seq = [x[0] for x in seq]
                time_val_seq = [x[1] for x in seq]
                each_time_num = max(time_seq) + 1
                if each_time_num > time_num:
                    time_num = each_time_num

                curr_ele_id = ele_id_map[fashion_ele]

                trends.append(seq)
                ori_trends.append(ori_data[group_name][fashion_ele])
                trends_info.append(seq_name)

                norm = data_norm[group_name][fashion_ele]
                norm = [float(x) for x in norm]
                trend_norm.append(norm)

                grp_ids.append(curr_grp_id)
                ele_ids.append(curr_ele_id)
                if self.conf["dataset"] == "FIT":
                    city_ids.append(city_id)
                    gender_ids.append(gender_id)
                    age_ids.append(age_id)

        trends = np.array(trends)
        ori_trends = np.array(ori_trends)
        trend_norm = np.array(trend_norm)
        grp_ids = np.array(grp_ids)
        ele_ids = np.array(ele_ids)

        json.dump(grp_id_map, open("./dataset/%s/grp_id_map.json" %(self.conf["dataset"]), "w"), indent=4)
        json.dump(ele_id_map, open("./dataset/%s/ele_id_map.json" %(self.conf["dataset"]), "w"), indent=4)
        if self.conf["dataset"] == "FIT":
            json.dump(city_id_map, open("./dataset/%s/city_id_map.json" %(self.conf["dataset"]), "w"), indent=4)
            json.dump(gender_id_map, open("./dataset/%s/gender_id_map.json" %(self.conf["dataset"]), "w"), indent=4)
            json.dump(age_id_map, open("./dataset/%s/age_id_map.json" %(self.conf["dataset"]), "w"), indent=4)

        self.trends_info = trends_info
        self.grp_id_map = grp_id_map
        self.ele_id_map = ele_id_map
        self.seq_num = trends.shape[0]
        if self.conf["dataset"] == "FIT":
            self.city_id_map = city_id_map
            self.gender_id_map = gender_id_map
            self.age_id_map = age_id_map

        res = [ori_trends, trends, trend_norm, grp_ids, ele_ids, time_num]
        if self.conf["dataset"] == "FIT":
            res += [np.array(city_ids), np.array(gender_ids), np.array(age_ids)]

        return res


    def preprocess_data(self, trends, trend_norm, grp_ids, ele_ids, city_ids=None, gender_ids=None, age_ids=None):
        ori_seq_len = trends.shape[1]
        ttl_len = self.conf["input_len"] + self.conf["output_len"]
        output_len = self.conf["output_len"]
        assert ori_seq_len > ttl_len + output_len
        train_ids, train_shift, train_grps, train_eles, train_norm = [], [], [], [], []
        if self.conf["dataset"] == "FIT":
            train_city, train_gender, train_age = [], [], []
        for i in range(ori_seq_len-ttl_len-output_len): # the last one for testing
            train_ids.append(np.array([j for j in range(trends.shape[0])]))
            train_shift.append(np.array([i]*trends.shape[0]))
            train_norm.append(trend_norm)
            train_grps.append(grp_ids)
            train_eles.append(ele_ids)
            if self.conf["dataset"] == "FIT":
                train_city.append(city_ids)
                train_gender.append(gender_ids)
                train_age.append(age_ids)
        train_ids = np.concatenate(train_ids, axis=0)
        train_shift = np.concatenate(train_shift, axis=0)
        train_norm = np.concatenate(train_norm, axis=0)
        train_grps = np.concatenate(train_grps, axis=0)
        train_eles = np.concatenate(train_eles, axis=0)
        if self.conf["dataset"] == "FIT":
            train_city = np.concatenate(train_city, axis=0)
            train_gender = np.concatenate(train_gender, axis=0)
            train_age = np.concatenate(train_age, axis=0)

        test_ids = np.array([j for j in range(trends.shape[0])])
        test_shift = np.array([ori_seq_len-ttl_len] * trends.shape[0])
        test_norm = trend_norm
        test_grps = grp_ids
        test_eles = ele_ids
        if self.conf["dataset"] == "FIT":
            test_city = city_ids
            test_gender = gender_ids
            test_age = age_ids

        #print("train data: ", train_ids.shape)
        #print("test data: ", test_ids.shape)
        if self.conf["dataset"] == "FIT":
            return train_ids, train_norm, train_shift, train_grps, train_eles, test_ids, test_norm, test_shift, test_grps, test_eles, train_city, train_gender, train_age, test_city, test_gender, test_age
        else:
            return train_ids, train_norm, train_shift, train_grps, train_eles, test_ids, test_norm, test_shift, test_grps, test_eles
