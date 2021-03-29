import os
import yaml
from test import test


def get_test_conf():
    test_conf = {
        "FIT": {
            "Half Year": {
                "KERN": {
                    "output_len": 12,
                    "ext_kg": True,
                    "int_kg": True,
                    "triplet_lambda": 0.002,
                    "sample_range": 500
                },
                "KERN-E": {
                    "output_len": 12,
                    "ext_kg": False,
                    "int_kg": True,
                    "triplet_lambda": 0.002,
                    "sample_range": 500
                },
                "KERN-I": {
                    "output_len": 12,
                    "ext_kg": True,
                    "int_kg": False
                },
                "KERN-IE": {
                    "output_len": 12,
                    "ext_kg": False,
                    "int_kg": False
                }
            },
            "One Year": {
                "KERN": {
                    "output_len": 24,
                    "ext_kg": True,
                    "int_kg": True,
                    "triplet_lambda": 0.001,
                    "sample_range": 500
                },
                "KERN-E": {
                    "output_len": 24,
                    "ext_kg": False,
                    "int_kg": True,
                    "triplet_lambda": 0.001,
                    "sample_range": 500
                },
                "KERN-I": {
                    "output_len": 24,
                    "ext_kg": True,
                    "int_kg": False,
                },
                "KERN-IE": {
                    "output_len": 24,
                    "ext_kg": False,
                    "int_kg": False
                }
            },
        },
        "GeoStyle": {
            "Half Year": {
                "KERN": {
                    "output_len": 26,
                    "ext_kg": False,
                    "int_kg": True,
                    "triplet_lambda": 0.002,
                    "sample_range": 500
                },
                "KERN-I": {
                    "output_len": 26,
                    "ext_kg": False,
                    "int_kg": False
                },
                "KERN-IE": {
                    "output_len": 26,
                    "ext_kg": False,
                    "int_kg": False
                }
            }
        }
    }

    return test_conf



def get_test_res(global_conf, test_conf):
    result = {}
    for dataset, data_res in test_conf.items():
        if dataset not in result:
            result[dataset] = {}
        for pred_len, model_res in data_res.items():
            if pred_len not in result[dataset]:
                result[dataset][pred_len] = {}
            for model_name, configs in model_res.items():
                print("Generating testing results for %s:%s:%s" %(dataset, pred_len, model_name))
                conf = dict(global_conf[dataset])
                conf["dataset"] = dataset
                for k, v in configs.items():
                    conf[k] = v
                test_mae, test_mape = test(conf, _print=False)
                result[dataset][pred_len][model_name] = {"mae": float(test_mae), "mape": float(test_mape)}

    return result


def print_test_res(res):
    print("\nThe reproduce result of Table 3 in the companion paper:")
    print("Dataset:\tGeoStyle\t|\tFIT")
    print("PredLen:\tHalf Year\t|\tHalf Year\t|\tOne Year")
    print("Metrics:\tMAE\tMAPE\t|\tMAE\tMAPE\t|\tMAE\tMAPE")
    print("Results:\t%.4f\t%.2f\t|\t%.3f\t%.2f\t|\t%.3f\t%.2f" %(
        res["GeoStyle"]["Half Year"]["KERN"]["mae"],
        res["GeoStyle"]["Half Year"]["KERN"]["mape"],
        res["FIT"]["Half Year"]["KERN"]["mae"],
        res["FIT"]["Half Year"]["KERN"]["mape"],
        res["FIT"]["One Year"]["KERN"]["mae"],
        res["FIT"]["One Year"]["KERN"]["mape"]
    ))

    print("\nThe reproduce result of Table 4 in the companion paper (MAPE):")
    print("Dataset:\tGeoStyle\t|\tFIT")
    print("PredLen:\tHalf Year\t|\tHalf Year\t|\tOne Year")

    for model_name in ["KERN-IE", "KERN-E", "KERN-I", "KERN"]:
        if model_name == "KERN-E":
            print("%s:   \t    -   \t|\t%.4f    \t|\t%.4f" %(
                model_name,
                res["FIT"]["Half Year"][model_name]["mae"],
                res["FIT"]["One Year"][model_name]["mae"],
            ))
        else:
            print("%s:   \t  %.4f  \t|\t%.4f   \t|\t%.4f" %(
                model_name,
                res["GeoStyle"]["Half Year"][model_name]["mae"],
                res["FIT"]["Half Year"][model_name]["mae"],
                res["FIT"]["One Year"][model_name]["mae"],
            ))


def main():
    global_conf = yaml.safe_load(open("./config.yaml"))
    test_conf = get_test_conf()

    test_result = get_test_res(global_conf, test_conf)

    print_test_res(test_result)


if __name__ == "__main__":
    main()
