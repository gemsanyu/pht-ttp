import pathlib

import multiprocessing as mp
import subprocess

def run(dataset_name:str):
    process_args = ["python",
                    "test.py",
                    "--title",
                    "PN-PHN",
                    "--dataset-name",
                    dataset_name,
                    "--device",
                    "cuda"
                ]
    subprocess.run(process_args)

if __name__=="__main__":
    nipc_list = [20,30,50]
    ic_list = [0,1,2]
    cf_list = list(range(1,11))
    dataset_name_list = []
    for nipc in nipc_list:
        for ic in ic_list:
            for cf in cf_list:
                dataset_name = "nn_50_nipc_"+str(nipc)+"_ic_"+str(ic)+"_cf_"+str(cf)+"_0"
                dataset_name_list += [dataset_name]
    config_list = [dataset_name_list[i] for i in range(len(dataset_name_list))]
    for dataset_name in config_list:
        run(dataset_name)
