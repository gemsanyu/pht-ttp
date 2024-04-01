import pathlib

import multiprocessing as mp
import subprocess

from tracer import trace_and_save_model

TITLE = "AM-DRLMOA"

def run(dataset_name:str):
    # python test.py --title AM-DRLMOA --total-weight 100 --dataset-name ch150_n1490_bounded-strongly-corr_01 &
    process_args = ["python",
                    "test_mp.py",
                    "--title",
                    TITLE,
                    "--total-weight",
                    str(100),
                    "--dataset-name",
                    dataset_name,
                    "--device",
                    "cuda"
                ]
    subprocess.run(process_args)

if __name__=="__main__":
    # prepare models first
    for i in range(1,101):
        print("tracing",i,"-th model")
    #     trace_and_save_model(TITLE,i,100)
    
    nipc_list = [20,30,50]
    ic_list = [0,1,2]
    cf_list = list(range(1,11))
    dataset_name_list = []
    for nipc in nipc_list:
        for ic in ic_list:
            for cf in cf_list:
                dataset_name = "nn_101_nipc_"+str(nipc)+"_ic_"+str(ic)+"_cf_"+str(cf)+"_0"
                dataset_name_list += [dataset_name]
    config_list = [dataset_name_list[i] for i in range(len(dataset_name_list))]
    for it, dataset_name in enumerate(config_list):
        print("---------it:",it)
        run(dataset_name)
