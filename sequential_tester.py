import pathlib

import multiprocessing as mp
import subprocess

def run(dataset_name:str):
    # python test.py --title AM-DRLMOA --total-weight 100 --dataset-name ch150_n1490_bounded-strongly-corr_01 &
    process_args = ["python",
                    "test.py",
                    "--title",
                    "AM-DRLMOA",
                    "--total-weight",
                    str(100),
                    "--dataset-name",
                    dataset_name,
                    "--device",
                    "cuda"
                ]
    subprocess.run(process_args)

if __name__=="__main__":
    
    graph_name_list = [
                    "eil76",
                    "ch150",
                    # "ch130",
                    "d657",
                    "eil51",
                    "eil101",
                    # "gil262",
                    
                    "kroA100",
                    "kroA150",
                    # "kroA200",
                    # "kroB100",
                    "kroB150",
                    
                    # "kroB200",
                    # "kroC100",
                    # "kroD100",
                    # "kroE100"
                    ]
    num_nodes_list = []
    for graph_name in graph_name_list:
        num_nodes = 0
        for c in graph_name:
            if c in "0123456789":
                num_nodes = num_nodes*10+int(c)
        num_nodes_list += [num_nodes]
    num_items_list = [1,3,5,10]
    instance_type_list = [
        "bounded-strongly-corr",
        # "uncorr",
        # "uncorr-similar-weights"
        ]
    dataset_name_list = []
    for gi, graph_name in enumerate(graph_name_list):
        num_nodes = num_nodes_list[gi]
        for num_items in num_items_list:
            total_num_items = num_items*(num_nodes-1)
            for instance_type in instance_type_list:
                for idx in ["01","02","03","04","05","06","07","08","09","10"]:    
                    dataset_name = graph_name+"_n"+str(total_num_items)+"_"+instance_type+"_"+idx
                    dataset_name_list += [dataset_name]
    config_list = [dataset_name_list[i] for i in range(len(dataset_name_list))]
    for dataset_name in config_list:
        run(dataset_name)
