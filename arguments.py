import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='TTP-MORL')
    # GENERAL
    parser.add_argument('--dataset-name',
                        type=str,
                        default="a280-n279",
                        help="dataset's name for real testing")
    parser.add_argument('--title',
                        type=str,
                        default="init-sop",
                        help="title for experiment")
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device to be used cpu or cuda(gpu)')
    
    parser.add_argument('--max-epoch',
                        type=int,
                        default=100000,
                        help='maximum epoch training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='seed for random generator')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        help="dataloader batch size")
    parser.add_argument('--num-training-samples',
                        type=int,
                        default=1000,
                        help="dataloader batch size")
    
    # Agent
    parser.add_argument('--max-grad-norm',
                        type=int,
                        default=1,
                        help="gradient clipping")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    

    #R1_NES
    parser.add_argument('--ld',
                        type=float,
                        default=0,
                        help="lambda e^ld in R1-NES")
    parser.add_argument('--pop-size',
                        type=int,
                        default=1e-4,
                        help="r1 nes pop size in 1 generation")
    parser.add_argument('--negative-hv',
                        type=float,
                        default=-1e-5,
                        help="negative hv penalty")

    return parser
