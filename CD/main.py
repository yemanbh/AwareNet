from training import run
from get_train_config import training_configs
# -o D:\Projects\Myeloma_BM_mapping\Data\20200401\models -d D:\Projects\Myeloma_BM_mapping\Data\20200401\combined_polyscope-rectangles\patch -n 4

if __name__ == '__main__':
    configs, args = training_configs()
    
    if args.cluster:
        print('*' * 50)
        print('job started running on a cluster')
        print('*' * 50)
        
        print('*' * 50)
        print('running configuration number: {}'.format(args.config_id))
        print('*' * 50)
        if args.config_id == '-1':
            for n in range(1, len(configs) + 1):
               print('*' * 50)
               print('running configuration number: {}'.format(n))
               print('*' * 50)
               run(**configs[str(n)], all_params=configs[str(n)])
            else:
                run(**configs[args.config_id])
    else:

        for n in range(1, len(configs) + 1):
            
            print('*'*50)
            print('running configuration number: {}'.format(n))
            print('*' * 50)
            run(**configs[str(n)], all_params=configs[str(n)])
