import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='graph metadata directory')
    parser.add_argument('-o', dest='output_dir', help='directory to save kernel')
    parser.add_argument('--cluster', dest='cluster', help='using cluster or local machine', default=False,
                        action='store_true')
    
    args = parser.parse_args()
    return args
