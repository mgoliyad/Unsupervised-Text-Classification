import argparse
import json
from pathlib import Path
from unsupervised_class import Unsupervised_Classifier

def main():

    '''Reading config file with all input data for ML modeling'''
    condif_filename = Path(Path.cwd(), 'config.json')
    with open(condif_filename) as f:
        config = json.load(f)
           
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory containing texts.', type=Path, default=Path(Path.cwd().parent, 'CLEAN_DATA'))
    parser.add_argument('--printout', help='Do you want to see printouts for each step?', type=bool, default=True)
    args = parser.parse_args()
    printout = args.printout
    data_dir = args.data_dir
    
    '''Running Unclassified text classification'''
    U_classifier = Unsupervised_Classifier(data_dir, config, printout)
    U_classifier.text_clustering()

 
if __name__ == '__main__':
    main()
    print('Press any key to exit the program...')
    input()
