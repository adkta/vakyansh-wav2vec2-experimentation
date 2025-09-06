
 # Usage: python labels.py --jobs 64 --tsv <path to train.tsv>train.tsv --output-dir <destination dir> --output-name test --txt-dir

import argparse
import os
import re
from tqdm import tqdm
from joblib import Parallel, delayed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type = str, help = "TSV file for which labels need to be generated")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--txt-dir")
    parser.add_argument("--jobs", default=-1, type=int, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tsv_file=args.tsv
    output_dir=args.output_dir
    output_name=args.output_name
    transcrip_file = os.path.join( args.txt_dir , 'transcription.txt' )

    #create a wav2trans dictionary
    wav2trans = dict()

    with open(transcrip_file, 'r') as transcrip:
        lines = transcrip.read().strip().split('\n')
        for line in lines:
            if '\t' in line:
                file, trans = line.split("\t")
            else:
                splitted_line = line.split(" ")
                file, trans = splitted_line[0], " ".join(splitted_line[1:])
            wav2trans[file] = trans

        
    with open(tsv_file) as tsv, open(
            os.path.join(output_dir, output_name + ".ltr"), "w",encoding="utf-8"
        ) as ltr_out, open(
            os.path.join(output_dir, output_name + ".wrd"), "w",encoding="utf-8"
        ) as wrd_out:

        root = next(tsv).strip()

        if not args.txt_dir:
            args.txt_dir = root

        #wrd file must be the same order as tsv file
        local_arr = []
        for line in tqdm(tsv):
            if '\t' in line:
                filepath, tot_frames = line.split("\t")
            else:
                splitted_line = line.split(" ")
                filepath, tot_frames = splitted_line[0], " ".join(splitted_line[1:])
            file_id = os.path.splitext(os.path.basename(filepath))[0]
            label = wav2trans[file_id]
            local_arr.append(label);
        
        formatted_text = []
        for text in local_arr:
            local_list = list( text.replace(" ", "|") )
            final_text = " ".join(local_list) + ' |'
            formatted_text.append(final_text)


        wrd_out.writelines("\n".join(local_arr))
        ltr_out.writelines("\n".join(formatted_text))



if __name__ == "__main__":
    main()
