import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import argparse
import os, json
from generative_agents.memory_utils import get_session_facts
from global_methods import set_openai_key
from task_eval.rag_utils import get_embeddings
import pickle

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--use-date', action="store_true")
    parser.add_argument('--overwrite', action="store_true", help="set flag to overwrite existing outputs")
    parser.add_argument('--retriever', type=str, default="dragon")

    args = parser.parse_args()
    return args


def main():

    
    # set openai API key (only needed for some components that call OpenAI)
    set_openai_key()

    # get arguments
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    dataset_prefix = os.path.splitext(os.path.split(args.data_file)[-1])[0]

    for data in samples:

        observations = []
        date_times = []
        context_ids = []

        # check for existing output
        if data['sample_id'] in out_samples:
            output = out_samples['sample_id']
        else:
            output = {'sample_id': data['sample_id']}

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if 'session' in k and 'date_time' not in k]
        for i in tqdm(range(min(session_nums), max(session_nums) + 1), desc='Generating observations for %s' % data['sample_id']):

            # get the observations
            session_obs_key = 'session_%s_observation' % i
            # prefer observations already present in the data file (no LLM call)
            if ('observation' in data) and (session_obs_key in data['observation']):
                facts = data['observation'][session_obs_key]
                output[session_obs_key] = facts
            else:
                if session_obs_key not in output or args.overwrite:
                    facts = get_session_facts(args, data['conversation'], data['conversation'], i, return_embeddings=False)
                    output[session_obs_key] = facts
                else:
                    facts = output[session_obs_key]

            date_time = data['conversation']['session_%s_date_time' % i]
            for k, v in facts.items():
                for fact, dia_id in v:
                    observations.append(fact)
                    context_ids.append(dia_id)
                    date_times.append(date_time)

            # save intermittently to prevent loss of data
            out_samples[output['sample_id']] = output.copy()
            with open(args.out_file, 'w') as f:
                json.dump(list(out_samples.values()), f, indent=2)

        # use date + observation as context when getting embeddings, if flag is set to True
        if args.use_date:
            inputs = ['. '.join([k, v]) for k, v in zip(date_times, observations)]
            embeddings = get_embeddings(args.retriever, inputs, 'context')
        else:
            embeddings = get_embeddings(args.retriever, observations, 'context')
        
        assert embeddings.shape[0] == len(observations)

        # save everything to a pickle file; separate pickle file for each sample
        database = {'embeddings': embeddings,
                            'date_time': date_times,
                            'dia_id': context_ids,
                            'context': observations}

        # ensure emb_dir is set; default to directory of out_file if empty
        emb_dir = args.emb_dir if args.emb_dir else os.path.dirname(args.out_file)
        os.makedirs(emb_dir, exist_ok=True)
        pkl_path = os.path.join(emb_dir, '%s_observation_%s.pkl' % (dataset_prefix, data['sample_id']))
        with open(pkl_path, 'wb') as f:
            pickle.dump(database, f)

        out_samples[output['sample_id']] = output.copy()
    
    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)


main()