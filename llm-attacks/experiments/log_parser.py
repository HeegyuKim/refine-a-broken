import json
import pandas as pd
from datasets import load_dataset
import fire
import os
import fnmatch
import numpy as np
from glob import glob
from tqdm import tqdm

def find_and_sort_files(directory, pattern):
    # 특정 패턴을 만족하는 파일들을 찾습니다.
    files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
    # 찾은 파일들을 정렬합니다.
    sorted_files = sorted(files)
    return sorted_files


def make_best_control(data ,stop_on_success=True):
  results = {}
  results['control'] = []
  results['goal'] =[]
  results['target'] = []
  results['train_passed']=[]
  results['passed'] = []
  results['loss'] = []


  pass_rate = [x['n_passed'][1]/x['total'][1] if x['n_passed'][1] >0 else 0 for x in  data['tests'] ]
  logstep = data['params']['n_steps']//data['params']['test_steps']*1+1


  for i in range(len(data['tests'])//logstep):
    results['goal'].append(data['params']['goals'][i])
    results['target'].append(data['params']['targets'][i])
    if stop_on_success:
      start = np.where(np.array(data['losses']) ==1000000.0)[0][i]

      if i+1 == len(data['params']['goals']):
        end = len(data['losses'])-1
      else:
        end =  np.where(np.array(data['losses']) ==1000000.0)[0][i+1] -1

    else:
      start = i*logstep
      end = start + logstep
    success_index =[]
    for i, x in enumerate(data['tests'][start:end+1]):
        if x['n_passed'][0] == 1:
            success_index.append(i)
    if len(success_index)>0:
      min_loss = min([ data['losses'][start:end+1][x] for x in success_index])
    else:
      min_loss = min(data['losses'][start:end+1])
    min_loss_index = data['losses'][start:end+1].index(min_loss)
    results['train_passed'].append(data['tests'][start:end+1][min_loss_index]['n_passed'][0])
    results['control'].append(data['controls'][start:end+1][min_loss_index])
    results['passed'].append(pass_rate[start:end+1][min_loss_index])
    results['loss'].append(min_loss)



  return results

def make_best_contorl_multiple(data):
    results ={}
    results['control'] = []
    results['train_passed'] = []
    results['test_passed'] = []
    results['loss'] = []

    results['train_passed'] = [x['n_passed'][0]/x['total'][0] for x in  data['tests'] ]
    results['test_passed'] = [x['n_passed'][1]/x['total'][1] for x in  data['tests'] ]
    results['loss'] = data['losses']
    results['control'] = data['controls']
    df = pd.DataFrame(results)
    df=df.sort_values(by='test_passed', ascending=False).reset_index(drop=True)

    return df

def make_best_attack_multiple(data, advbench):
    results ={}
    results['goal'] = []
    results['target'] = []
    results['control'] = []

    results['goal'] =advbench['goal'][:100]
    results['control'] = [ data['control'][0] for x in range(100) ]
    results['target'] = advbench['target'][:100]


    return results


def main(
  file_path:str = None,
  attack_method:str = 'autodan',
  attack_data: str = 'individual',
  model_id:str = 'vicuna7b',
  output_path:str = None,
):
  print(f'Start pasrer {model_id} {attack_method} {attack_data}...')
  #individual
  if attack_data == 'individual':
    results = find_and_sort_files('results', f'individual_behaviors_{model_id}*.json')
    if not results:
      print(f"There are no files in results")
      return
    l = []
    for result in tqdm(results):
        with open(f'results/{result}', 'r') as f:
            data = json.load(f)
        l.append(data)
    data_frames = []
    for data in l:
        data_frames.append(pd.DataFrame(make_best_control(data)))

    new_data = pd.concat(data_frames, axis=0).reset_index(drop=True)
    print(f" {len(new_data)} goals parser clear")
  else:
    filelist= list(glob(f"results/{file_path}*.json", recursive=True))
    if len(filelist)<0:
      print(f"There are no files in results")
      return
    file_path = filelist[0]

    if file_path:
      with open(f'{file_path}','r') as f:
        data = json.load(f)
    #multiple
    advbench = pd.read_csv('../data/advbench/harmful_behaviors.csv')
    results = make_best_contorl_multiple(data)
    print(f" {len(results)} goals parser clear")
    results = make_best_attack_multiple(results, advbench)
    new_data = pd.DataFrame(results)

  if output_path:
    new_data.to_json(f'{output_path}/{model_id}7b-{attack_method}-{attack_data}.json', orient='records', lines=True)
  else:
    new_data.to_json(f'logs/{model_id}7b-{attack_method}-{attack_data}.json', orient='records', lines=True)



if __name__ == '__main__':
    fire.Fire(main)




