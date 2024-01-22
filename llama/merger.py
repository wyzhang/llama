import argparse
import json
import os
import shutil
import torch
import glob
from typing import List, Dict

_MODEL_SIZE_TO_NUM_SHARDS_MAP = {
    "7B": 1,
    "13B": 2,
    "70B": 8,
}

_LAYER_NAME_TO_SHARDING_TYPE_MAP = {
    'tok_embeddings.weight': 'ParallelEmbedding',
    'norm.weight': None,
    'output.weight': None,
    # layer start
    'attention.wq.weight':'ColumnParallelLinear',
    'attention.wk.weight': 'ColumnParallelLinear',
    'attention.wv.weight': 'ColumnParallelLinear',
    'attention.wo.weight':'RowParallelLinear',
    'feed_forward.w1.weight':'ColumnParallelLinear',
    'feed_forward.w2.weight':'RowParallelLinear',
    'feed_forward.w3.weight':'ColumnParallelLinear',
    'attention_norm.weight': None,
    'ffn_norm.weight': None,
    # layer end
    'rope.freqs': None,
}

def create_dir(target_dir:str)->None:
  if os.path.exists(target_dir):
    # If it exists, remove it and all its contents
    shutil.rmtree(target_dir)
  os.makedirs(target_dir)

def checkpoints_have_same_weight_keys(checkpoint_list:List[Dict[str, torch.Tensor]]):
  if (not checkpoint_list) or len(checkpoint_list) <= 1:
    return True
  for m in checkpoint_list[1:]:
    if set(checkpoint_list[0].keys()) != set(m.keys()):
      return False
  return True

def tensors_have_same_shape(tensors):
  if (not tensors) or len(tensors) <= 1:
    return True
  # Iterate through the remaining tensors in the list
  for t in tensors[1:]:
    if t.shape != tensors[0].shape:
      return False
  return True

def read_json(path):
  with open(path, "r") as f:
    return json.load(f)

def write_json(text, path):
  with open(path, "w") as f:
    json.dump(text, f)

def merge_weights(
    input_ckpt_dir: str,
    output_ckpt_dir: str,
    model_size: str,
) -> None:
  """Merge sharded checkpoint files into a single checkpoint file

  Args:
    input_ckpt_dir: The path to the input directory.
    model_size:
    output_ckpt_dir:

  Returns:
    None
  """
  params = read_json(os.path.join(input_ckpt_dir, "params.json"))
  n_layers = params['n_layers']
  print(f'Loaded params. n_layers={n_layers}')

  print(f'Loading checkpoint files from {input_ckpt_dir}')
  checkpoints = [
      torch.load(path, map_location=torch.device('cpu'))
      for path in glob.glob(os.path.join(input_ckpt_dir, '*.pth'))
  ]
  assert len(checkpoints) > 0, f'No *.pth found in input dir {input_ckpt_dir}'

  print(f'Merging weights')
  state_dict = {}
  assert checkpoints_have_same_weight_keys(checkpoints)
  weight_keys = checkpoints[0].keys()
  for key in weight_keys:
    print(f'Merging weights for {key}')
    tensors: List[torch.Tensor]= [c[key] for c in checkpoints]
    assert(tensors_have_same_shape(tensors))
    for pattern, kind in _LAYER_NAME_TO_SHARDING_TYPE_MAP.iterms():
      if not key.endswith(pattern):
        continue
      assert 'key' not in state_dict
      with torch.no_grad():
        if kind == 'ColumnParallelLinear':
          assert 'key' not in state_dict
          state_dict[key] = torch.cat(tensors, 0)
        elif kind in ('ParallelEmbedding', 'RowParallelLinear'):
          assert 'key' not in state_dict
          state_dict[key] = torch.cat(tensors, 1)
        else:
          assert(all(torch.allclose(tensors[0], tensor, atol=1e-6)
                     for tensor in tensors[1:]))
          state_dict[key] = tensors[0]

    print(f'Writing merged weights to dir {output_ckpt_dir}')
    create_dir(output_ckpt_dir)
    write_json(params, os.path.join(output_ckpt_dir, "params.json"))
    torch.save(state_dict, os.path.join(output_ckpt_dir, "consolidated.00.pth"))
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_ckpt_dir",
      type=str,
      required=True,
      help="Location of input Llama weights to be merged",
  )
  parser.add_argument(
      "--model_size",
      type=str,
      required=True,
      choices=["7B", "13B", "70B"],
  )
  parser.add_argument(
      "--output_ckpt_dir",
      type=str,
      required=True,
      help="Location of input Llama weights to be merged",
  )
  args = parser.parse_args()

  merge_weights(
      input_ckpt_dir=args.input_ckpt_dir,
      model_size=args.model_size,
      output_ckpt_dir=args.output_ckpt_dir
  )


if __name__ == "__main__":
  main()