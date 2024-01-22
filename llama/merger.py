import argparse
import json
import shutil
import torch
import glob
from typing import List, Dict
import hashlib
import os

_MODEL_SIZE_TO_NUM_SHARDS_MAP = {
    "7B": 1,
    "13B": 2,
    "70B": 8,
}

# ParallelEmbedding is row partitioned and the shape of each partition
# is (num_embeddings_per_partition, embedding_dim).
# ColumnParallelLinear is col partitioned.
# RowParallelLinear is row partitioned.
_WEIGHT_SHARDING_TYPE = {
    'tok_embeddings.weight': 'ParallelEmbedding',
    'rope.freqs': None,
    # ColumnParallelLinear is col partitioned
    'attention.wq.weight': 'ColumnParallelLinear',
    'attention.wk.weight': 'ColumnParallelLinear',
    'attention.wv.weight': 'ColumnParallelLinear',
    'attention.wo.weight': 'RowParallelLinear',
    'feed_forward.w1.weight': 'ColumnParallelLinear',
    'feed_forward.w2.weight': 'RowParallelLinear',
    'feed_forward.w3.weight': 'ColumnParallelLinear',
    'attention_norm.weight': None,
    'ffn_norm.weight': None,
    'norm.weight': None,
    'output.weight': 'ColumnParallelLinear',
}

def _compute_md5(file_path:str)->None:
  md5_hash = hashlib.md5()
  with open(file_path, "rb") as file:
    while True:
      data = file.read(4096)  # Read in 4KB chunks
      if not data:
        break
      md5_hash.update(data)
  return md5_hash.hexdigest()

def _generate_md5_checklist(target_dir:str)->None:
  files = [os.path.join(target_dir, file)
               for file in os.listdir(target_dir)
               if os.path.isfile(os.path.join(target_dir, file))]

  checklist_file = os.path.join(target_dir, "checklist.chk")
  with open(checklist_file, 'w') as checklist:
    for f in files:
      md5_checksum = _compute_md5(f)
      checklist.write(f'{md5_checksum}  {os.path.basename(f)}\n')

def _create_dir(target_dir:str)->None:
  if os.path.exists(target_dir):
    # If it exists, remove it and all its contents
    shutil.rmtree(target_dir)
  os.makedirs(target_dir)

def _checkpoints_have_same_weight_keys(checkpoint_list:List[Dict[str, torch.Tensor]]):
  if (not checkpoint_list) or len(checkpoint_list) <= 1:
    return True
  for m in checkpoint_list[1:]:
    if set(checkpoint_list[0].keys()) != set(m.keys()):
      return False
  return True

def _tensors_have_same_shape(tensors):
  if (not tensors) or len(tensors) <= 1:
    return True
  # Iterate through the remaining tensors in the list
  for t in tensors[1:]:
    if t.shape != tensors[0].shape:
      return False
  return True

def _read_json(path):
  with open(path, "r") as f:
    return json.load(f)

def _write_json(text, path):
  with open(path, "w") as f:
    json.dump(text, f)

def _merge_weights(
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
  params = _read_json(os.path.join(input_ckpt_dir, "params.json"))
  n_layers = params['n_layers']
  print(f'Loaded params. n_layers={n_layers}')

  print(f'Loading checkpoint files from {input_ckpt_dir}.')
  checkpoints = [
      torch.load(path, map_location=torch.device('cpu'))
      for path in glob.glob(os.path.join(input_ckpt_dir, '*.pth'))
  ]
  assert len(checkpoints) > 0, f'No *.pth found in input dir {input_ckpt_dir}'

  print(f'Starting to merge weights.')
  state_dict = {}
  assert _checkpoints_have_same_weight_keys(checkpoints)
  weight_keys = checkpoints[0].keys()
  for key in weight_keys:
    tensors: List[torch.Tensor] = [c[key] for c in checkpoints]
    assert(_tensors_have_same_shape(tensors))
    print(f'Merging weights across '
          f'{len(tensors)} shards (shape = {tensors[0].shape}) for {key})')
    for pattern, kind in _WEIGHT_SHARDING_TYPE.items():
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
  _create_dir(output_ckpt_dir)
  _write_json(params, os.path.join(output_ckpt_dir, "params.json"))
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

  _merge_weights(
      input_ckpt_dir=args.input_ckpt_dir,
      model_size=args.model_size,
      output_ckpt_dir=args.output_ckpt_dir
  )


if __name__ == "__main__":
  main()