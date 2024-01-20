import argparse
import json
import os
import shutil
import torch
import glob

_MODEL_SIZE_TO_NUM_SHARDS_MAP = {
    "7B": 1,
    "13B": 2,
    "70B": 8,
}

_LAYER_NAME_TO_SHARDING_TYPE_MAP = {
    'tok_embeddings': 'ParallelEmbedding',
    'output': 'ColumnParallelLinear',
    'attention.wq': 'ColumnParallelLinear',
    'attention.wk': 'ColumnParallelLinear',
    'attention.wv': 'ColumnParallelLinear',
    'attention.wo': 'RowParallelLinear',
    'feed_forward.w1': 'ColumnParallelLinear',
    'feed_forward.w2': 'RowParallelLinear',
    'feed_forward.w3': 'ColumnParallelLinear',
    'attention_norm': None,
    'ffn_norm': None,
    'norm': None,
    'rope.freqs': None,
}


def read_json(path):
  with open(path, "r") as f:
    return json.load(f)

def write_json(text, path):
  with open(path, "w") as f:
    json.dump(text, f)

def merge_weights(
    input_ckpt_dir : str,
    model_size: str,
    output_ckpt_dir : str,
) -> None:
  """Merge sharded checkpoint files into a single checkpoint file

  Args:
    input_ckpt_dir: The path to the input directory.
    model_size:
    output_ckpt_dir:

  Returns:
    None
  """
  print(f'Loading checkpoint files from {input_ckpt_dir}')
  checkpoints = [
      torch.load(path, map_location=torch.device('cpu'))
      for path in glob.glob(os.path.join(input_ckpt_dir, '*.pth'))
  ]
  for key in checkpoints.keys():
    print(f'wyzhang: chekcpoint key is {key}')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_ckpt_dir",
      help="Location of input Llama weights to be merged",
  )
  parser.add_argument(
      "--model_size",
      choices=["7B", "13B", "70B"],
  )
  parser.add_argument(
      "--output_ckpt_dir",
      help="Location of input Llama weights to be merged",
  )
  args = parser.parse_args()

  merge_weight(
      input_ckpt_dir=args.input_ckpt_dir,
      output_ckpt_dir=args.output_ckpt_dir

  )


if __name__ == "__main__":
  main()