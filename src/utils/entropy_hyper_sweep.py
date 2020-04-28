import subprocess
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Parameters for train/test script')
  parser.add_argument('--job-name', type=str, default='GSN-hw1')
  parser.add_argument('--partition', type=str, default='common')
  parser.add_argument('--qos', type=str, default='8gpu3d')
  parser.add_argument('--gpu_card_type', type=str, default='')
  parser.add_argument('--gpu_number', type=str, default='1')
  parser.add_argument('--nodelist', type=str, default='asusgpu4')

  parser.add_argument('--learning-rates', '-lrs', nargs='+', type=float, help='Learning rates')
  parser.add_argument('--epochs', type=int, default=5, help='training epochs')
  parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
  parser.add_argument('--test-batch-size', type=int, default=256, help='Testing batch size')
  parser.add_argument('--optimizer', choices=['SGD'], default='SGD')
  parser.add_argument('--data-dir', help='Path to data folders', required=True)
  parser.add_argument('--use-cpu', action='store_true')
  return parser.parse_args()

def prepare_entropy_command(args):
  return '#!/bin/bash\n' + \
        '\#\n' + \
        f'\#SBATCH --job-name={args.job_name}\n' + \
        f'\#SBATCH --partition={args.partition}\n' + \
        f'\#SBATCH --qos={args.qos}\n' + \
        f'\#SBATCH --gres=gpu:{args.gpu_card_type}rtx2080ti:{args.gpu_number}8\n' + \
        f'\#SBATCH --nodelist={args.nodelist}asusgpu2\n'


if __name__ == '__main__':
  args = parse_args()
  cmd = prepare_entropy_command(args)

  for lr in args.learning_rates:
    cmd = f'python3 src/main.py --data-dir {args.data_dir} --learning-rate {lr} ' + \
          f'--optimizer {args.optimizer} --epochs {args.epochs} ' + \
          f'--train-batch-size {args.train_batch_size} --test-batch-size {args.test_batch_size} '
    cmd += f'--use-cpu' if args.use_cpu else ''

    print(f'Executing command: {cmd}')

    subprocess.run(cmd, shell=True)
