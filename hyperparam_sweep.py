import os
import subprocess
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Parameters for train/test script')
  parser.add_argument('--job-name', type=str, default='GSN-hw1')
  parser.add_argument('--partition', type=str, default='common')
  parser.add_argument('--qos', type=str, default='8gpu3d')
  parser.add_argument('--gpu-card-type', type=str, default='')
  parser.add_argument('--gpu-number', type=str, default='1')
  parser.add_argument('--nodelist', type=str, default='asusgpu4')

  parser.add_argument('--learning-rates', '-lrs', nargs='+', type=float, help='Learning rates')
  parser.add_argument('--epochs', type=int, default=5, help='training epochs')
  parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
  parser.add_argument('--test-batch-size', type=int, default=256, help='Testing batch size')
  parser.add_argument('--optimizer', choices=['SGD'], default='SGD')
  parser.add_argument('--momentum', type=float, default=0.0)
  parser.add_argument('--weight-decay', type=float, default=0.0001)
  parser.add_argument('--data-dir', help='Path to data folders', required=True)
  parser.add_argument('--use-cpu', action='store_true')
  return parser.parse_args()

def prepare_entropy_command(args):
  cmd = '#!/bin/bash\n'
  cmd += '#\n'
  cmd += f'#SBATCH --job-name={args.job_name}\n'
  cmd += f'#SBATCH --partition={args.partition}\n'
  cmd += f'#SBATCH --qos={args.qos}\n'
  cmd += f'#SBATCH --gres=gpu:{args.gpu_card_type}:{args.gpu_number}\n'
  cmd += f'#SBATCH --nodelist={args.nodelist}\n\n'
  return cmd


if __name__ == '__main__':
  args = parse_args()

  for lr in args.learning_rates:
    # os.remove('job.sh')
    # cmd = prepare_entropy_command(args)
    cmd = f'python3 src/main.py --data-dir {args.data_dir} --learning-rate {lr} '
    cmd += f'--optimizer {args.optimizer} --epochs {args.epochs} '
    cmd += f'--train-batch-size {args.train_batch_size} --test-batch-size {args.test_batch_size} '
    cmd += f'--momentum {args.momentum} --weight-decay {args.weight_decay} '
    cmd += f'--use-cpu' if args.use_cpu else ''
    cmd += '\n'
    subprocess.run(cmd, shell=True)

    # with open('job.sh', 'w') as f:
    #   f.write(cmd)
    # print(f'Executing command: {cmd}')
    # subprocess.run('sbatch job.sh', shell=True)
