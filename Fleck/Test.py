import argparse


parser = argparse.ArgumentParser()
parser.add_argument("Name", help="Input filename", type=str)
parser.add_argument("-n", "--Number", help="Number of Samples.", type=int, default=100)
args = parser.parse_args()

print(args.Number)