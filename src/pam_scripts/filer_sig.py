import argparse
import json

CUTOFF = 0.99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signature")
    args = parser.parse_args()
    print(args)
