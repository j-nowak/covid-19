#!/usr/bin/env python

from prepare_datasets import prepare_ieee8023_dataset, prepare_figure1_dataset, prepare_rsna_challenge_dataset, split_train_test

def main():
    prepare_ieee8023_dataset()
    prepare_figure1_dataset()
    prepare_rsna_challenge_dataset()
    split_train_test()


if __name__ == "__main__":
    main()
