"""Evaluation script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed


def run():

    # load dataset
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # Load models
    src_encoder = init_model(net=LeNetEncoder(), restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(), restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(), restore=params.tgt_encoder_restore)

    # Evalute target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)


if __name__ == '__main__':
    run()
