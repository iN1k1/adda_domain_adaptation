"""Pretrain script for ADDA"""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from torchsummary import summary


def run():

    # load source dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)

    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)

    # pre-train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    im, _ = next(iter(src_data_loader))
    summary(src_encoder, input_size=im[0].size())
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)


if __name__ == '__main__':
    run()
