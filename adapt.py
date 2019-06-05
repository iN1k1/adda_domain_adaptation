"""Adversrial Adaptation script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model
from torchsummary import summary


def run():

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    tgt_data_loader = get_data_loader(params.tgt_dataset)

    # load models
    src_encoder = init_model(net=LeNetEncoder(), restore=params.src_encoder_restore)
    tgt_encoder = init_model(net=LeNetEncoder(), restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims, hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims), restore=params.d_model_restore)

    # Adapt target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    im, _ = next(iter(tgt_data_loader))
    summary(tgt_encoder, input_size=im[0].size())
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    # Train target
    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)


if __name__ == '__main__':
    run()
