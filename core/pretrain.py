"""Pre-train encoder and classifier for source dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import params
from utils import make_variable, save_model


def train_on_source_dset(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(tqdm(data_loader, desc="Epoch {}/{}".format(epoch+1, len(range(params.num_epochs_pre))))):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            """if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))
            """
        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_on_source_dset(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pth".format(epoch + 1))
            save_model(classifier, "ADDA-source-classifier-{}.pth".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pth")
    save_model(classifier, "ADDA-source-classifier-final.pth")

    return encoder, classifier


def eval_on_source_dset(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # No grad neede
    with torch.no_grad():

        # set loss function
        criterion = nn.CrossEntropyLoss()

        # evaluate network
        for (images, labels) in data_loader:
            images = make_variable(images)
            labels = make_variable(labels)

            preds = classifier(encoder(images))
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).to("cpu").sum()

        loss = loss / len(data_loader)
        acc = acc.item() / len(data_loader.dataset)

    print("=====================================================================")
    print("Evaluating Model on the Source Dataset")
    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print("=====================================================================")
