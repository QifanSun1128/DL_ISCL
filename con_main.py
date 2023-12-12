from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import adentropy
from datetime import datetime
from utils.loss import ConLoss

# Training settings
parser = argparse.ArgumentParser(description="SSDA Classification")
parser.add_argument(
    "--steps",
    type=int,
    default=8000,
    metavar="N",
    help="maximum number of iterations " "to train (default: 8000)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--multi",
    type=float,
    default=0.1,
    metavar="MLT",
    help="learning rate multiplication",
)
parser.add_argument(
    "--T", type=float, default=0.05, metavar="T", help="temperature (default: 0.05)"
)
parser.add_argument(
    "--lamda", type=float, default=0.1, metavar="LAM", help="value of lamda"
)
parser.add_argument(
    "--save_check", action="store_true", default=False, help="save checkpoint or not"
)
parser.add_argument(
    "--checkpath", type=str, default="./save_model_ssda", help="dir to save checkpoint"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging " "training status",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=500,
    metavar="N",
    help="how many batches to wait before saving a model",
)
parser.add_argument("--net", type=str, default="alexnet", help="which network to use")
parser.add_argument("--source", type=str, default="real", help="source domain")
parser.add_argument("--target", type=str, default="sketch", help="target domain")
parser.add_argument(
    "--dataset",
    type=str,
    default="multi",
    choices=["multi", "office", "office_home"],
    help="the name of dataset",
)
parser.add_argument(
    "--num", type=int, default=3, help="number of labeled examples in the target"
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    metavar="S",
    help="early stopping to wait for improvment "
    "before terminating. (default: 5 (5000 iterations))",
)
parser.add_argument(
    "--early",
    action="store_false",
    default=True,
    help="early stopping on validation or not",
)

args = parser.parse_args()
print(
    "Dataset %s Source %s Target %s Labeled num perclass %s Network %s"
    % (args.dataset, args.source, args.target, args.num, args.net)
)
(
    source_loader,
    target_loader,
    target_loader_unl,
    target_loader_val,
    target_loader_test,
    class_list,
) = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = "record/%s" % (args.dataset)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(
    record_dir,
    "net_%s_%s_to_%s_num_%s" % (args.net, args.source, args.target, args.num),
)

if use_gpu:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

if args.net == "resnet34":
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError("Model cannot be recognized.")

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if "classifier" not in key:
            params += [{"params": [value], "lr": args.multi, "weight_decay": 0.0005}]
        else:
            params += [
                {"params": [value], "lr": args.multi * 10, "weight_decay": 0.0005}
            ]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)
lr = args.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.to(device)  # Feature extractor
F1.to(device)  # classifier

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def create_label_groups(output_logits, labels=None):
    """
    Args:
        output_logits (list): logits which are the softmax output from F(G(x))
        have_label (boolean): distinguish between source and unlabeled target

    Returns:
        dictionary: dictionary: key-(pseudo)label, value-list of logits
    """
    if labels is None:  # create pseudo-label for non-labeled data
        _, labels = torch.max(output_logits, dim=-1)
    label_groupings = {}  # dictionary: key-(pseudo)label, value-list of logits
    for label, logit in zip(labels, output_logits):
        if label.item() not in label_groupings:
            label_groupings[label.item()] = []
        label_groupings[label.item()].append(logit)
    return label_groupings


def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(
        list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True
    )

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().to(device)
    #######
    criterion_con = ConLoss().to(
        device
    )  ### self created contrastive loss, defined in loss.py
    #######
    all_step = args.steps

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0
    counter = 0

    info_dict = {
        "train_cla_loss": [],
        "train_con_loss": [],
        "train_entropy": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
        lr = optimizer_f.param_groups[0]["lr"]
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        im_data_s = Variable(data_s[0].to(device))  # label data source
        gt_labels_s = Variable(data_s[1].to(device))  # ground truth label
        im_data_t = Variable(data_t[0].to(device))
        gt_labels_t = Variable(data_t[1].to(device))
        im_data_tu = Variable(data_t_unl[0].to(device))
        zero_grad_all()

        # labeld data: L_CE
        data_label = torch.cat((im_data_s, im_data_t), 0)
        target_label = torch.cat((gt_labels_s, gt_labels_t), 0)

        output = G(data_label)
        out_label = F1(output)

        loss_ce = criterion(out_label, target_label)

        # unlabeled data: Contrastive Loss
        output2 = G(im_data_tu)
        feat_target_unlabeled = torch.softmax(F1(output2), dim=-1)  # logits
        group_target_unlabeled = create_label_groups(
            output_logits=feat_target_unlabeled
        )

        ns = im_data_s.size(0)  # number of source image
        feat_source = torch.softmax(out_label[:ns], dim=-1)  # feature from source
        group_source = create_label_groups(
            output_logits=feat_source, labels=gt_labels_s
        )

        # calculate contrastive loss between source samples and unlabeled samples
        loss_con = criterion_con(group_source, group_target_unlabeled)

        output = G(im_data_tu)
        loss_t = adentropy(F1, output, args.lamda)  # adversatrial entropy
        ################################
        loss_comb = loss_ce + 0.4 * loss_con + loss_t

        loss_comb.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        ###############################
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_train = (
            f"[{current_time}] Source: {args.source} | Target: {args.target} | "
            f"Epoch: {step} | Learning Rate: {lr:.6f} | "
            f"Contrastive Loss: {loss_con.data.item():.6f} | "
            f"Classification Loss: {loss_ce.data.item():.6f} | "
            f"Total Loss: {loss_comb.data.item():.6f} | "
        )

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)

            info_dict["train_cla_loss"].append(loss_ce.data.item())
            info_dict["train_con_loss"].append(loss_con.data.item())

            G.train()
            F1.train()
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val, is_val=True)
            G.train()
            F1.train()

            info_dict["val_loss"].append(loss_val)
            info_dict["val_acc"].append(acc_val)

            info_dict["test_loss"].append(loss_test)
            info_dict["test_acc"].append(acc_test)

            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print(f"Best Acc Test: {best_acc_test:.6f} | Best Acc Val: {acc_val:.6f}")
            print(f"Record: {record_file}\n")

            with open(record_file, "a") as f:
                f.write(
                    f"Step: {step} | Best: {best_acc_test:.6f} | Final: {acc_val:.6f}\n"
                )

            G.train()
            F1.train()
            if args.save_check:
                print("Model has been saved.\n")
                torch.save(
                    G.state_dict(),
                    os.path.join(
                        args.checkpath,
                        "G_iter_model_{}_"
                        "to_{}_step_{}.pth.tar".format(args.source, args.target, step),
                    ),
                )
                torch.save(
                    F1.state_dict(),
                    os.path.join(
                        args.checkpath,
                        "F1_iter_model_{}_"
                        "to_{}_step_{}.pth.tar".format(args.source, args.target, step),
                    ),
                )

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)

    def scale_epochs(array, epochs_per_point=100):
        return np.arange(len(array)) * epochs_per_point

    # Plot for train loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        scale_epochs(to_numpy(info_dict["train_cla_loss"])),
        to_numpy(info_dict["train_cla_loss"]),
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.savefig("train_loss_plot.png")  # Save the plot as a PNG file
    plt.close()

    # Plot for train contrastive loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        scale_epochs(to_numpy(info_dict["train_con_loss"])),
        to_numpy(info_dict["train_con_loss"]),
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Contrastive Loss")
    plt.savefig("train_con_loss_plot.png")  # Save the plot as a PNG file
    plt.close()

    # Plot for validation loss vs test loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        scale_epochs(to_numpy(info_dict["val_loss"]), epochs_per_point=500),
        to_numpy(info_dict["val_loss"]),
        label="Validation Loss",
    )
    plt.plot(
        scale_epochs(to_numpy(info_dict["test_loss"]), epochs_per_point=500),
        to_numpy(info_dict["test_loss"]),
        label="Test Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation and Test Loss")
    plt.legend()
    plt.savefig("val_test_loss_plot.png")  # Save the plot as a PNG file
    plt.close()

    # Plot for validation accuracy vs test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        scale_epochs(to_numpy(info_dict["val_acc"]), epochs_per_point=500),
        to_numpy(info_dict["val_acc"]),
        label="Validation Accuracy",
    )
    plt.plot(
        scale_epochs(to_numpy(info_dict["test_acc"]), epochs_per_point=500),
        to_numpy(info_dict["test_acc"]),
        label="Test Accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation and Test Accuracy")
    plt.legend()
    plt.savefig("val_test_accuracy_plot.png")  # Save the plot as a PNG file
    plt.close()


def test(loader, is_val=False):
    G.eval()
    F1.eval()
    total_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = Variable(data_t[0].to(device))
            gt_labels_t = Variable(data_t[1].to(device))
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            total_loss += criterion(output1, gt_labels_t) / len(loader)

    set_type = "Validation" if is_val else "Test"
    print(
        f"{set_type} set: Average loss: {total_loss:.4f}, "
        f"Accuracy: {correct}/{size} ({100.0 * correct / size:.0f}%)\n"
    )
    return total_loss.data.item(), 100.0 * float(correct) / size


if __name__ == "__main__":
    train()
