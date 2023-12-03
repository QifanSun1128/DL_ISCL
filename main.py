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
from utils.loss import entropy, adentropy
from datetime import datetime

# Training settings
parser = argparse.ArgumentParser(description="SSDA Classification")
parser.add_argument(
    "--steps",
    type=int,
    default=1100,
    metavar="N",
    help="maximum number of iterations " "to train (default: 1100)",
)
parser.add_argument(
    "--method",
    type=str,
    default="MME",
    choices=["S+T", "ENT", "MME"],
    help="MME is proposed method, ENT is entropy minimization,"
    " S+T is training only on labeled examples",
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
record_dir = "record/%s/%s" % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(
    record_dir,
    "%s_net_%s_%s_to_%s_num_%s"
    % (args.method, args.net, args.source, args.target, args.num),
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


im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.to(device)
im_data_t = im_data_t.to(device)
im_data_tu = im_data_tu.to(device)
gt_labels_s = gt_labels_s.to(device)
gt_labels_t = gt_labels_t.to(device)
sample_labels_t = sample_labels_t.to(device)
sample_labels_s = sample_labels_s.to(device)

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


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
        "train_loss": [],
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
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])  # label data source
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])  # ground truth label
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        out1 = F1(output)
        loss = criterion(out1, target)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not args.method == "S+T":
            output = G(im_data_tu)  # im_data_tu is unlabled data in target domain
            if args.method == "ENT":
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == "MME":
                loss_t = adentropy(F1, output, args.lamda)  # adversatrial entropy
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError("Method cannot be recognized.")
            log_train = (
                f"[{current_time}] Source: {args.source} | Target: {args.target} | "
                f"Epoch: {step} | Learning Rate: {lr:.6f} | "
                f"Classification Loss: {loss.data.item():.6f} | "
                f"Entropy: {-loss_t.data.item():.6f} | "
                f"Method: {args.method}\n"
            )

            info_dict["train_loss"].append(loss.data.item())
            info_dict["train_entropy"].append(-loss_t.data.item())

        else:
            log_train = (
                f"[{current_time}] Source: {args.source} | Target: {args.target} | "
                f"Epoch: {step} | Learning Rate: {lr:.6f} | "
                f"Classification Loss: {loss.data.item():.6f} | "
                f"Method: {args.method}\n"
            )

            info_dict["train_g_loss"].append(loss.data.item())

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val, is_val=True)

            info_dict["test_loss"].append(loss_test)
            info_dict["test_acc"].append(acc_test)
            info_dict["val_loss"].append(loss_val)
            info_dict["val_acc"].append(acc_val)

            G.train()
            F1.train()
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
                        "G_iter_model_{}_{}_"
                        "to_{}_step_{}.pth.tar".format(
                            args.method, args.source, args.target, step
                        ),
                    ),
                )
                torch.save(
                    F1.state_dict(),
                    os.path.join(
                        args.checkpath,
                        "F1_iter_model_{}_{}_"
                        "to_{}_step_{}.pth.tar".format(
                            args.method, args.source, args.target, step
                        ),
                    ),
                )

    # Plot for train_loss
    plt.figure(figsize=(10, 6))
    plt.plot(info_dict["train_loss"])
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.show()

    # Plot for train_entropy
    plt.figure(figsize=(10, 6))
    plt.plot(info_dict["train_entropy"])
    plt.xlabel("Steps")
    plt.ylabel("Entropy")
    plt.title("Train Entropy")
    plt.show()

    # Plot for validation loss vs test loss
    plt.figure(figsize=(10, 6))
    plt.plot(info_dict["val_loss"], label="Validation Loss")
    plt.plot(info_dict["test_loss"], label="Test Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train vs Validation vs Test Loss")
    plt.legend()
    plt.show()

    # Plot for validation accuracy vs test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(info_dict["val_acc"], label="Validation Accuracy")
    plt.plot(info_dict["test_acc"], label="Test Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test Accuracy")
    plt.legend()
    plt.show()


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
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
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
    return total_loss.data, 100.0 * float(correct) / size


if __name__ == "__main__":
    train()
