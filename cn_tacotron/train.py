import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from tqdm import tqdm

from cn_tacotron import config
from cn_tacotron.text.feature_converter import label_to_sequence
from cn_tacotron.text.phones_mix import phone_to_id
from tacotron_pytorch import Tacotron

use_cuda = torch.cuda.is_available()

if use_cuda:
    print("use GPU")
    cudnn.benchmark = False


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(
        step * warmup_steps ** -1.5, step ** -0.5)
    return lr


model = Tacotron(n_vocab=len(phone_to_id),
                 embedding_dim=512,
                 mel_dim=config.num_mels,
                 linear_dim=config.num_freq,
                 r=config.outputs_per_step,
                 padding_idx=config.padding_idx,
                 use_memory_mask=config.use_memory_mask,
                 )

optimizer = optim.Adam(model.parameters(),
                       lr=config.initial_learning_rate, betas=(
        config.adam_beta1, config.adam_beta2),
                       weight_decay=config.weight_decay)


def train(model,
          data_loader,
          optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = torch.nn.L1Loss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y) in tqdm(enumerate(data_loader)):
            # Decay learning rate
            current_lr = _learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            x, mel, y = x[indices], mel[indices], y[indices]

            # Feed data
            x, mel, y = Variable(x), Variable(mel), Variable(y)
            if use_cuda:
                x, mel, y = x.cuda(), mel.cuda(), y.cuda()
            mel_outputs, linear_outputs, attn = model(
                x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (config.sample_rate * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                          + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                            y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                print(1)
                # save_states(
                #     global_step, mel_outputs, linear_outputs, attn, y,
                #     sorted_lengths, checkpoint_dir)
                # save_checkpoint(
                #     model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(
                model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            # log_value("loss", float(loss.data.item()), global_step)
            # log_value("mel loss", float(mel_loss.data.item()), global_step)
            # log_value("linear loss", float(linear_loss.data.item()), global_step)
            # log_value("gradient norm", grad_norm, global_step)
            # log_value("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.data.item()

        averaged_loss = running_loss / (len(data_loader))
        # log_value("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1


def collate_fn(batch):
    def _pad(seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)),
                      mode='constant', constant_values=0)

    def _pad_2d(x, max_len):
        x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
                   mode="constant", constant_values=0)
        return x

    """Create batch"""
    r = config.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, y_batch


class FileLoad:
    base_dir = "/Users/tristan/gitproject/tacotron_pytorch/cn_tacotron/train_data"

    def __init__(self, suffix):
        self.suffix = suffix
        if suffix == "lab":
            self.sub_dir = "labels"
        if suffix == "npy":
            self.sub_dir = "mels"

    def __getitem__(self, idx):
        file_path = f"{self.base_dir}/{self.sub_dir}/{idx + 1:06d}.{self.suffix}"
        if self.suffix == "lab":
            return label_to_sequence(file_path)
        return 1

    def __len__(self):
        return 49


class ChineseDataset:
    def __init__(self):
        self.X = FileLoad("lab")
        self.Mel = FileLoad("npy")
        self.Y = FileLoad("npy")

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise StopIteration

        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return 49


if __name__ == '__main__':
    dataset = ChineseDataset()
    for i, _, _ in dataset:
        print(len(i))
    # from torch.utils.data import DataLoader
    #
    # dataset = ChineseDataset()
    # data_loader = DataLoader(
    #     dataset, batch_size=config.batch_size,
    #     num_workers=config.num_workers, shuffle=True,
    #     collate_fn=collate_fn, pin_memory=True)
    #
    # train(model, data_loader, optimizer,
    #       init_lr=config.initial_learning_rate,
    #       checkpoint_dir=None,
    #       checkpoint_interval=config.checkpoint_interval,
    #       nepochs=config.nepochs,
    #       clip_thresh=config.clip_thresh)
