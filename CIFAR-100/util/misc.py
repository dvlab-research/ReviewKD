import torch

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def lr_schedule(lr, epoch, optim, args):
    if epoch in args.lr_adjust_step:
        lr *= args.gamma
    for param_group in optim.param_groups:
        param_group['lr']=lr
    return lr

class Logger():
    def __init__(self, args, filename='log.txt'):

        self.filename = filename
        self.file = open(filename, 'a')
        # Write model configuration at top of file
        for arg in vars(args):
            self.file.write(arg+': '+str(getattr(args, arg))+'\n')
        self.file.flush()

    def writerow(self, row):
        for k in row:
            self.file.write(k+': '+row[k]+'  ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()

def load_teacher_weight(teacher, teacher_weight, teacher_model):
    weight = torch.load(teacher_weight)
    teacher.load_state_dict(weight)
    for p in teacher.parameters():
        p.requires_grad = False

