from model.models import *
from model.PerceptualLoss import *
import time
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from data_utils.RESIDE_data import *
import cv2
from option import opt, model_name, log_dir
from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter
from metrics import msssim
warnings.filterwarnings('ignore')
print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {
    'MCP': Net(3),
}
loaders_ = {
    'haze_train': HAZE_train_loader,
    'haze_test': HAZE_test_loader
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        out1, out2, out3 = net(x)
        # Loss = nn.L1Loss(size_average=True, reduce=True)
        # L2Loss = nn.MSELoss(size_average=True,reduce=True)
        Loss = nn.SmoothL1Loss(size_average=True,reduce=True)
        loss1 = Loss(out1, y)
        loss2 = Loss(out2, y)
        loss3 = Loss(out3, y)
        loss_L1 = loss3 + loss1 + loss2

        ssim_loss = msssim
        loss_SSIM = 1 - ssim_loss(out3, y)
        vgg = vgg16(pretrained=True).features.to(opt.device)
        perloss = LossNetwork(vgg).to(opt.device)
        ploss = perloss(out3, y)

        loss = 0.5 * loss_SSIM + loss_L1 + 0.01 * ploss

        loss.backward()
        # torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10, norm_type=2)

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        # writer.add_scalar('data/loss',loss,step)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            with SummaryWriter(log_dir=log_dir, comment=log_dir) as writer:
                writer.add_scalar('data/ssim', ssim_eval, step)
                writer.add_scalar('data/psnr', psnr_eval, step)
                writer.add_scalars('group', {
                    'ssim': ssim_eval,
                    'psnr': psnr_eval,
                    'loss': loss
                }, step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model_state_dict': net.state_dict(),  # 保存模型参数
                    'optimizer_state_dict': optimizer.state_dict()  # 保存优化器参数
                }, opt.model_dir + '.tar')
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    # s=True
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        x1, x2, pred = net(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer)