from model import DeepLight, Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import create_circular_mask
from loss import adversarial_loss, mse_loss
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import torch


class Trainer():
    def __init__(self, train_data, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_data = train_data
        self.deeplight = DeepLight().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.gen_opt = torch.optim.AdamW(self.deeplight.parameters(), lr=args.lr)
        self.dis_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=args.lr)

        self.mask = torch.cat(3*[create_circular_mask(32, 32).unsqueeze_(0)])

    def run_epoch(self, epoch, training):
        self.deeplight.train(training)
        self.discriminator.train(training)
        self.training = training
        self.epoch = epoch
        if training:
            description = 'Train'
            dataset = self.train_data
            shuffle = True
        else:
            description = 'Valid'
            dataset = self.train_data
            shuffle = False

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        for idx, (raw, silver_ball, white_ball, yellow_ball) in trange:
            if idx == 0:
                gen_loss, dis_loss = self._run_iter(raw, silver_ball, white_ball, yellow_ball, True)
            else:
                gen_loss, dis_loss = self._run_iter(raw, silver_ball, white_ball, yellow_ball)
            trange.set_postfix(gen_loss=gen_loss, dis_loss=dis_loss)

    def _run_iter(self, raw, silver_ball, white_ball, yellow_ball, save_fig=False):
        b, c, w, h = raw.shape
        raw = raw.to(self.device)
        masks = torch.cat(b*[self.mask.unsqueeze(0)]).to(self.device)

        silver_ball = silver_ball.to(self.device) * masks
        white_ball = white_ball.to(self.device) * masks
        yellow_ball = yellow_ball.to(self.device) * masks

        real = Variable(torch.FloatTensor(b, 1).fill_(1.0), requires_grad=False).to(self.device)
        fake = Variable(torch.FloatTensor(b, 1).fill_(0.0), requires_grad=False).to(self.device)

        output = self.deeplight(raw)
        silver_output = output[:, 0:3, :, :] * masks
        white_output = output[:, 3:6, :, :] * masks
        yellow_output = output[:, 6:9, :, :] * masks

        if save_fig == True:
            self.save_fig(silver_output, white_output, yellow_output)

        gen_loss = mse_loss(silver_output, silver_ball) / 3
        gen_loss += mse_loss(white_output, white_ball) / 3
        gen_loss += mse_loss(yellow_output, yellow_ball) / 3
        gen_loss += adversarial_loss(self.discriminator(silver_output), real) / 3
        gen_loss += adversarial_loss(self.discriminator(white_output), real) / 3
        gen_loss += adversarial_loss(self.discriminator(yellow_output), real) / 3

        if self.training:
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()

        dis_loss = adversarial_loss(self.discriminator(silver_ball), real)
        dis_loss += adversarial_loss(self.discriminator(silver_output.detach()), fake)
        dis_loss += adversarial_loss(self.discriminator(white_ball), real)
        dis_loss += adversarial_loss(self.discriminator(white_output.detach()), fake)
        dis_loss += adversarial_loss(self.discriminator(yellow_ball), real)
        dis_loss += adversarial_loss(self.discriminator(yellow_output.detach()), fake)
        dis_loss /= 6

        if self.training:
            self.dis_opt.zero_grad()
            dis_loss.backward()
            self.dis_opt.step()

        return gen_loss.item(), dis_loss.item()

    def save(self, path):
        torch.save({"deeplight": self.deeplight.state_dict(), "discriminator": self.discriminator.state_dict()}, path)

    def save_fig(self, silver, white, yellow):
        silver = (silver[0].detach().cpu() - 1) / 2
        white = (white[0].detach().cpu() - 1) / 2
        yellow = (yellow[0].detach().cpu() - 1) / 2
        silver_img = TF.to_pil_image(silver)
        white_img = TF.to_pil_image(white)
        yellow_img = TF.to_pil_image(yellow)

        silver_img.save(f"result/silver-{self.epoch}.png")
        white_img.save(f"result/white-{self.epoch}.png")
        yellow_img.save(f"result/yellow-{self.epoch}.png")
