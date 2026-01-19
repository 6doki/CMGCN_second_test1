import math
import torch
import torch.optim as optim


class Optim(object):

    def __init__(self, params, config):
        self.params = list(params)  # careful: params may be a generator
        self.config = config
        self.last_ppl = None
        self.lr = self.config.get("lr", 0.001)  # 0.001
        self.max_grad_norm = self.config.get("clip", 10)    # 10
        self.method = self.config.get("optim", "adam")  # adam
        self.lr_decay = self.config.get("lr_decay", False)  # false
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')  # 'multisteplr'
        self.lr_decay_ratio = self.config.get("lr_decay_ratio", 0.1)    # 0.1
        self.milestones = self.config.get("lr_decay_steps", []) # [5, 20, 40, 70]
        self.step_size = self.config.get("step_size", 10)   # 10

        self._makeOptimizer()
        self.lr_scheduler = self._build_lr_scheduler()  # None


    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':     # adam
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            else:
                print('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:   # false
            lr_scheduler = None
        return lr_scheduler


    def step(self):
        # Compute gradients norm. 计算梯度的范数(norm)
        grad_norm = 0  # 用于计算梯度的范数(norm)。梯度范数用于存储计算出的梯度范数

        if self.max_grad_norm is not None:  # 执行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)  # 用于限制模型参数的梯度范数不超过给定的最大值max_grad_norm

        self.optimizer.step()
        return grad_norm

    
    def zero_grad(self):
        self.optimizer.zero_grad()
        return


    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
