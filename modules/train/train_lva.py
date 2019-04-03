from data.data_process import DataProcessor
from gv_tools.util.logger import Logger
from data.dataloader import TrajectoryDataLoader
from tputils.dataprocessing.pixel_normalize import trajectory_matrix_norm
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from modules.models.lva import LVAttNet
from torch.optim import lr_scheduler
from torch.autograd import Variable
from modules.train.pytorchtools import EarlyStopping
from tputils.evaluate.evaluator import get_ade, get_fde


class Train:
    USE_GPU = torch.cuda.is_available()
    LOSS_FUNC = nn.MSELoss()

    def __init__(self, train_data_dir, test_data_dir, obs_len, pred_len, logger: Logger, results_logger: Logger,
                 normalize_type, img_width, img_height, batch_size, validation_split=0.2):
        self._logger = logger
        self.results_logger = results_logger
        self.batch_size = batch_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.normalize_type = normalize_type
        self.img_height = img_height
        self.img_width = img_width
        self.train_obs = None
        self.train_pred = None
        self.test_obs = None
        self.test_pred = None
        self.dataloader = None
        self.val_dataloader = None
        self.train_sampler = None
        self.validation_sampler = None
        self.train_data = np.load(train_data_dir)
        self.test_data = np.load(test_data_dir)

        if normalize_type != 0:
            self.train_data = trajectory_matrix_norm(self.train_data, img_width=img_width,
                                                     img_height=img_height, mode=normalize_type)
            self.test_data = trajectory_matrix_norm(self.test_data, img_width=img_width,
                                                    img_height=img_height, mode=normalize_type)

        self.data_process(validation_split=validation_split)
        self.net = None
        self.optimizer = None

    def data_process(self, validation_split=0.2):
        self.train_obs = self.train_data[:, 0: self.obs_len, :]
        self.train_pred = self.train_data[:, self.obs_len: self.obs_len + self.pred_len, :]
        self.test_obs = self.test_data[:, 0: self.obs_len, :]
        self.test_pred = self.test_data[:, self.obs_len:, :]
        self.train_obs, self.train_pred = self.process_velocity(self.train_obs, self.train_pred)
        self.test_obs, self.test_pred = self.process_velocity(self.test_obs, self.test_pred)
        self._logger.field('Train Obs Data Size', np.shape(self.train_obs))
        self._logger.field('Train Pred Data Size', np.shape(self.train_pred))
        self._logger.field('Test Obs Data Size', np.shape(self.test_obs))
        self._logger.field('Test Pred Data Size', np.shape(self.test_pred))

        # randomly select validation set
        indices = list(range(len(self.train_data)))
        val_len = int(np.floor(validation_split * len(self.train_data)))
        validation_idx = np.random.choice(indices, size=val_len, replace=False)
        train_idx = list(set(indices) - set(validation_idx))
        # Defining the samplers for each phase based on the random indices:
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.validation_sampler = SubsetRandomSampler(validation_idx)

    def build_dataloader(self):
        self._logger.log('Start Building Data Loader ...')
        dataset = TrajectoryDataLoader(self.train_obs, self.train_pred)
        self.dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            sampler=self.train_sampler
        )
        self._logger.log('Train Data Loader Finished!')

        self.val_dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            sampler=self.validation_sampler
        )
        self._logger.log('Validation Data Loader Finished!')

    def load_network(self, emb_dim, hidden_dim, dropout, lr, step_size, gamma=0.1, output_dim=2):
        self.net = LVAttNet(embedding_dim=emb_dim, hidden_dim=hidden_dim, output_dim=output_dim, obs_len=self.obs_len,
                            pred_len=self.pred_len, drop_out=dropout, gpu=self.USE_GPU)
        if self.USE_GPU:
            self.net.cuda()
        self._logger.log(self.net.__repr__())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        if step_size is not None:
            exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            # Train the model
            exp_lr_scheduler.step()
        self._logger.log('Loc-Vel Net Built')

    def train(self, epochs, save_path, early_stop=False, verbose_step=100, early_stop_patience=10, test_step=500):
        self._logger.log('Start Training ......')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        epoch = 0
        if early_stop:
            early_checker = EarlyStopping(logger=self._logger, patience=early_stop_patience)
        early_stop_flag = False
        data_loaders = {'train': self.dataloader, 'val': self.val_dataloader}
        loss_for_plot = {'train': [], 'val': []}

        while epoch < epochs and not early_stop_flag:
            losses = {'train': [], 'val': []}
            epoch += 1

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train(True)
                else:
                    self.net.train(False)

                for i, (batch_x, batch_y) in enumerate(data_loaders[phase]):
                    if self.USE_GPU:
                        data, target = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                    else:
                        data, target = Variable(batch_x), Variable(batch_y)
                    self.optimizer.zero_grad()
                    hidden = self.net.init_hidden(data.size(0))
                    out = self.net(data, hidden)
                    loss = self.LOSS_FUNC(out, target)

                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        self.optimizer.step()
                    if self.USE_GPU:
                        losses[phase].append(loss.data.cpu()[0])
                    else:
                        losses[phase].append(loss.data[0])
                if epoch % verbose_step == 0:
                    self._logger.field('Epoch', epoch)
                    self._logger.field(phase + ' loss', np.mean(losses[phase]))
                loss_for_plot[phase].append(np.mean(losses[phase]))

            # interval testing
            if epoch % test_step == 0:
                self.interval_test(epoch=epoch, losses=losses, save_path=save_path)

        # final testing
        if not early_stop_flag:
            self.interval_test(epoch=-1, losses=losses, save_path=save_path)

    def save_model(self, save_path, save_name):
        torch.save(self.net.state_dict(), os.path.join(save_path, save_name + '_params.pkl'))
        torch.save(self.net, os.path.join(save_path, save_name + '.pkl'))

    def predict(self, save_path, epoch):
        self.net.eval()
        self._logger.log('Start Predicting ...')
        predicted = []
        for i in range(len(self.test_obs)):
            predicted.append(self.predict_one(self.test_obs[i]))
        predicted = np.reshape(predicted, [len(self.test_obs), self.pred_len, 2])
        norm_ade = get_ade(predicted, self.test_pred, self.pred_len)
        norm_fde = get_fde(predicted, self.test_pred)
        np.save(os.path.join(save_path, str(epoch) + '_predicted.npy'), predicted)

        ground_truth = trajectory_matrix_norm(self.test_pred, img_width=self.img_width,
                                              img_height=self.img_height, mode=2)
        recovered_predicted = trajectory_matrix_norm(predicted, img_width=self.img_width,
                                                     img_height=self.img_height, mode=2)
        ade = get_ade(recovered_predicted, ground_truth, self.pred_len)
        fde = get_fde(recovered_predicted, ground_truth)

        return norm_ade, norm_fde, ade, fde

    def predict_one(self, obs):
        if self.USE_GPU:
            obs = Variable(torch.Tensor(np.expand_dims(obs, axis=0))).cuda()
        else:
            obs = Variable(torch.Tensor(np.expand_dims(obs, axis=0)))
        pred = self.net(obs, self.net.init_hidden())
        predicted_one = pred.data.cpu().numpy()
        predicted_one = np.reshape(predicted_one, [self.pred_len, 2])

        return predicted_one

    def interval_test(self, epoch, losses, save_path):
        self.results_logger.field('Test Epoch', epoch)
        self.results_logger.field('Train Loss', np.mean(losses['train']))
        self.results_logger.field('Val Loss', np.mean(losses['val']))
        norm_ade, norm_fde, ade, fde = self.predict(save_path=save_path, epoch=epoch)
        self.results_logger.field('Test Normalized ADE', norm_ade)
        self.results_logger.field('Test Normalized FDE', norm_fde)
        self.results_logger.field('Test ADE', ade)
        self.results_logger.field('Test FDE', fde)
        self.save_model(save_path=save_path, save_name='interval_epoch_' + str(epoch))

    @staticmethod
    def get_vel(trajectories):
        out = []
        for traj in trajectories:
            offset = []
            for i in range(len(traj)):
                if i == 0:
                    offset.append(np.subtract(traj[1], traj[0]))
                elif i > 0:
                    offset.append(np.subtract(traj[i], traj[i - 1]))
            out.append(offset)

        out = np.reshape(out, [len(trajectories), len(traj), 2])

        return out

    def process_velocity(self, obs, pred):
        data = np.concatenate((obs, pred), axis=1)
        vel = self.get_vel(data)

        input_data = np.concatenate((obs, vel[:, 0: self.obs_len, :]), axis=2)
        output_data = pred

        return input_data, output_data
