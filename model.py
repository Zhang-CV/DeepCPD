import gc
import numpy as np
import torch
from torch import optim
from network import Network
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from data import refinement_data
from UtilsNetwork import *
import math

src_refine = refinement_data()


class DeepCPD:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.batch_size_train = args.batch_size_train
    self.batch_size_val = args.batch_size_val
    self.batch_size_test = args.batch_size_test
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.num_clusters = args.num_clusters
    self.eval = args.eval
    self.icp = args.icp
    self.network = Network(args, self.num_clusters)
    self.model_path = args.model_path

    if self.cuda:
      self.network = self.network.cuda()

  def register_loss(self, rotation_pred, translation_pred, rotation_ab, translation_ab):

    identity = torch.eye(3).unsqueeze(0).repeat(rotation_pred.size(0), 1, 1)
    if self.cuda == 1:
      identity = identity.cuda()
    loss_fn = torch.nn.MSELoss()
    loss_register = loss_fn(torch.matmul(rotation_pred.transpose(2, 1), rotation_ab), identity) + \
                    loss_fn(translation_pred, translation_ab)
    loss_register = loss_register
    angle_error = np.rad2deg(np.arccos(
      np.minimum((np.trace(np.matmul(rotation_ab.detach().cpu(), rotation_pred.detach().cpu().transpose(2, 1)), axis1=1, axis2=2) - 1.) / 2., 1.)))
    angle_error = angle_error.mean()
    translation_error = np.linalg.norm(translation_ab.detach().cpu() - translation_pred.detach().cpu(), axis=2).mean()
    return loss_register, angle_error, translation_error

  def train_epoch(self, optimizer, data_loader):
    """Train the model for one epoch
    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data
    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    if self.model_path:
        self.network.load_state_dict(torch.load(self.model_path))
    # total_loss = 0.
    register_loss = 0.
    num_batches = 0.
    angles_error = 0.
    translations_error = 0.

    for src, target, rotation_ab, translation_ab in tqdm(data_loader):
      if self.cuda == 1:
        src, target, rotation_ab, translation_ab = src.cuda(), target.cuda(), rotation_ab.cuda(), translation_ab.cuda()
      optimizer.zero_grad()

      y_src = self.network(src)
      y_target = self.network(target)
      pi_src, mu_src, sigma_src = gmm_params(y_src, src)
      rotation_pred, translation_pred = gmm_register(mu_src, sigma_src, y_target, target)
      registration_loss1, angle_error, translation_error = \
        self.register_loss(rotation_pred, translation_pred, rotation_ab, translation_ab)

      rotation_ab = rotation_pred.inverse() @ rotation_ab
      rotation_ab = rotation_ab.detach()
      translation_ab = rotation_pred.inverse() @ (translation_ab - translation_pred).transpose(2, 1).contiguous()
      translation_ab = translation_ab.transpose(2, 1).contiguous().detach()
      target = rotation_pred.inverse() @ (target.transpose(2, 1).contiguous() - translation_pred.transpose(2, 1).contiguous())
      target = target.transpose(2, 1).contiguous().detach()

      y_target = self.network(target)
      rotation_pred, translation_pred = gmm_register(mu_src, sigma_src, y_target, target)
      registration_loss2, angle_error, translation_error = \
        self.register_loss(rotation_pred, translation_pred, rotation_ab, translation_ab)

      total = registration_loss1 + registration_loss2

      # accumulate values
      angles_error += angle_error
      translations_error += translation_error
      register_loss += registration_loss1 + registration_loss2

      # perform backpropagation
      total.backward()
      optimizer.step()
   
      num_batches += 1.

    # average per batch
    register_loss /= num_batches
    angles_error /= num_batches
    translations_error /= num_batches

    return register_loss, angles_error, translations_error

  def test(self, data_loader):
    """Test the model with new data
    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
    Return:
        accuracy and error for the given test data
    """
    self.network.eval()
    if self.eval:
        model_path = "checkpoints/deepcpd/models/model.best.t7"
        self.network.load_state_dict(torch.load(model_path))
    register_loss = 0.
    num_batches = 0.
    angles_error = 0.
    translations_error = 0.
    time_costs = 0.
    removes = 0.
    if self.eval:
      batch_size = self.batch_size_test
    else:
      batch_size = self.batch_size_val
    with torch.no_grad():
      for src, target, rotation_ab, translation_ab in tqdm(data_loader):
        if self.cuda == 1:
          src, target, rotation_ab, translation_ab = src.cuda(), target.cuda(), rotation_ab.cuda(), translation_ab.cuda()

        # forward call
        y_src = self.network(src)
        y_target = self.network(target)
        time_start = time.time()
        pi_src, mu_src, sigma_src = gmm_params(y_src, src)
        rotation_pred, translation_pred = gmm_register(mu_src, sigma_src, y_target, target)
        registration_loss1, _, _ = \
          self.register_loss(rotation_pred, translation_pred, rotation_ab, translation_ab)

        target2src = rotation_pred.inverse() @ (target.transpose(2, 1).contiguous() - translation_pred.transpose(2, 1).contiguous())
        target2src = target2src.transpose(2, 1)

        y_target = self.network(target2src)
        rotation_pred, translation_pred = gmm_register(mu_src, sigma_src, y_target, target)
        registration_loss2, angle_error, translation_error = \
          self.register_loss(rotation_pred, translation_pred, rotation_ab, translation_ab)

        if(self.icp):
          rotation_ab = rotation_pred.inverse() @ rotation_ab
          translation_ab = rotation_pred.inverse() @ (translation_ab - translation_pred).transpose(2, 1)
          translation_ab = translation_ab.transpose(2, 1)
          target = rotation_pred.inverse() @ (target.transpose(2, 1) - translation_pred.transpose(2, 1))
          target = target.transpose(2, 1)
          rotation_pred_icp, translation_pred_icp, _ = ICP(src_refine, target.squeeze(0).cpu())
          rotation_pred_icp = torch.from_numpy(rotation_pred_icp.astype('float32')).unsqueeze(0).cuda()
          translation_pred_icp = torch.from_numpy(translation_pred_icp.astype('float32')).unsqueeze(0).cuda()
          _, angle_error, translation_error = \
            self.register_loss(rotation_pred_icp, translation_pred_icp, rotation_ab, translation_ab)

        time_end = time.time()
        # accumulate values
        register_loss += registration_loss1 + registration_loss2
        angles_error += angle_error
        translations_error += translation_error
        time_costs += (time_end - time_start)
        num_batches += 1.
    # average per batch
    # if return_loss:
    register_loss /= num_batches
    angles_error /= num_batches
    translations_error /= num_batches
    time_costs /= (num_batches * batch_size)
    return register_loss, angles_error, translations_error, time_costs

  def train(self, train_loader, val_loader, textio, boardio):
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=self.lr_decay)

    best_test_loss = np.inf

    for epoch in range(1, self.num_epochs + 1):
      train_loss, train_angle_error, train_translation_error = self.train_epoch(optimizer, train_loader)
      gc.collect()
      with torch.no_grad():
        val_loss, val_angle_error, val_translation_error, _ = self.test(val_loader)
      if best_test_loss >= val_loss:
        best_test_loss = val_loss
        torch.save(self.network.state_dict(), 'checkpoints/deepcpd/models/model.best.t7')

      textio.write('==Train==' + '\n')
      textio.write('A-->B' + '\n')
      textio.write('EPOCH:: %d, Loss: %f' % (epoch, train_loss) + '\n')
      textio.write('==Test==' + '\n')
      textio.write('A-->B' + '\n')
      textio.flush()
      boardio.add_scalar('A-B/train/loss', train_loss, epoch)
      boardio.add_scalar('A-B/train/train_angle_error', train_angle_error, epoch)
      boardio.add_scalar('A-B/train/train_translation_error', train_translation_error, epoch)
      boardio.add_scalar('A-B/test/loss', val_loss, epoch)
      boardio.add_scalar('A-B/test/test_angle_error', val_angle_error, epoch)
      boardio.add_scalar('A-B/test/test_translation_error', val_translation_error, epoch)
      boardio.add_scalar('A-B/best_test/loss', best_test_loss, epoch)

      if epoch % 10 == 0:
        torch.save(self.network.state_dict(), 'checkpoints/deepcpd/models/model.%d.t7' % epoch)
      scheduler.step()
      gc.collect()  # clear memory