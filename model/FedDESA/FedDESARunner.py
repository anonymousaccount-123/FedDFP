import sys, os
from sklearn.preprocessing import label_binarize
from copy import deepcopy
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from tqdm import tqdm
from utils.Get_model import define_model
from utils.Get_data import define_data
from utils.trainer_util import get_optim, get_loss
import torch
from torch import nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from utils.data_utils import get_split_loader, CategoriesSampler, save_syn_img
from utils.core_util import clam_runner, raw_feature_extract
from torch.utils.data import DataLoader, TensorDataset
from model.condensation import distribution_matching, distribution_matching_woMIL
from sklearn.metrics import roc_curve
import random
from utils.core_util import Distance_loss
import torch.nn.functional as F

def check_sublists_equal_size(lst):
    if not lst:  # Handle the case of an empty list
        return True

    # Get the size of the first sublist
    first_size = len(lst[0])

    # Check if all other sublists have the same size
    return all(len(sublist) == first_size for sublist in lst)

def get_images(images_all, indices_class, c, n): # get random n images from class c
    print(f'Sample {n} images from class {c} with {len(indices_class[c])} images')
    if n < len(indices_class[c]):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
    else:
        idx_shuffle_0 = np.random.permutation(indices_class[c])
        idx_shuffle_1 = np.random.permutation(indices_class[c])[:n-len(indices_class[c])]
        idx_shuffle = np.concatenate([idx_shuffle_0, idx_shuffle_1], axis=0)
    img_pth = images_all[idx_shuffle[0]]
    X = torch.load(img_pth).unsqueeze(0)
    return X

def calculate_kd_loss(y_pred_student, y_pred_teacher, y_true, loss_fn, temp=20., distil_weight=0.9):
    """
    Function used for calculating the KD loss during distillation

    :param y_pred_student (torch.FloatTensor): Prediction made by the student model
    :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
    :param y_true (torch.FloatTensor): Original label
    """

    soft_teacher_out = F.softmax(y_pred_teacher / temp, dim=1)
    soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)

    loss = (1. - distil_weight) * F.cross_entropy(y_pred_student, y_true)
    loss += (distil_weight * temp * temp) * loss_fn(
        soft_student_out, soft_teacher_out
    )

    return loss


class FedDESA:
    def __init__(self, args, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.logger = logger
        require_image_function = True if self.args.fed_method in ['fed_dm', 'fed_af', 'fed_desa'] else False
        self.train_dataset, self.test_dataset, self.n_clients = define_data(args, logger,
                                                                            require_image=require_image_function,
                                                                            image_size=args.syn_size)
        self.init_loss_fn()
        print('Number of clients:', len(self.n_clients))
        for i in range(len(self.n_clients)):
            print(f'    Train: {len(self.train_dataset[i])}; Test: {len(self.test_dataset[i])}')

    def get_train_loader(self, ds):
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(ds.labels,
                                              n_batch=len(ds.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=ds, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(ds, training=True, weighted=self.args.weighted_sample)
        return train_loader

    def get_test_loader(self, ds):
        if 'frmil' in self.args.mil_method:
            test_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            test_loader = get_split_loader(ds)
        return test_loader

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)
        self.dist_loss = nn.MSELoss()
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.CE_loss = nn.CrossEntropyLoss()
        self.mil_loss = get_loss(self.args)
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")

    def init_dataset(self, local_train_ds, local_test_ds):
        train_dataset = local_train_ds
        test_dataset = local_test_ds
        return self.get_train_loader(train_dataset), self.get_test_loader(test_dataset)

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list)

    def mil_run(self, model,
                data,
                label,
                loss_fn,
                return_lgt=False,
                return_feature=False,
                raw_image=False):
        if 'CLAM' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = clam_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = clam_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = clam_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = clam_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'TransMIL' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = transmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'Attention' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = abmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob

    def local_train(self, agent_idx, local_model, local_optim, train_loader, test_loader):
        local_model.train()
        epoch_loss = 0.
        for ep in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                local_model.zero_grad()
                loss, error, y_prob = self.mil_run(local_model, images, labels, self.mil_loss)
                loss.backward()
                local_optim.step()
                if batch_idx % 20 == 0:
                    self.logger.info(f'Agent: {agent_idx}, Iter: {ep}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()

            batch_loss /= len(train_loader)
            batch_error /= len(train_loader)
            epoch_loss += batch_loss
        return epoch_loss / self.args.local_epochs

    def local_test(self, model, test_loader):
        model.eval()
        total_loss = 0.
        total_error = 0.
        all_probs = np.zeros((len(test_loader), self.args.n_classes))
        all_labels = np.zeros(len(test_loader))
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            loss, error, Y_prob = self.mil_run(model, images, labels, self.mil_loss)
            total_loss += loss.item()
            total_error += error
            probs = Y_prob.detach().cpu().numpy()

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = labels.item()

        total_loss /= len(test_loader)
        total_error /= len(test_loader)
        if self.args.n_classes == 2:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        else:
            fpr = dict()
            tpr = dict()
            y_true_bin = label_binarize(all_labels, classes=list(range(self.args.n_classes)))
            for i in range(y_true_bin.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        return total_loss, total_error, fpr, tpr

    def pretrain_clients(self, repeat):
        clients_models_pre = []
        optimizers_pre = []
        MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']# more will be added
        for idx in range(len(self.n_clients)):
            if self.args.heter_model:
                seed = 33 + repeat + len(MIL_pool)
                random.seed(seed)
                self.args.mil_method = random.choice(MIL_pool)
                if len(MIL_pool) > 0:
                    MIL_pool.remove(self.args.mil_method)
                if len(MIL_pool) == 0:
                    MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']
                print(f'=====> Agent {idx} uses {self.args.mil_method} {self.args.opt}')
                clients_models_pre.append(define_model(self.args))
                optimizers_pre.append(get_optim(self.args, clients_models_pre[idx]))
            else:
                clients_models_pre.append(define_model(self.args))
                optimizers_pre.append(get_optim(self.args, clients_models_pre[idx]))
        # clients_models_pre = [define_model(self.args) for _ in range(len(self.n_clients))]
        # optimizers_pre = [get_optim(self.args, model) for model in clients_models_pre]
        self.logger.info('=========================Pretraining Clients=========================')
        for client_idx in range(len(self.n_clients)):
            train_loader, test_loader = self.init_dataset(self.train_dataset[client_idx], self.test_dataset[client_idx])
            pretrained_model_path = f'{self.args.results_dir}/client_{client_idx}_{clients_models_pre[client_idx].__class__.__name__}_pretrain.pt'
            if os.path.exists(pretrained_model_path):
                clients_models_pre[client_idx].load_state_dict(torch.load(pretrained_model_path))
                self.logger.info(f'Client {client_idx} Pretrained model loaded from {pretrained_model_path}')
            else:
                clients_models_pre[client_idx].to(self.device)
                train_loss = self.local_train(client_idx, clients_models_pre[client_idx],
                                              optimizers_pre[client_idx], train_loader, test_loader)
            test_loss, test_error,fpr, tpr = self.local_test(clients_models_pre[client_idx], test_loader)
            ''' Save checkpoint '''
            self.logger.info(f'Client {client_idx} Test Loss: {test_loss}, Test Error: {test_error}')
            torch.save(clients_models_pre[client_idx].state_dict(), pretrained_model_path)
        return clients_models_pre

    def run(self, repeat):
        repeat = 2
        ''' Pretrain/Load local models '''
        clients_models_pre = self.pretrain_clients(repeat)

        '''Train/Load virtual data'''
        self.logger.info('=========================Training synthetic data=========================')

        label_syns_tmp = torch.tensor(np.array([np.ones(self.args.ipc) * i for i in range(self.args.n_classes)]), dtype=torch.long,
                                      requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        image_syns = [torch.randn(size=(self.args.n_classes*self.args.ipc, self.args.nps, 3, self.args.syn_size, self.args.syn_size),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device) for idx in range(len(self.n_clients))]
        # number of element that require grad in image_syns
        print('Number of element that require grad in image_syns:', image_syns[0].size(), image_syns[0].numel())

        if self.args.init_real:
            for idx in range(len(self.n_clients)):
                for i, c in enumerate(range(self.args.n_classes)):
                    image_syns[idx].data[i * self.args.ipc: (i + 1) * self.args.ipc] = self.train_dataset[idx].get_image(c, self.args.ipc, self.args.nps).detach().data
        label_syns = [copy.deepcopy(label_syns_tmp).to(self.device) for idx in range(len(self.n_clients))]
        client_label_list = [[] for _ in range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            # organize the real dataset
            indices_class = [[] for c in range(self.args.n_classes)]
            # images_all = [torch.unsqueeze(self.train_dataset[client_idx][i][0], dim=0) for i in
            #               range(len(self.train_dataset[client_idx]))]
            images_all, labels_all = [], []
            for i in range(len(self.train_dataset[client_idx])):
                images_all.append(self.train_dataset[client_idx].__getitem__(i, path=True)[2])
                labels_all.append(self.train_dataset[client_idx].__getitem__(i)[1])

            for idx, lab in enumerate(labels_all):
                lab_item = int(lab.item()) if isinstance(lab, torch.Tensor) else int(lab)
                indices_class[lab_item].append(idx)
            for i in range(len(indices_class)):
                if len(indices_class[i]) == 0:
                    print(f'[WARNINNG] Client {client_idx} has no label {i}')
                    # remove the class with no label
                    indices_class.pop(i)
                else:
                    client_label_list[client_idx].append(i)
            # images_all = torch.cat(images_all, dim=0).to(self.device)
            # labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
            # print('Images all size:', images_all.size(), labels_all.size())

            # modify the size of synthetic data if client has less label than self.args.n_classes
            if len(indices_class) < self.args.n_classes:
                image_syns[client_idx] = deepcopy(image_syns[client_idx][:len(indices_class) * self.args.ipc].detach())
                label_syns[client_idx] = deepcopy(label_syns[client_idx][:len(indices_class) * self.args.ipc].detach())
                image_syns[client_idx].requires_grad = True
            if self.args.image_opt == 'adam':
                optimizer_img = torch.optim.Adam([image_syns[client_idx], ], lr=self.args.image_lr,)
            elif self.args.image_opt == 'sgd':
                optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=self.args.image_lr,
                                                momentum=0.5)
            # optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=10.0,
            #                                 momentum=0.5)  # optimizer_img for synthetic data
            inv_iters = self.args.dc_iterations
            image_batch = 1
            dm_loss_avg = 0
            pbar_dm = tqdm(range(inv_iters), desc=f'Client {client_idx} - DM loss {dm_loss_avg}')
            for it in pbar_dm:
                # get real images for each class
                image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(len(indices_class))]
                # print([image_real[i].size(0) for i in range(len(image_real))])
                # check if image_syns[client_idx] requries grad update
                loss, image_syns[client_idx] = distribution_matching_woMIL(
                                                                     image_real,
                                                                     image_syns[client_idx],
                                                                     optimizer_img,
                                                                     3,
                                                                     len(indices_class),
                                                                     self.args.syn_size,
                                                                     self.args.ipc,
                                                                     self.args.nps,
                                                                     args=self.args,
                                                                     loss_fn=self.mil_loss,)
                # report averaged loss
                dm_loss_avg += loss
                dm_loss_avg /= self.args.n_classes
                if it % 100 == 0:
                    self.logger.info('client = %2d, iter = %2d, loss = %.4f' % (client_idx, it, dm_loss_avg))
                    save_syn_img(image_syns[client_idx], self.args.results_dir, iter=it, client_idx=client_idx)
                    # save current synthetic images
                    torch.save(image_syns[client_idx].detach().cpu(), f'{self.args.results_dir}/client{client_idx}/{it}/synthetic_images.pt')
                    # check if current synthetic images are updated by comparing with the previous one

                pbar_dm.set_description(f'Client {client_idx} - DM loss {dm_loss_avg}')

            del images_all, labels_all

        ''' Save generated data '''
        self.logger.info(' Saving generated data to {}'.format(self.args.results_dir))
        save_syn_img(image_syns, self.args.results_dir, iter='final')
        ''' Prepare mixup vitual data '''
        self.logger.info('=========================Mixup vitual data=========================')
        global_virtual_images = [copy.deepcopy(image_syns[client_idx].detach().cpu()).to(self.device) for client_idx in range(len(self.n_clients))]
        global_virtual_labels = [copy.deepcopy(label_syns[client_idx].detach().cpu()).to(self.device) for client_idx in range(len(self.n_clients))]
        # mixup images (syn images)
        # compress raw patch into feature for combining with real data
        for client_idx in range(len(self.n_clients)):
            print('Client:', client_idx, global_virtual_images[client_idx].size())
        #TODO need check
        not_missing_label = check_sublists_equal_size(client_label_list)
        if not_missing_label:
            mixup_virtual_images = torch.mean(torch.stack(global_virtual_images), dim=0).detach().cpu()
            mixup_virtual_labels = global_virtual_labels[0].detach().cpu()
        else:
            global_virtual_images_tmp = [[] for _ in range(self.args.n_classes)]
            for i in range(len(client_label_list)):
                client_label = client_label_list[i]
                for c in client_label:
                    global_virtual_images_tmp[c].append(global_virtual_images[i][c*self.args.ipc: (c+1)*self.args.ipc])
            mixup_virtual_images = torch.cat([torch.mean(torch.stack(global_virtual_images_tmp[c]), dim=0) for c in range(self.args.n_classes)], dim=0)
            mixup_virtual_labels = global_virtual_labels[0].detach().cpu()

        mixup_train_set = TensorDataset(mixup_virtual_images, mixup_virtual_labels)
        shuffled_idx = list(range(0, len(mixup_train_set)))
        random.shuffle(shuffled_idx)
        shuffled_mixup_train_set = torch.utils.data.Subset(mixup_train_set, shuffled_idx[:len(mixup_train_set)])
        kd_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=1, shuffle=False,
                                                      num_workers=0)
        reg_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=1, shuffle=True,
                                                       num_workers=0)

        # concatenated train sets (real + syn)
        concated_train_sets = [torch.utils.data.ConcatDataset([train_dataset, mixup_train_set]) for train_dataset in
                               self.train_dataset]
        concated_train_loaders = [
            torch.utils.data.DataLoader(concated_train_set, batch_size=1, shuffle=True, num_workers=0) for
            concated_train_set in concated_train_sets]

        ''' Knowledge Distillation '''
        distance_loss = Distance_loss(device=self.device)
        # prepare model and optimizer
        clients_models_kd = []
        optimizers_kd = []
        MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']# more will be added
        for idx in range(len(self.n_clients)):
            if self.args.heter_model:
                seed = 33 + repeat + len(MIL_pool)
                random.seed(seed)
                self.args.mil_method = random.choice(MIL_pool)
                if len(MIL_pool) > 0:
                    MIL_pool.remove(self.args.mil_method)
                if len(MIL_pool) == 0:
                    MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']
                print(f'=====> Agent {idx} uses {self.args.mil_method} {self.args.opt}')
                clients_models_kd.append(define_model(self.args))
                if self.args.pretrain_kd:
                    pretrained_model_path = f'{self.args.results_dir}/client_{idx}_{clients_models_kd[idx].__class__.__name__}_pretrain.pt'
                    print(f'Loading pretrained {clients_models_kd[idx].__class__.__name__} from ', pretrained_model_path)
                    clients_models_kd[idx].load_state_dict(torch.load(pretrained_model_path))
                optimizers_kd.append(get_optim(self.args, clients_models_pre[idx]))
            else:
                clients_models_kd.append(define_model(self.args))
                if self.args.pretrain_kd:
                    pretrained_model_path = f'{self.args.results_dir}/client_{idx}_{clients_models_kd[idx].__class__.__name__}_pretrain.pt'
                    print(f'Loading pretrained {clients_models_kd[idx].__class__.__name__} from ',
                          pretrained_model_path)
                    clients_models_kd[idx].load_state_dict(torch.load(pretrained_model_path))
                optimizers_kd.append(get_optim(self.args, clients_models_pre[idx]))
        # clients_models_kd = [define_model(self.args) for _ in range(len(self.n_clients))]
        # if self.args.pretrain_kd:
        #     for client_idx in range(len(self.n_clients)):
        #         pretrained_model_path = f'{self.args.results_dir}/client_{client_idx}_{clients_models_pre[client_idx].__class__.__name__}_pretrain.pt'
        #         print(f'Loading pretrained {clients_models_kd[client_idx].__class__.__name__} from ', pretrained_model_path)
        #         clients_models_kd[client_idx].load_state_dict(torch.load(pretrained_model_path))
        # optimizers_kd = [get_optim(self.args, model) for model in clients_models_kd]
        kd_loss_fun = nn.KLDivLoss()
        self.logger.info('=========================Knowledge Distillation=========================')

        for kd_idx in range(self.args.kd_iters):
            # get averaged logits
            virtual_iter = iter(kd_train_loader)
            output_targets = [[] for _ in clients_models_kd]
            for step in range(len(virtual_iter)):
                x_virtual, y_virtual = next(virtual_iter)
                x_virtual = x_virtual.to(self.device).float()
                y_virtual = y_virtual.to(self.device).long()
                # print('X virtual size ', x_virtual.size(), y_virtual.size())
                with torch.no_grad():
                    # for i, teacher_model in enumerate(teacher_models):
                    for i in range(len(self.n_clients)):
                        teacher_model = clients_models_kd[i].to(self.device)
                        teacher_model.eval()
                        _, _, _, output_target_tmp = self.mil_run(teacher_model,
                                                                  x_virtual,
                                                                  y_virtual,
                                                                  self.mil_loss,
                                                                  return_lgt=True,
                                                                  raw_image=True)
                        output_targets[i].append(output_target_tmp)

            if (kd_idx + 1) % 10 == 0:
                print('----------')
            tr_mean_loss, reg_mean_loss, kd_mean_loss, tr_mean_acc, te_mean_loss, te_mean_acc = [], [], [], [], [], []
            for client_idx in range(len(self.n_clients)):
                clients_models_kd[client_idx].train()
                correct = 0
                loss_all = 0
                loss_ori_all = 0
                loss_kd_all = 0
                loss_reg_all = 0
                train_iter = iter(concated_train_loaders[client_idx])
                kd_iter = iter(kd_train_loader)
                kd_step = 0
                reg_iter = iter(reg_train_loader)
                for step in range(len(train_iter)):
                    # get classification loss
                    x, y = next(train_iter)
                    x = x.to(self.device).float()
                    y = y.to(self.device).long()
                    if len(x.size()) > 3:
                        use_raw = True
                    else:
                        use_raw = False

                    loss_ori, _, _, feature, output = self.mil_run(clients_models_kd[client_idx],
                                                      x.squeeze(0),
                                                      y,
                                                      self.mil_loss,
                                                      return_lgt=True,
                                                      return_feature=True,
                                                      raw_image=use_raw)
                    # get kd loss
                    try:
                        x_kd, y_kd = next(kd_iter)
                    except:
                        kd_iter = iter(kd_train_loader)
                        x_kd, y_kd = next(kd_iter)
                        kd_step = 0  # to make sure we get the correct logits from other clients
                    x_kd = x_kd.to(self.device).float()
                    y_kd = y_kd.to(self.device).long()
                    # _, kd_output = model(x_kd)

                    _, _, _, kd_output = self.mil_run(clients_models_kd[client_idx],
                                                      x_kd.squeeze(0),
                                                      y_kd,
                                                      self.mil_loss,
                                                      return_lgt=True,
                                                      raw_image=True)
                    teacher_logits = []
                    for i, logits in enumerate(output_targets):
                        if i != client_idx:
                            teacher_logits.append(logits[kd_step])
                    teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
                    loss_kd = calculate_kd_loss(kd_output, teacher_logits, y_kd, kd_loss_fun)
                    kd_step += 1

                    # get regularization loss
                    try:
                        x_reg, y_reg = next(reg_iter)
                    except:
                        reg_iter = iter(reg_train_loader)
                        x_reg, y_reg = next(reg_iter)
                    x_reg = x_reg.to(self.device).float()
                    y_reg = y_reg.to(self.device).long()
                    # reg_feature = model.embed(x_reg).detach()
                    _, _, _, reg_feature = self.mil_run(clients_models_kd[client_idx],
                                                      x_reg.squeeze(0),
                                                      y_reg,
                                                      self.mil_loss,
                                                      return_feature=True,
                                                      raw_image=True)
                    loss_reg = distance_loss(feature, reg_feature, y, y_reg)  # sup contrastive

                    loss = 1.0 * loss_ori + 1.0 * loss_kd + 1.0 * loss_reg
                    loss_kd_all += loss_kd.item()

                    optimizers_kd[client_idx].zero_grad()
                    loss.backward()
                    loss_all += loss.item()
                    loss_ori_all += loss_ori.item()
                    loss_reg_all += loss_reg.item()
                    optimizers_kd[client_idx].step()

                    pred = output.data.max(1)[1]
                    correct += pred.eq(y.view(-1)).sum().item()
                test_loader = self.get_test_loader(self.test_dataset[client_idx])
                test_loss, test_error, fpr, tpr = self.local_test(clients_models_kd[client_idx], test_loader)
                if (kd_idx + 1) % 10 == 0:
                    self.logger.info(
                        'Epoch {}: KD Train|  Client {} - Loss: {:4f}; Ori Loss: {:4f}; KD Loss: {:4f}; Reg Loss: {:4f}; Acc: {:4f}'.format(
                            kd_idx, client_idx, loss_all, loss_ori_all, loss_kd_all, loss_reg_all, correct / len(train_iter)))
                    self.logger.info('Epoch {}: KD Test |  Client {} - Loss: {:4f}; Acc: {:4f}'.format(kd_idx, client_idx, test_loss, 1-test_error))

        ''' Save checkpoint '''
        for client_idx in range(len(self.n_clients)):
            model_path = f'{self.args.results_dir}/client{client_idx}_iterative_kd_CLAM_model.pt'
            self.logger.info(' Saving checkpoints to {}...'.format(model_path))
            torch.save(clients_models_kd[client_idx].state_dict(), model_path)

        ''' Final testing '''
        best_acc_per_agent = []
        for client_idx in range(len(self.n_clients)):
            test_loader = self.get_test_loader(self.test_dataset[client_idx])
            pre_test_loss, pre_test_error, pre_fpr, pre_tpr =self.local_test(clients_models_pre[client_idx], test_loader)
            kd_test_loss, kd_test_error, kd_fpr, kd_tpr = self.local_test(clients_models_kd[client_idx], test_loader)
            pre_test_acc = 1 - pre_test_error
            kd_test_acc = 1 - kd_test_error
            self.logger.info(f'Client {client_idx} with CLAM')
            self.logger.info('PRE Test |  Loss: {:4f}; Acc: {:4f}'.format(pre_test_loss, pre_test_acc))
            self.logger.info('KD  Test |  Loss: {:4f}; Acc: {:4f}'.format(kd_test_loss, kd_test_acc))
            agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{client_idx}_iter_{iter}_fpr.npy')
            agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{client_idx}_iter_{iter}_tpr.npy')
            np.save(agent_fpr_save_pth, kd_fpr)
            np.save(agent_tpr_save_pth, kd_tpr)

            if pre_test_acc > kd_test_acc:
                best_acc_per_agent.append(pre_test_acc)
            else:
                best_acc_per_agent.append(kd_test_acc)
        best_acc_overall = np.mean(best_acc_per_agent)
        list_acc_wt = [0] * len(self.n_clients)
        for i in range(len(self.n_clients)):
            list_acc_wt[i] = best_acc_per_agent[i] * self.weight_list[i]
        train_acc_wt = sum(list_acc_wt)
        return best_acc_overall, train_acc_wt, best_acc_per_agent