import os
import logging
import numpy as np
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.trainer_util import calculate_error, compute_accuracy, RAdam, FeatMag, average_weights, get_mdl_params, set_client_from_params
from utils.data_utils import get_split_loader, CategoriesSampler
from utils.Get_model import define_model
from utils.Get_data import define_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Meta(nn.Module):
    def __init__(self, args, logger=None):
        super(Meta, self).__init__()
        self.args = args
        self.global_model = define_model(args)
        self.logger = logger
        self.logger.info(' '.join(f'--{k}={v} \n' for k, v in vars(args).items()))
        self.device = device

    def get_optim(self, model, alpha=None):
        params_gather = []
        mommen = self.args.reg if alpha is None else alpha + self.args.reg
        params_gather.append(
            {'params': filter(lambda p: p.requires_grad, model.parameters()),
             'lr': self.args.lr,
             'weight_decay': mommen}
        )
        if self.args.opt == "adam":
            optimizer = optim.Adam(params_gather)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(params_gather)
        elif self.args.opt == 'sgd':

            optimizer = optim.SGD(params_gather, momentum=mommen, nesterov=True)
        elif self.args.opt == 'radam':
            optimizer = RAdam(params_gather)
        else:
            raise NotImplementedError
        return optimizer

    def get_loss(self):
        if self.args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes=self.args.n_classes)
            loss_fn = loss_fn.cuda()
        elif self.args.bag_loss == 'mag':
            loss_fn = FeatMag(margin=self.args.mag).cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def clam_runner(self, model, data, label, loss_fn):
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.args.bag_weight * loss + (1 - self.args.bag_weight) * instance_loss
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def hipt_runner(self, model, data, label, loss_fn):
        data = data.unsqueeze(0)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def transmil_runner(self, model, data, label, loss_fn):
        data = data.unsqueeze(0)
        results_dict = model(data=data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def abmil_runner(self, model, data, label, loss_fn):
        logits, Y_prob, Y_hat, A = model.forward(data)
        total_loss = loss_fn(logits, label)
        error = 1. - Y_hat.eq(label).cpu().float().mean().item()
        return total_loss, error

    def frmil_runner(self, model, data, label, loss_fn, bce_weight, ce_weight):
        norm_idx = torch.where(label.cpu() == 0)[0].numpy()[0]
        ano_idx = 1 - norm_idx
        if self.args.drop_data:
            data = F.dropout(data, p=0.20)
        logits, query, max_c = model(data)

        # all losses
        max_c = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, label.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, label, weight=ce_weight)
        loss_ft = loss_fn(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                           w_scale=query.shape[1])
        loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
        acc = compute_accuracy(logits, label)

        return loss, 1 - acc/100

    def get_train_loader(self, train_dataset):
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(train_dataset.labels,
                                              n_batch=len(train_dataset.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(train_dataset, training=True, weighted=self.args.weighted_sample)
        return train_loader

    def get_test_loader(self, test_dataset):
        if 'frmil' in self.args.mil_method:
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            test_loader = get_split_loader(test_dataset)
        return test_loader

    def local_update(self, agent_idx, train_dataset, model, loss_fn):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                if self.args.fed_method == 'fed_prox':
                    proximal_loss = (self.args.mu/2) * sum((w - v).norm(2) for w, v in zip(model.parameters(), self.global_model.parameters()))
                    loss += proximal_loss
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(local_train_loader)
            batch_error /= len(local_train_loader)
            epoch_loss += batch_loss
        return model.state_dict(), epoch_loss/self.args.local_epochs

    def local_inference(self, agent_idx, test_dataset, model, loss_fn):
        local_test_dataset = test_dataset[agent_idx]
        local_test_loader = self.get_test_loader(local_test_dataset)

        model.to(device)
        model.eval()
        total_loss = 0.
        total_error = 0.
        for batch_idx, (images, labels) in enumerate(local_test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            if 'CLAM' in self.args.mil_method:
                loss, error = self.clam_runner(model, images, labels, loss_fn)
            else:
                self.logger.error(f'{self.args.mil_method} not implemented')
                raise NotImplementedError

            total_loss += loss.item()
            total_error += error
        total_loss /= len(local_test_loader)
        total_error /= len(local_test_loader)
        return total_loss, total_error

    def forward_fedavg(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        self.global_model.to(device)
        self.global_model.train()

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses = [], []
            # print(f'\n | Global Training Round : {epoch + 1} |\n')
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()

            for idx in agents:
                w, agent_loss = self.local_update(idx,
                                                  train_dataset,
                                                  deepcopy(self.global_model),
                                                  loss_fn)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))

            # update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               self.global_model,
                                               loss_fn)
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(self.global_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
        return best_accuracy, best_accuracy_per_agent

    def local_update_feddyn(self, agent_idx,
                            train_dataset,
                            model, loss_fn,
                            alpha_coef_adpt,
                            cld_mdl_param_tensor,
                            local_param_list_curr):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model, alpha=alpha_coef_adpt)

        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                loss_algo = alpha_coef_adpt * torch.sum(
                    local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                # current_local_parameter * (last_step_local_parameter - global_parameter)
                loss = loss + loss_algo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)  # Clip gradients
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                batch_loss /= len(local_train_loader)
            epoch_loss += batch_loss
            model.train()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model, epoch_loss/self.args.local_epochs

    def local_update_fedscaf(self, agent_idx,
                            train_dataset,
                            model, loss_fn,
                            state_params_diff_curr):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError

                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * state_params_diff_curr)
                loss = loss + loss_algo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)  # Clip gradients
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                batch_loss /= len(local_train_loader)
            epoch_loss += batch_loss
            model.train()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model, epoch_loss / self.args.local_epochs

    def forward_feddyn(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        n_clnt = len(train_dataset)
        weight_list = np.asarray([len(train_dataset[i]) for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list) * n_clnt

        n_par = len(get_mdl_params([self.global_model])[0])
        local_param_list = np.zeros((n_clnt, n_par)).astype('float32') # [n_clnt X n_par]
        init_par_list = get_mdl_params([self.global_model], n_par)[0]
        clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                        -1)  # [n_clnt X n_par]
        clnt_models = list(range(n_clnt))
        avg_model = deepcopy(self.global_model).to(device)
        cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_losses = []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) # [n_par]
            for idx in agents:
                # Warm start from current avg model
                clnt_models[idx] = deepcopy(avg_model).to(device)
                model = clnt_models[idx]
                for params in model.parameters():
                    params.requires_grad = True

                alpha_coef_adpt = self.args.alpha_coef / weight_list[idx]  # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[idx], dtype=torch.float32, device=device)
                local_trained_model, agent_loss = self.local_update_feddyn(idx,
                                                                            train_dataset,
                                                                            model,
                                                                            loss_fn,
                                                                            alpha_coef_adpt,
                                                                            cld_mdl_param_tensor,
                                                                            local_param_list_curr)
                clnt_models[idx] = local_trained_model
                curr_model_par = get_mdl_params([clnt_models[idx]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[idx] += curr_model_par-cld_mdl_param
                clnt_params_list[idx] = curr_model_par

                local_losses.append(agent_loss)

            avg_mdl_param = np.mean(clnt_params_list, axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
            avg_model = set_client_from_params(self.global_model, avg_mdl_param)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            avg_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               avg_model,
                                               loss_fn)
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(avg_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
        return best_accuracy, best_accuracy_per_agent

    def forward_fedscaf(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        n_clnt = len(train_dataset)
        weight_list = np.asarray([len(train_dataset[i]) for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list) * n_clnt

        n_par = len(get_mdl_params([self.global_model])[0])
        state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
        init_par_list = get_mdl_params([self.global_model], n_par)[0]
        clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_clnt X n_par
        clnt_models = list(range(n_clnt))
        avg_model = deepcopy(self.global_model).to(device)

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_losses = []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for idx in agents:
                # Warm start from current avg model
                clnt_models[idx] = deepcopy(avg_model).to(device)
                model = clnt_models[idx]
                for params in model.parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(
                    -state_param_list[idx] + state_param_list[-1] / weight_list[idx], dtype=torch.float32,
                    device=device)

                local_trained_model, agent_loss = self.local_update_fedscaf(idx,
                                                                            train_dataset,
                                                                            model,
                                                                            loss_fn,
                                                                            state_params_diff_curr)
                clnt_models[idx] = local_trained_model
                curr_model_par = get_mdl_params([clnt_models[idx]], n_par)[0]

                new_c = state_param_list[idx] - state_param_list[-1] + 1 / self.args.global_epochs / self.args.lr * (
                            prev_params - curr_model_par)
                delta_c_sum += (new_c - state_param_list[idx]) * weight_list[idx]
                state_param_list[idx] = new_c
                clnt_params_list[idx] = curr_model_par
                local_losses.append(agent_loss)

            avg_model_params = np.mean(clnt_params_list, axis=0)
            state_param_list[-1] += 1 / n_clnt * delta_c_sum
            avg_model = set_client_from_params(self.global_model, avg_model_params)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            avg_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               avg_model,
                                               loss_fn)
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(avg_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
        return best_accuracy, best_accuracy_per_agent