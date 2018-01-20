
        #for i in range(num_epochs):
                          
        #    for k, (images, labels) in enumerate(train_loader,1):              
         #       images = Variable(images)
         #       labels = Variable(labels)
         #         # labels = labels.type()
         #          #data = (images , labels)
         #       optim.zero_grad()
         #           
         #       output = model(images)
         #       loss_t = self.loss_func(output, labels)
         #       loss_t.backward()
         #       optim.step()                 
         #           #print(i, j,
         #           #    "loss = ",loss_t.data[0]) 
         #       if k % log_nth == 0:
         #           self.val_loss_history.append(loss_t.data[0])      
         #                      
         #       correct = 0
         ##      total = 0         
         #      _, predicted = torch.max(output.data, 1)
         #      total += labels.size(0)
         #       correct += (predicted == labels.data.long()).sum()
         #       acc = correct / total
         #       #print("train acc = " ,acc)
         #       self.train_acc_history.append(acc)
              
                #print("validation"
         #   for q,  (images1, labels1)  in enumerate(val_loader,1):
         #       images1 = Variable(images1)
         #       labels1 = Variable(labels1)
         #           #labels1 = labels1.type(torch.LongTensor)
         #           #data1 = (images, labels)
         #       optim.zero_grad()
         #       output1 = model(images1)
         #       loss_v = self.loss_func(output1, labels1)
         #       loss_v.backward()
         #       optim.step()
         #           #
                    #print(i, j, 
                    #    "loss = " ,loss_v.data[0])
          #      if q % log_nth == 0:
          #          self.val_loss_history.append(loss_v.data[0])                   
                
           #     correct1 = 0
           #     total1 = 0 
            
            #    _, predicted1 = torch.max(output1.data, 1)
            #    total1 += labels1.size(0)
             #   correct1 += (predicted1 == labels1.data.long()).sum()
             #   acc1 = correct1 / total1
            #    print("val acc = " ,acc1)
           #     self.val_acc_history.append(acc1)
                
             #print(j,k,loss)
                 
                # for q, (images, labels) in enumerate(val_loader):
                 #       images = Variable(images)
                  #      labels = Variable(labels)
                   #     optim.zero_grad()
                   #     output = model(images)
                   #     loss_v = self.loss_func(output, labels)
                   #     loss_v.backward()
                   #     optim.step()
                    
              
        
                

           
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
       
        
        
        
from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        params = [param for param in model.parameters() if param.requires_grad]

        optim = self.optim(params, **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        for epoch in range(num_epochs):
            num_correct = 0
            num_samples = 0
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            model.train()
            for t, (x, y) in enumerate(train_loader):

                x_var = Variable(x)
                y_var = Variable(y)

                scores = model(x_var)
                loss = self.loss_func(scores, y_var)

                if (t + 1) % 100 == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

                model.eval()
                _, preds = scores.data.cpu().max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                model.train()

                optim.zero_grad()
                loss.backward()
                optim.step()

            acc = float(num_correct) / num_samples
            print('Training acc for epoch %d is %.3f' %(epoch+1,acc))
            check_accuracy(model, val_loader)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')