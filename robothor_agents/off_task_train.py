import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import matplotlib.pyplot as plt

from robothor_agents.agent import OffTaskModel

def load_dataset(path, dir_list):
    
    sample_idx = torch.randint(0, len(dir_list), (1,))

    data = torch.load(os.path.join(path,dir_list[sample_idx]))

    return data

    
def get_accuracy(y_tgt, y_pred):

    # decision boundary at 0.5
    y_pred = y_pred > 0.5

    tp = 0
    fp = 0
    tn = 0
    fn =  0

    for sample in range(y_tgt.shape[0]):
        
        tp += np.sum([elem1 and elem2 for elem1, elem2 in zip(y_tgt[sample], y_pred[sample])])
        tn += np.sum([not(elem1) and not(elem2) for elem1, elem2 in zip(y_tgt[sample], y_pred[sample])])
        fn += np.sum([elem1 and not(elem2) for elem1, elem2 in zip(y_tgt[sample], y_pred[sample])])
        fp += np.sum([not(elem1) and elem2 for elem1, elem2 in zip(y_tgt[sample], y_pred[sample])])


    print(tp, tn, fn, fp)
    accuracy = (tp+tn) / (tp+tn+fn+fp)

    # compute recall and precision backwards (often denominator would be 0 from perspective of positives)
    recall = (tn) / (tn+fp)

    precision = (tn) / (tn+fn)

    return accuracy, recall, precision

def train(): 

    my_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/")
    dir_list = os.listdir(my_path)


    ## hyperparameters
    max_epochs = 1000
    lr = 3e-4
    batch_size = 32
    disp_it = 10

    #weight different loss contributions to start out at the same order of magnitude
    w_depth, w_auto, w_seg, w_class = 0.25/12, 0.25/2.4e4, 0.25/2.4e4, 0.25/2 

    # segmentation, depth, and autoencoder tasks all use MSE loss
    loss_fn_depth = nn.MSELoss()
    # classification (mult-class) with binary cross entropy
    loss_fn_class = nn.BCELoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    off_task_model = OffTaskModel()
    
    if(0):
        model_fn = "./temp_off_task_model.pt"
        off_task_model.load_state_dict(torch.load(model_fn))
        import pdb; pdb.set_trace()
    
    optimizer = torch.optim.Adam(off_task_model.parameters(), lr=lr)
    
    t0 = time.time()
    
    off_task_model.to(device)

    for epoch in range(max_epochs):

        dataset = load_dataset(my_path, dir_list)

        for key in dataset.keys():
            if "info" not in key:
                dataset[key] = dataset[key].to(torch.float32)

        epoch_size = dataset["l_next_obs_x"].shape[0]
        smooth_loss = 0.0
        smooth_losses = [0.0] * 4
        t1 = time.time()
        for batch_idx in range(0, epoch_size-batch_size, batch_size):

            t2 = time.time()

            off_task_model.zero_grad()

            x = dataset["l_next_obs_x"][batch_idx:batch_idx+batch_size].to(device)
            y_tgt_d = dataset["l_depth_frame"][batch_idx:batch_idx+batch_size].to(device)
            y_tgt_s = dataset["l_class_frame"][batch_idx:batch_idx+batch_size].to(device)
            y_tgt_c = dataset["l_class_labels"][batch_idx:batch_idx+batch_size].to(device)
            
            y_depth, y_auto, y_seg, y_class = off_task_model(x)

            loss_depth = w_depth * loss_fn_depth(y_depth.squeeze(), y_tgt_d)
            loss_auto = w_auto * loss_fn_depth(y_auto, x)
            loss_seg = w_seg * loss_fn_depth(y_seg, y_tgt_s)
            loss_class = w_class * loss_fn_class(y_class, y_tgt_c)

            loss = loss_depth + loss_auto + loss_seg + loss_class
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                smooth_loss = smooth_loss * 0.9 + 0.1 * loss
                smooth_losses = [loss_res*0.1 + smooth_res \
                        for loss_res, smooth_res in \
                        zip([loss_depth, loss_auto, loss_seg, loss_class],smooth_losses)]

            t3 = time.time()


        if epoch % disp_it == 0:

            acc, recall, prec = get_accuracy(y_tgt_c, y_class)
            print("training sample accuracy/recall/precision = {:.3f}/{:.3f}/{:.3f}".format(acc,recall,prec))
            print("smooth loss at epoch {}: {:.3e}, time elapsed total: {:.2e} epoch: {:.2e}"\
                    .format(epoch, smooth_loss, t3-t0, t3-t1))
            print("depth, auto, seg, classification smooth losses: \n", smooth_losses)

            model_fn = "./temp_off_task_model.pt"
            print("saving model to {}".format(model_fn))

            torch.save(off_task_model.state_dict(), model_fn)

    import pdb; pdb.set_trace()
        
if __name__ == "__main__":
    train()

