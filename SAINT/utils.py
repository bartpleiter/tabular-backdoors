import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from .augmentations import embed_data_mask
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        try:
            auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
        except:  # in case we only have class in our test set (like for ASR)
            auc = 0.0
    return acc.cpu().numpy(), auc

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse


def prepareData(train, valid, test, test_backdoor, cat_cols, num_cols, target):
    X_train_df = train.drop(target, axis=1)
    y_train_df = train[target].values
    
    X_valid_df = valid.drop(target, axis=1)
    y_valid_df = valid[target].values
    
    X_test_df = test.drop(target, axis=1)
    y_test_df = test[target].values
    
    X_test_backdoor_df = test_backdoor.drop(target, axis=1)
    y_test_backdoor_df = test_backdoor[target].values

    combined_df = pd.concat([X_train_df, X_valid_df, X_test_df])

    cat_dims = []
    for col in cat_cols:
        l_enc = LabelEncoder() 
        l_enc.fit(combined_df[col].values)
        X_train_df[col] = l_enc.transform(X_train_df[col].values)
        X_valid_df[col] = l_enc.transform(X_valid_df[col].values)
        X_test_df[col] = l_enc.transform(X_test_df[col].values)
        X_test_backdoor_df[col] = l_enc.transform(X_test_backdoor_df[col].values)
        cat_dims.append(len(l_enc.classes_))
        
    cat_idxs = [X_train_df.columns.get_loc(c) for c in cat_cols]
    con_idxs = [X_train_df.columns.get_loc(c) for c in num_cols]
    
    X_train = {
    'data': X_train_df.to_numpy(dtype="float64"),
    'mask': np.full(X_train_df.shape, 1)
    }
    y_train = {
        'data': y_train_df.reshape(-1, 1)
    }
    
    X_valid = {
    'data': X_valid_df.to_numpy(dtype="float64"),
    'mask': np.full(X_valid_df.shape, 1)
    }
    y_valid = {
        'data': y_valid_df.reshape(-1, 1)
    }
    
    X_test = {
    'data': X_test_df.to_numpy(dtype="float64"),
    'mask': np.full(X_test_df.shape, 1)
    }
    y_test = {
        'data': y_test_df.reshape(-1, 1)
    }
    
    X_test_backdoor = {
    'data': X_test_backdoor_df.to_numpy(dtype="float64"),
    'mask': np.full(X_test_backdoor_df.shape, 1)
    }
    y_test_backdoor = {
        'data': y_test_backdoor_df.reshape(-1, 1)
    }
    
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    return (cat_dims, cat_idxs, con_idxs, 
        X_train, y_train, X_valid, y_valid, 
        X_test, y_test, X_test_backdoor, y_test_backdoor, 
        train_mean, train_std)