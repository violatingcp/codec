import h5py 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import VICRegLoss,CorrLoss,CorrLoss2
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader, SubsetRandomSampler, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

class simple_Edge(torch.nn.Module):
    def __init__(self,input_size,out_channels=1,act_out=True,nhidden=16,batchnorm=False):
        super().__init__()

        self.input_feat = input_size
        self.part_encode = nn.Sequential(
            nn.Linear(input_size, nhidden, bias=False),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU()
            #nn.Linear(nhidden, out_channels)
            #nn.Sigmoid()
        )
        self.part_encode.cuda()
        self.conv = DynamicEdgeConv(nn=nn.Sequential(nn.Linear(2*nhidden, nhidden), nn.ELU()),
                                    k=8
        )
        self.conv.cuda()
        self.output = nn.Sequential(
            nn.Linear(nhidden, nhidden//2),
            nn.ELU(),
            nn.Linear(nhidden//2, out_channels),
        )
        self.output.cuda()
        self.runbatchnorm = batchnorm
        self.batchnorm    = torch.nn.BatchNorm1d(out_channels)
        self.output_act  = torch.nn.Sigmoid()
        self.act_out = act_out 

    def forward(self, x):
        parts=(x.shape[1])//self.input_feat
        x_batch = torch.arange(0,x.shape[0])
        x_batch = x_batch.repeat_interleave(parts)
        x_batch = x_batch.cuda()
        x_part     = x.reshape((parts*x.shape[0],self.input_feat))
        x_part_enc = self.part_encode(x_part)
        #feats1     = self.conv(x=x_part_enc)#, batch=batch_part)
        feats1     = self.conv(x=(x_part_enc,x_part_enc), batch=(x_batch,x_batch))
        batch      = x_batch
        out, batch = avg_pool_x(batch, feats1, batch)
        out = self.output(out)
        if self.runbatchnorm:
            out = self.batchnorm(out)
        if self.act_out:
            out = self.output_act(out)
        return out

class simple_MLP(torch.nn.Module):
    def __init__(self,input_size,out_channels=1,act_out=True,nhidden=64,batchnorm=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, nhidden, bias=False),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels)
            #nn.Sigmoid()
        )
        self.runbatchnorm = batchnorm
        self.batchnorm    = torch.nn.BatchNorm1d(out_channels)
        self.output  = torch.nn.Sigmoid()
        self.act_out = act_out 

    def forward(self, x):
        x = self.model(x)        
        if self.runbatchnorm:
            x = self.batchnorm(x)
        if self.act_out:
            x = self.output(x)
        return x

class simple_MLP_onelayer(torch.nn.Module):
    def __init__(self,input_size,out_channels=1,act_out=True,nhidden=64,batchnorm=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels),
        )
        self.output  = torch.nn.Sigmoid()
        self.act_out = act_out 

    def forward(self, x):
        x = self.model(x)        
        if self.act_out:
            x = self.output(x)
        return x
        
class simple_model(LightningModule):
    def __init__(self,input_size,dataset,batch_size=1000,out_channels=1,nhidden=128,batchnorm=True):
        super().__init__()
        self.model = simple_MLP(input_size,out_channels)
        ntotal = len(dataset)
        self.batch_size = batch_size
        self.dataset_train, self.dataset_test = random_split(dataset,[int(0.8*ntotal),ntotal-int(0.8*ntotal)])
        self.val_accuracy  = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)        
        return x

    def loss(self, x, y):
        loss = F.mse_loss(x, y)
        #loss = F.cross_entropy(x,y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        return self.loss(y_hat,y)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss   = self.loss(logits,y)
        preds  = torch.argmax(logits, dim=1)
        self.val_accuracy.update(y, preds)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss   = self.loss(logits,y)
        preds = torch.argmax(logits, dim=1)
        self.test_scores = logits.cpu()
        self.test_labels = y.cpu()
        self.test_accuracy.update(y, preds)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            ntotal = len(self.dataset_train)
            self.data_train, self.data_val = random_split(self.dataset_train, [int(0.9*ntotal),ntotal-int(0.9*ntotal)])
        if stage == "test" or stage is None:
            self.data_test = self.dataset_test

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,num_workers=16)

class codec_model():
    def __init__(self,input_size,dataset_sig,dataset_bkg,inspace=2,iCLR=1,iVar=1,iCorr=1,cor_nspace=[], iCorr1=0, acor_nspace=[], iCorr2=0,batch_size=250,n_epochs=20,n_epochs_mse=20):
        super().__init__()
        #self.model1      = simple_MLP(input_size,inspace)
        self.model1      = simple_Edge(input_size,inspace)
        self.model2      = simple_MLP_onelayer(inspace,out_channels=1)
        self.model2_base = simple_MLP(input_size,out_channels=1)
        self.model1.cuda(); self.model2.cuda(); self.model2_base.cuda()
        self.ctr_loss   = VICRegLoss(lambda_param=1.0,mu_param=1.0,nu_param=1.0)
        self.cor_loss   = CorrLoss2()
        self.acor_loss  = CorrLoss2(anti=True)
        self.mse_loss   = nn.MSELoss()
        self.clr_const  = iCLR
        self.var_const  = iVar
        self.corr_const = iCorr
        self.corr_ax_const  = iCorr1
        self.acorr_ax_const = iCorr2
        self.batch_size = batch_size
        self.n_epochs     = n_epochs
        self.n_epochs_mse = n_epochs_mse
        self.cor_nspace   = cor_nspace
        self.acor_nspace  = acor_nspace
        self.opt1       = torch.optim.Adam(self.model1.parameters(),lr=0.005)
        self.opt2       = torch.optim.Adam(self.model2.parameters(),lr=0.005)
        self.opt_base   = torch.optim.Adam(self.model2_base.parameters(),lr=0.005)
        self.dataset_sig = dataset_sig
        self.dataset_bkg = dataset_bkg
        self.sig_train, self.sig_test = random_split(dataset_sig,[int(0.8*len(dataset_sig)),len(dataset_sig)-int(0.8*len(dataset_sig))])
        self.bkg_train, self.bkg_test = random_split(dataset_bkg,[int(0.8*len(dataset_bkg)),len(dataset_bkg)-int(0.8*len(dataset_bkg))])

    def forward(self, x):
        x = self.model1(x)        
        return x

    def corr_loss(self, x1, x2, m1, m2):
        loss_corr1 = 0
        for idim in self.cor_nspace:
            loss_corr1   += self.cor_loss(x1[:,idim:(idim+1)],m1)+self.cor_loss(x2[:,idim:(idim+1)],m2)
        return loss_corr1 

    def acorr_loss(self, x1, x2, m1, m2):
        loss_corr2 = 0
        for idim in self.acor_nspace:
            loss_corr2   += self.acor_loss(x1[:,idim:(idim+1)],m1)+self.acor_loss(x2[:,idim:(idim+1)],m2)
        return loss_corr2 
        
    def training_loss(self, x1, x2, m1, m2):
        x1_hat = self.model1(x1)
        x2_hat = self.model1(x2)
        loss_clr,loss_corr,loss_var = self.ctr_loss(x1_hat,x2_hat)
        loss_corr1 = self.corr_loss (x1_hat,x2_hat,m1,m2) 
        loss_corr2 = self.acorr_loss(x1_hat,x2_hat,m1,m2) 
        return self.clr_const*loss_clr + self.var_const*loss_var + self.corr_const*loss_corr+self.corr_ax_const*loss_corr1+self.acorr_ax_const*loss_corr2, self.clr_const*loss_clr,self.var_const*loss_var,self.corr_const*loss_corr,self.corr_ax_const*loss_corr1,self.acorr_ax_const*loss_corr2  

    def training_clr_epoch(self):
        running_loss = 0.0; running_loss_clr = 0.0; running_loss_var = 0.0; running_loss_corr = 0.0; running_loss_corr1 = 0.0; running_loss_corr2 = 0.0
        train_loader1,train_loader2 = self.train_dataloader()
        x1=None; m1=None
        updates=0
        for batch_idx, ((xp, _ , mp) , (xn, _ , mn))  in enumerate(zip(train_loader1,train_loader2)):
            xp=xp.cuda(); xn=xn.cuda(); mp=mp.cuda(); mn=mn.cuda()
            _,indp = torch.sort(mp.flatten())
            _,indn = torch.sort(mn.flatten())
            x2=torch.cat((xp[indp],xn[indn])); m2=torch.cat((mp[indp],mn[indn]))
            #x2=torch.cat((xp,xn)); m2=torch.cat((mp,mn))
            if (batch_idx+1) % 2 == 0:
                if len(x1) == len(x2):
                    self.opt1.zero_grad()
                    loss,loss_clr,loss_var,loss_corr,loss_corr1,loss_corr2 = self.training_loss(x1,x2,m1,m2)
                    loss.backward()
                    self.opt1.step()
                    running_loss += loss; running_loss_clr += loss_clr; running_loss_var += loss_var; running_loss_corr += loss_corr; running_loss_corr1 += loss_corr1; running_loss_corr2 += loss_corr2
                    updates = updates + 1
            else:
                x1=x2
                m1=m2
        return running_loss/updates, running_loss_clr/updates, running_loss_var/updates, running_loss_corr/updates, running_loss_corr1/updates, running_loss_corr2/updates

    def validate_clr_epoch(self):
        running_loss = 0.0
        val_loader1,val_loader2 = self.val_dataloader()
        x1=None
        m1=None
        updates=0
        for batch_idx, ((xp, _ , mp) , (xn, _ , mn))  in enumerate(zip(val_loader1,val_loader2)):
            xp=xp.cuda(); xn=xn.cuda(); mp=mp.cuda(); mn=mn.cuda()
            _,indp = torch.sort(mp.flatten())
            _,indn = torch.sort(mn.flatten())
            x2=torch.cat((xp[indp],xn[indn])); m2=torch.cat((mp[indp],mn[indn]))
            if (batch_idx + 1) % 2 == 0:
                if len(x1) == len(x2):
                    loss,_,_,_,_,_ = self.training_loss(x1,x2,m1,m2)
                    running_loss += loss 
                    updates = updates + 1
            else:
                x1=x2
                m1=m2
        return running_loss/updates

    def training_mse_epoch(self,imodel,iloader, iopt):
        running_loss = 0.0
        train_loader = iloader
        updates=0
        for batch_idx, (x, y , _) in enumerate(train_loader):
            iopt.zero_grad()
            x     = x.cuda(); y = y.cuda()
            x_out = imodel(x)
            loss  = self.mse_loss(x_out, y)
            loss.backward()
            iopt.step()
            running_loss += loss 
            updates = updates+1
        return running_loss/updates

    def validate_mse_epoch(self,imodel,iloader):
        running_loss = 0.0
        val_loader   = iloader
        updates=0
        for batch_idx, (x, y, _)   in enumerate(val_loader):
            x     = x.cuda(); y = y.cuda()
            x_out = imodel(x) 
            loss  = self.mse_loss(x_out, y)
            running_loss += loss 
            updates = updates+1
        return running_loss/updates

    def test_all(self):
        test_loader   = self.test_dataloader()
        scores_out = np.array([])
        labels_out = np.array([])
        mass_out   = np.array([])
        space_out  = np.array([])
        for batch_idx, (x, y, m)   in enumerate(test_loader):
            x = x.cuda() 
            x1_out = self.model1(x) 
            x2_out = self.model2(x1_out) 
            scores_out = np.append(scores_out,x2_out.cpu().detach().numpy())
            labels_out = np.append(labels_out,y)
            mass_out   = np.append(mass_out,m)
            space_out  = np.append(space_out,x1_out.cpu().detach().numpy())
        return scores_out,labels_out,mass_out,space_out

    def test_base(self):
        test_loader   = self.test_dataloader()
        scores_out = np.array([])
        labels_out = np.array([])
        mass_out   = np.array([])
        for batch_idx, (x, y, m)   in enumerate(test_loader):
            x = x.cuda() 
            x_out = self.model2_base(x) 
            scores_out = np.append(scores_out,x_out.cpu().detach().numpy())
            labels_out = np.append(labels_out,y)
            mass_out   = np.append(mass_out,m)
        return scores_out,labels_out,mass_out

    def training_clr(self):
        for epoch in range(self.n_epochs):
            self.model1.train(True)
            loss_train,loss_train_clr,loss_train_var,loss_train_corr, loss_train_corr1, loss_train_corr2 = self.training_clr_epoch()
            self.model1.train(False)
            loss_valid = self.validate_clr_epoch()
            print('Epoch: {} LOSS train: {:.4f} valid: {:.4f} clr: {:.4f} var: {:.4f} cor: {:.4f} corr1: {:.4f} corr2: {:.4f}'.format(epoch,loss_train,loss_valid,loss_train_clr,loss_train_var,loss_train_corr,loss_train_corr1,loss_train_corr2))
    
    def training_mse(self):
        for epoch in range(self.n_epochs_mse):
            self.model2.train(True)
            loss_train = self.training_mse_epoch(self.model2,self.train_dataloader_mse(),self.opt2)
            self.model2.train(False)
            loss_valid = self.validate_mse_epoch(self.model2,self.val_dataloader_mse())
            print('Epoch: {} LOSS train: {} valid: {}'.format(epoch,loss_train,loss_valid))

    def training_base_mse(self):
        for epoch in range(self.n_epochs_mse):
            self.model2_base.train(True)
            loss_train = self.training_mse_epoch(self.model2_base,self.train_dataloader_base_mse(),self.opt_base)
            self.model2_base.train(False)
            loss_valid = self.validate_mse_epoch(self.model2_base,self.val_dataloader_base_mse())
            print('Epoch: {} LOSS train: {} valid: {}'.format(epoch,loss_train,loss_valid))
        
    def setup(self, stage=None):
        nsig = len(self.sig_train); nbkg = len(self.bkg_train)
        self.sdata_train, self.sdata_val = random_split(self.sig_train, [int(0.9*nsig),nsig-int(0.9*nsig)])
        self.bdata_train, self.bdata_val = random_split(self.bkg_train, [int(0.9*nbkg),nbkg-int(0.9*nbkg)])
        self.data_test = torch.utils.data.ConcatDataset([self.sig_test,self.bkg_test])

    def setup_mse(self, stage=None):
        train_loader1,train_loader2 = self.fulltrain_dataloader()
        with torch.no_grad():
            for batch_idx, (xsig, ysig, msig)   in enumerate(train_loader1):
                xsig = xsig.cuda()
                sig_train_mse_out = self.model1(xsig)
            for batch_idx, (xbkg, ybkg, mbkg)   in enumerate(train_loader2):
                xbkg = xbkg.cuda()
                bkg_train_mse_out = self.model1(xbkg)
            train_mse_out     = torch.cat((sig_train_mse_out.cpu(),bkg_train_mse_out.cpu()))
            train_mass_out    = torch.cat((msig,mbkg))
            train_label_out   = torch.cat((ysig,ybkg))
            randind           = torch.randperm(len(train_mse_out))
            train_mse_out     = train_mse_out[randind]
            train_mass_out    = train_mass_out[randind]
            train_label_out   = train_label_out[randind]
            train_mse         = DataSet(samples=train_mse_out,labels=train_label_out,masses=train_mass_out)
            ntot              = len(train_mse)
            self.data_train_mse, self.val_train_mse = random_split(train_mse, [int(0.9*ntot),ntot-int(0.9*ntot)])
            train_out         = torch.cat((xsig.cpu(),xbkg.cpu()))
            train_out         = train_out[randind]
            train_base_mse    = DataSet(samples=train_out,    labels=train_label_out,masses=train_mass_out)
            self.data_base_train_mse, self.val_base_train_mse = random_split(train_base_mse, [int(0.9*ntot),ntot-int(0.9*ntot)])

    def fulltrain_dataloader(self):
        return DataLoader(self.sig_train, batch_size=len(self.sig_train),num_workers=16),DataLoader(self.bkg_train, batch_size=len(self.bkg_train),num_workers=16)

    def train_dataloader(self):
        return DataLoader(self.sdata_train, batch_size=self.batch_size,num_workers=16),DataLoader(self.bdata_train, batch_size=self.batch_size,num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.sdata_val, batch_size=self.batch_size,num_workers=16),DataLoader(self.bdata_val, batch_size=self.batch_size,num_workers=16)

    def train_dataloader_mse(self):
        return DataLoader(self.data_train_mse, batch_size=self.batch_size,num_workers=16)

    def train_dataloader_base_mse(self):
        return DataLoader(self.data_base_train_mse, batch_size=self.batch_size,num_workers=16)

    def val_dataloader_mse(self):
        return DataLoader(self.val_train_mse, batch_size=self.batch_size,num_workers=16)

    def val_dataloader_base_mse(self):
        return DataLoader(self.val_base_train_mse, batch_size=self.batch_size,num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,num_workers=16)

    def run_all(self):
        self.setup()
        self.training_clr()
        self.setup_mse()
        self.training_mse()
        scores_out,labels_out,mass_out,space_out = self.test_all()
        mask = (mass_out > 70) & (mass_out < 90)
        auc = roc_auc_score(y_score=scores_out[mask], y_true=labels_out[mask])
        print("AUC",auc)
        return scores_out,labels_out,mass_out,space_out

    def run_base(self):
        self.setup()
        self.setup_mse()
        self.training_base_mse()
        scores_out,labels_out,mass_out = self.test_base()
        mask = (mass_out > 70) & (mass_out < 90)
        auc = roc_auc_score(y_score=scores_out[mask], y_true=labels_out[mask])
        print("AUC",auc)
        return scores_out,labels_out,mass_out

class DataSet(Dataset):
    def __init__(self, samples, labels, masses):
        self.labels  = labels
        self.samples = samples
        self.masses  = masses
        if len(samples) != len(labels):
            raise ValueError(
                f"should have the same number of samples({len(samples)}) as there are labels({len(labels)})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        y = self.labels[index]
        m = self.masses[index]
        x = self.samples[index]
        return x, y, m


class RandomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )


def load(iName):
    f = h5py.File('testsamples_v2/'+str(iName)+'.h5', 'r')
    #f = h5py.File('testsamples/'+str(iName)+'.h5', 'r')
    key = list(f.keys())[0]
    dset = f[key]
    return dset

def prepDataset():
    prong1  = load('1prong_floatP_flat_parts')
    prong2  = load('2prong_floatP_flat_parts')
    mass1   = load('1prong_floatP_flat_mass')
    mass2   = load('2prong_floatP_flat_mass')
    data    = np.concatenate([prong1,prong2]).astype("float32")
    mass    = np.concatenate([mass1,mass2]).astype("float32")
    labels  = np.concatenate([np.ones(len(prong1)), np.zeros(len(prong2))]).astype("float32")
    labels  = np.reshape(labels,(len(labels),1))
    data    = (data - data.mean(axis=0))/data.std(axis=0)
    output  = DataSet(samples=data,labels=labels,masses=mass)
    size    = prong1.shape[1]
    return output,size

def prepPartDataset(imasscut=0):
    prong1  = load('1prong_floatP_flat_32p_parts').astype("float32")
    prong2  = load('2prong_floatP_flat_32p_parts').astype("float32")
    mass1   = load('1prong_floatP_flat_32p_mass').astype("float32")
    mass2   = load('2prong_floatP_flat_32p_mass').astype("float32")
    mass1   = np.reshape(mass1[:,0],(len(mass1),1))
    mass2   = np.reshape(mass2[:,0],(len(mass2),1))
    #cuts1   = mass1[:].flatten() > imasscut
    #cuts2   = mass2[:].flatten() > imasscut
    prong1  = np.reshape(prong1,(len(prong1),32,4))
    prong2  = np.reshape(prong2,(len(prong2),32,4))
    prong1  = prong1[:,:,0:3]
    prong2  = prong2[:,:,0:3]
    prong1  = np.reshape(prong1,(len(prong1),32*3))
    prong2  = np.reshape(prong2,(len(prong2),32*3))
    print("After:",prong1[0],prong1[:].shape)
    #mass1   = mass1 [cuts1,:]
    #mass2   = mass2 [cuts2,:]
    data    = np.concatenate([prong1,prong2]).astype("float32")
    labels1 = np.ones(len(prong1)).astype("float32")
    labels2 = np.zeros(len(prong2)).astype("float32")
    labels1 = np.reshape(labels1,(len(labels1),1))
    labels2 = np.reshape(labels2,(len(labels2),1))
    prong1  = (prong1 - data.mean(axis=0))/data.std(axis=0)
    prong2  = (prong2 - data.mean(axis=0))/data.std(axis=0)
    output1 = DataSet(samples=prong1,labels=labels1,masses=mass1)
    output2 = DataSet(samples=prong2,labels=labels2,masses=mass2)
    size    = prong1.shape[1]
    return output1,output2,size


def prepParetoDataset(imasscut=0):
    prong1  = load('1prong_floatP_flat_32p_theta')#1prong_floatP_flat_theta')
    prong2  = load('2prong_floatP_flat_32p_theta')#2prong_floatP_flat_theta')
    mass1   = load('1prong_floatP_flat_32p_mass')
    mass2   = load('2prong_floatP_flat_32p_mass')
    mass1   = np.reshape(mass1[:,0],(len(mass1),1))
    mass2   = np.reshape(mass2[:,0],(len(mass2),1))
    #cuts1   = mass1[:].flatten() > imasscut
    #cuts2   = mass2[:].flatten() > imasscut
    #prong1  = prong1[cuts1]
    #prong2  = prong2[cuts2]
    #mass1   = mass1[cuts1]
    #mass2   = mass2[cuts2]
    prong1  = np.reshape(prong1[:,0],(len(prong1),1))
    prong2  = np.reshape(prong2[:,0],(len(prong2),1))
    data    = np.concatenate([prong1,prong2]).astype("float32")
    mass    = np.concatenate([mass1,mass2]).astype("float32")
    labels  = np.concatenate([np.ones(len(prong1)), np.zeros(len(prong2))]).astype("float32")
    labels  = np.reshape(labels,(len(labels),1))
    #data    = (data - data.mean(axis=0))/data.std(axis=0)
    output  = DataSet(samples=data,labels=labels,masses=mass)
    size    = prong1.shape[1]
    return output,size


#dataset1,dataset2,size  = prepPartDataset()
#codec_model             = codec_model(size,dataset1,dataset2)
#codec_model.run_all()
#codec_model.run_base()

#dataset,size  = prepParetoDataset()
#train_indices = torch.arange(0, len(dataset)).numpy()
#train_sampler = SubsetRandomSampler(train_indices)
#train_loader  = DataLoader(dataset,batch_size=10000,num_workers=4,sampler=train_sampler)
#test_model    = simple_model(size,dataset)
#trainer = Trainer(accelerator="auto",devices=1 if torch.cuda.is_available() else None,max_epochs=10,callbacks=[TQDMProgressBar(refresh_rate=20)],logger=CSVLogger(save_dir="logs/"),)
#trainer.fit(test_model)
#trainer.test()

#a
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(0)
#num_epochs = 10
#for epoch in range(num_epochs):
#    print('Epoch:', epoch+1, end='')
#    for batch_idx, (x, y) in enumerate(train_loader):
#        #print(' | Batch index:', batch_idx, end='')
#        #print(' | Batch size:', y.size()[0])
#        simple_optimizer.zero_grad()
#        x = x.to(device)
#        y = y.to(device)
#        outputs = simple_model(x)
#        loss = simple_criterion(outputs, y)
#        loss.backward()
#        simple_optimizer.step()
#        current_loss = loss.item()
#        if batch_idx % 100 == 0: print('[%d] loss: %.4f  ' % (epoch + 1,  current_loss))

