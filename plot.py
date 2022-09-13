import corner
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,auc

def plotAUC(iscores,ilabels,imass,iweights):
    mask = (imass > 70) & (imass < 90)
    auc = roc_auc_score(y_score=iscores[mask], y_true=ilabels[mask].detach().numpy().flatten())#,sample_weight=iweights[mask])
    print("AUC",auc)
    fpr, tpr, cuts = roc_curve(y_score=iscores[mask], y_true=ilabels[mask])#,sample_weight=iweights[mask])
    #fig, ax = plt.subplots(1,1,figsize=(4,3),dpi=150)
    return fpr,tpr,auc

def plotAUC2(iscores,ilabels,imass,iweights):
    mask = (imass > 70) & (imass < 90)
    auc = roc_auc_score(y_score=iscores[mask], y_true=ilabels[mask])#,sample_weight=iweights[mask])
    print("AUC",auc)
    fpr, tpr, cuts = roc_curve(y_score=iscores[mask], y_true=ilabels[mask])#,sample_weight=iweights[mask])
    #fig, ax = plt.subplots(1,1,figsize=(4,3),dpi=150)
    return fpr,tpr,auc

def plot_hists(cut,scores,test_mass,test_labels,c="w",density=True):
    # sig_bkg
    fig, ax      = plt.subplots(1, 1, figsize=(4, 3.6))
    _,bins,_=plt.hist(test_mass[test_labels == 0],                   bins=80,density=density,histtype="step",label="Background",color="b",ls='--')
    _,bins,_=plt.hist(test_mass[test_labels == 1],                   bins=bins,density=density,histtype="step",label="Signal",color="r",ls='--')
    _,bins,_=plt.hist(test_mass[(test_labels == 1) & (scores > cut)],bins=bins,density=density,histtype="step",label="selected sig",color="r")
    _,bins,_=plt.hist(test_mass[(test_labels == 0) & (scores > cut)],bins=bins,density=density,histtype="step",label="selected bkg",color="b")
    plt.legend(loc='upper right', fontsize=12, ncol=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([40, 240])
    plt.yscale("log")
    ax.set_xlabel("Mass [GeV]", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    fig.tight_layout(pad=0)
    return bins,fig, ax

def pearson(pred, target):
    predscale   = ((pred   - pred.mean())  /pred.std()).reshape(len(pred),1)
    targetscale = ((target - target.mean())/target.std()).reshape(len(target),1)
    #print(predscale.shape,targetscale.shape,(np.dot(predscale.flatten() ,targetscale.flatten())).shape,((predscale * targetscale)).shape)
    ret = (np.dot(predscale.flatten(),targetscale.flatten())).mean()
    return ret/len(pred)

def jsd(iH0,iH1):
    if len(iH0) < 11:
        return 0
    quants=np.arange(0.,1.1,0.1)
    binrange=np.quantile(iH0.flatten(), quants, axis=0)
    hist0, bin_edges = np.histogram(iH0.flatten(), bins=binrange,density=True)
    hist1, bin_edges = np.histogram(iH1.flatten(), bins=bin_edges,density=True)
    lJSD = distance.jensenshannon(hist0,hist1)
    return lJSD

def jsdROC(iVar,iMass):
    fpr   = np.array([]);     lJSD   = np.array([])   
    nbins=500    
    quants=np.logspace(-5, 0., num=nbins)
    binrange=np.quantile(iVar, quants, axis=0)
    pMax=binrange[-1]
    pTot = len(iMass[iVar < pMax])
    for pVal in binrange:
        pJSD = jsd(iMass[iVar < pVal],iMass[iVar < pMax])
        lJSD = np.append(lJSD,pJSD)
        pEff = len(iMass[iVar < pVal])
        fpr  = np.append(fpr,(pEff/pTot))
    return fpr,lJSD

def roc2D(iVar, iMass,iLabel):
    nbinsM=np.arange(50,300,10)
    nbinsV=500
    hbkg, xedges, yedges  = np.histogram2d(iMass[iLabel==0], iVar[iLabel==0], bins=(nbinsM,nbinsV))
    hsig, xedges, yedges  = np.histogram2d(iMass[iLabel==1], iVar[iLabel==1], bins=(xedges,yedges))
    pTVal = 0;                 pBVal = 0;
    fpr   = np.array([]);     tpr   = np.array([])
    pTTot = np.sum(hsig[0:len(nbinsM),:]); pBTot = np.sum(hbkg[0:len(nbinsM),:])
    sigEff=np.arange(0,1,0.003)
    for pEff in sigEff:
        pTTemp=0
        pBTemp=0
        for pMass in range(len(nbinsM)-1):
            lBin=0
            for pBin in range(nbinsV):
                pFrac=np.sum(hbkg[pMass,(nbinsV-pBin):nbinsV])/ np.sum(hbkg[pMass,:])
                if pFrac > pEff:
                    lBin=pBin
                    break                
            pTTemp = pTTemp+np.sum(hsig[pMass,(nbinsV-pBin):nbinsV])
            pBTemp = pBTemp+np.sum(hbkg[pMass,(nbinsV-pBin):nbinsV])
        if pTTemp > 0 or pBTemp > 0:
            fpr = np.append(fpr,(pTTemp/pTTot))
            tpr = np.append(tpr,(pBTemp/pBTot))
    fpr = np.append(fpr,1.0)
    tpr = np.append(tpr,1.0)
    return fpr, tpr, auc(fpr,tpr)

def rocplot(iV,iLabel):
    auc_value                  = roc_auc_score(y_score=iV, y_true=iLabel)
    fpr_value, tpr_value, cuts = roc_curve    (y_score=iV, y_true=iLabel)
    if auc_value < 0.5:
        auc_value                  = roc_auc_score(y_score=iV, y_true=(1-iLabel))
        fpr_value, tpr_value, cuts = roc_curve    (y_score=iV, y_true=(1-iLabel))
    return fpr_value,tpr_value,auc_value
        
def plotCorr(iaxis, scores_out,labels_out,mass_out,iCut,iSign=False):
    lSign = 1
    if iSign:
        lSign = -1
    cutmass0 = mass_out[((labels_out==0) & (scores_out*lSign < iCut*lSign))]
    cutmass1 = mass_out[((labels_out==1) & (scores_out*lSign < iCut*lSign))]
    incmass0 =  mass_out[((labels_out==0) )]
    incmass1 =  mass_out[((labels_out==1) )]
    lJSD1 = jsd(cutmass0,incmass0)
    lJSD2 = jsd(cutmass1,incmass1)
    bins=np.arange(0,400,5)
    iaxis.hist(incmass0,alpha=0.5,bins=bins,density=True)
    iaxis.hist(incmass1,alpha=0.5,bins=bins,density=True)
    iaxis.hist(cutmass0,alpha=0.5,bins=bins,density=True)
    iaxis.hist(cutmass1,alpha=0.5,bins=bins,density=True)
    
    iaxis.text(250,0.016,"JSD 2prong= {:.4f}".format(lJSD1),color='green')
    iaxis.text(250,0.008,"JSD 1prong= {:.4f}".format(lJSD2),color='red')
    return

def plotAxes(iAxis,iXaxis,iYaxis,iLabels):
    lXaxis0 = iXaxis[(iLabels==0)]; lYaxis0 = iYaxis[(iLabels==0)]
    lXaxis1 = iXaxis[(iLabels==1)]; lYaxis1 = iYaxis[(iLabels==1)]
    corr1=pearson(lXaxis0,lYaxis0)
    corr2=pearson(lXaxis1,lYaxis1)
    print("test",corr1,corr2)
    iAxis.scatter(lXaxis0,lYaxis0,marker='o',s=0.01)
    iAxis.scatter(lXaxis1,lYaxis1,marker='o',s=0.01)
    lXMin = iXaxis.min(); lXMax = iXaxis.max(); lXR   = lXMax - lXMin
    lYMin = iYaxis.min(); lYMax = iYaxis.max(); lYR   = lYMax - lYMin
    iAxis.text(0.7*lXR+lXMin,0.7*lYR+lYMin,"corr={:.4f}".format(corr1),color="blue")
    iAxis.text(0.7*lXR+lXMin,0.5*lYR+lYMin,"corr={:.4f}".format(corr2),color="orange")
    iAxis.set_xlabel("latent-x")
    iAxis.set_ylabel("latent-y")


def plotAll(scores_out,labels_out,mass_out,space_out,theta,mass,labels,icut1,icut2):
    figure, axis = plt.subplots(2, 3,figsize=(20,5))
    plotCorr(axis[0,0],scores_out,labels_out,mass_out,icut1)
    plotCorr(axis[0,1],scores_out,labels_out,mass_out,icut2,True)
                                
    bins=np.arange(0,1,0.01)
    axis[0,2].hist(scores_out[(labels_out==0) ],alpha=0.5,bins=bins,density=True)
    axis[0,2].hist(scores_out[(labels_out==1) ],alpha=0.5,bins=bins,density=True)

    space_out2 = np.reshape(space_out,(len(space_out)//2,2))
    mass_out   = np.reshape(mass_out ,(len(mass_out),1) )
    space_out_mass = np.append(space_out2,mass_out,axis=1)
    plotAxes(axis[1,0],space_out2[:,0],space_out2[:,1],labels_out)
    plotAxes(axis[1,1],mass_out,       space_out2[:,0],labels_out)
    plotAxes(axis[1,2],mass_out,       space_out2[:,1],labels_out)
    plt.show()

    figure, axis = plt.subplots(1, 3,figsize=(20,5))
    fprb,tprb,aucb=rocplot(theta,labels)
    fprt,tprt,auct=rocplot(scores_out,labels_out)
    fpr1,tpr1,auc1=rocplot(space_out2[:,0],labels_out)
    fpr2,tpr2,auc2=rocplot(space_out2[:,1],labels_out)
    fpr2D,tpr2D,auc2D = roc2D(theta,mass,labels)
    axis[0].plot(fprb, tprb,label="pareto:{:.6f}".format(aucb),c='black')
    axis[0].plot(fpr2D,tpr2D,label="pareto-ddt:{:.6f}".format(auc2D),c='green')
    axis[0].plot(fprt, tprt,label="disc:{:.6f}".format(auct))
    axis[0].plot(fpr1, tpr1,label="x:{:.6f}".format(auc1))
    axis[0].plot(fpr2, tpr2,label="y:{:.6f}".format(auc2))
    axis[0].set_xscale('log')
    axis[0].set_xlabel("fpr")
    axis[0].set_ylabel("tpr")
    axis[0].legend(title="auc")

    fjsdrb1,jsdrb1=jsdROC(theta[labels==0],            mass[labels==0])
    fjsdrt1,jsdrt1=jsdROC(scores_out[labels_out==0],  mass_out[labels_out==0])
    fjsdr11,jsdr11=jsdROC(space_out2[labels_out==0,0],mass_out[labels_out==0])
    fjsdr21,jsdr21=jsdROC(space_out2[labels_out==0,1],mass_out[labels_out==0])
    axis[1].plot(fjsdrb1, jsdrb1,label="pareto",c='black')
    axis[1].plot(fjsdrt1, jsdrt1,label="disc")
    axis[1].plot(fjsdr11, jsdr11,label="x")
    axis[1].plot(fjsdr21, jsdr21,label="y")
    axis[1].set_xlim([0.001,1.0])
    axis[1].set_xscale('log')
    axis[1].set_xlabel("fpr")
    axis[1].set_ylabel("JSD(sig)")
    axis[1].legend()
    
    fjsdrb2,jsdrb2=jsdROC(theta[labels==1],          mass[labels==1])
    fjsdrt2,jsdrt2=jsdROC(scores_out[labels_out==1],mass_out[labels_out==1])
    fjsdr12,jsdr12=jsdROC(space_out2[labels_out==1,0],mass_out[labels_out==1])
    fjsdr22,jsdr22=jsdROC(space_out2[labels_out==1,1],mass_out[labels_out==1])
    axis[2].plot(fjsdrb2, jsdrb2,label="pareto",c='black')
    axis[2].plot(fjsdrt2, jsdrt2,label="disc")
    axis[2].plot(fjsdr12, jsdr12,label="x")
    axis[2].plot(fjsdr22, jsdr22,label="y")
    axis[2].set_xlim([0.001,1.0])
    axis[2].set_xscale('log')
    axis[2].set_xlabel("fpr")
    axis[2].set_ylabel("JSD(bkg)")
    axis[2].legend()
    plt.show()
    
    figure = corner.corner(space_out_mass[labels_out==0],labels=[r"$x$",r"$y$",r"$m$"],quantiles=[0.16, 0.5, 0.84],show_titles=True,title_kwargs={"fontsize": 12},color='green')
    corner.corner(space_out_mass[labels_out==1], fig=figure,color='orange')
    plt.show()
