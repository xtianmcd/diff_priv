from joblib import Memory
from sklearn.datasets import load_svmlight_file
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import math
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression

mem = Memory("./mycache")

@mem.cache
def get_data():
    #source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    tr_data = load_svmlight_file("adult_train.txt")
    te_data = load_svmlight_file("adult_test.t")

    tr_dummy = np.ones((tr_data[0].shape[0],1))
    te_dummy = np.ones((te_data[0].shape[0],1))

    trn_data = hstack((tr_dummy,tr_data[0])).toarray()
    tst_data = hstack((te_dummy,te_data[0],te_dummy)).toarray()

    return trn_data, tr_data[1], tst_data, te_data[1]

def sigma(x):
    #overflow avoidance source: https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
    signal = np.clip(x, -500, 500 )
    sigmoid = 1/(1+np.exp(-signal))
    return sigmoid

def ComputeGradient(b_vec,xy_pairs):
    n = len(xy_pairs)
    loss_grad = 1/n * \
        np.sum([np.multiply(sigma(pair[1]*\
        np.dot(np.transpose(b_vec),pair[0]))-1,
        np.multiply(pair[1],pair[0])) for pair in xy_pairs],axis=0)
    return loss_grad

def ApproxGrad(b_vec,xy_pairs,h=10**-5):
    n = len(xy_pairs)
    approx_grad=[]
    # print(b_vec)
    for i in range(len(b_vec)):
        b_plus = copy.deepcopy(b_vec)
        b_plus[i] = b_plus[i]+h
        b_minus = copy.deepcopy(b_vec)
        b_minus[i] = b_minus[i]-h
        grad_i = (((-1/n) * \
            sum([np.log(sigma(pair[1]*np.dot(np.transpose(b_plus),pair[0]))) \
            for pair in xy_pairs])) - ((-1/n) * \
            sum([np.log(sigma(pair[1]*np.dot(np.transpose(b_minus),pair[0]))) \
            for pair in xy_pairs])))/(2*h)
        approx_grad.append(grad_i)
    return approx_grad

def dp_sensitivity(b_vec,xy_pairs):
    # neighbors=[]
    # for col in range(b_vec.shape[0]):
    #     neighb = copy.deepcopy(b_vec)
    #     neighb[col] = 0
    #     neighbors.append(neighb)
    max_grad_dj = np.linalg.norm(1)*1
    sens = np.max([max_grad_dj for dj in range(len(xy_pairs))])
    print(f"Sensitivity Calculation for Differential Privacy: {sens}")
    return sens

def NoisyGradient(b_vec, xy_pairs, sensitivity, ep, delta=10**-5):
    var = sensitivity/ep * math.sqrt(2*math.log(1.25/delta))
    Y = np.random.normal(0,var**2,xy_pairs[0][0].shape[0])#*np.identiy(xy_pairs[0,0].shape[1])
    grad_tilda = np.reshape(np.add(np.transpose\
        (ComputeGradient(b_vec,xy_pairs))[0],Y),(b_vec.shape[0],1))
    return grad_tilda

def ComputeLoss(b_vec, xy_pair):
    # l_p = np.log(1+np.exp(-xy_pair[1]*np.dot(np.transpose(b_vec),xy_pair[0])))
    l_p = np.log(sigma(xy_pair[1]*np.dot(np.transpose(b_vec),xy_pair[0])))
    return l_p

def backtrack(b_vector,xy_pairs,grad,s=0.9,a=0.4):
    n=1
    while (-1/(len(xy_pairs)) * \
            sum([ComputeLoss(np.subtract(b_vector,np.multiply(n,grad)),\
            xy_pairs[i]) for i in range(len(xy_pairs))]))\
            > (-1/(len(xy_pairs)) * \
            sum([ComputeLoss(b_vector,xy_pairs[i])\
             - (a*n)*np.linalg.norm(grad)**2 for i in range(len(xy_pairs))])):
        n *= s
    return n

def Pred(x_test_i, beta):
    y_hat_i = round(1/(1+np.exp(-(np.dot(np.transpose(beta),x_test_i))))[0])
    if y_hat_i==0: y_hat_i=-1
    return y_hat_i

def logistic_reg_train(x,y,x_tst,y_tst,iters=100,epsilon=10**-10,
                         thresh=10**-6,glob_sens=None,dp=False,dp_ep=0.5):
    if not dp:
        with open('./gradient_diffs.txt','w') as gd:
            gd.write('\n---------- Calculated/Approx. Gradient Differences ----------\n\n')
    xy = [[np.reshape(x[i],(x.shape[1],1)),y[i]] for i in range(x.shape[0])]
    xy_test = [[np.reshape(x_tst[i],(x_tst.shape[1],1)),y_tst[i]]\
                for i in range(x_tst.shape[0])]
    beta_t = np.reshape(np.zeros(x.shape[1]),(x.shape[1],1)) #np.reshape(np.random.normal(0,0.01,x.shape[1]),(x.shape[1],1))
    if not glob_sens: glob_sens = dp_sensitivity(beta_t,xy)#0.011070701960557393#
    if dp: g_t = NoisyGradient(beta_t,xy,glob_sens,dp_ep)
    else: g_t = ComputeGradient(beta_t,xy)
    alpha_t = backtrack(beta_t,xy,g_t)
    l_new = -1/(len(xy))*sum([ComputeLoss(beta_t,xy[i])\
            for i in range(x.shape[0])])
    tr_ax=[]
    te_ax=[]
    objectives=[]
    obj_test=[]
    ex=0
    for k in range(iters):
        if dp: g_t = NoisyGradient(beta_t,xy,glob_sens,dp_ep)
        else: g_t = ComputeGradient(beta_t,xy)
        if k+1 in [2,20,200] and not dp:
            ex+=1
            approx_grad = ApproxGrad(beta_t,xy)
            grad_diffs = np.mean(np.subtract(g_t,approx_grad))
            print(f"Gradient Example {ex} - On iteration {k}, the average difference between the calculated gradient and the gradient approximation is {grad_diffs}.")
            with open('./gradient_diffs.txt','a') as gd:
                gd.write(f'++++ Gradient Example {ex}; Training Iteration {k} ++++\nMEAN DIFFERENCE: \t{grad_diffs}\n\nActual Differences: {np.subtract(g_t,approx_grad)}\nb-vector: \n{np.reshape(beta_t,(1,len(beta_t)))}\nCalc. Grad.: \n{np.reshape(g_t,(1,len(g_t)))}\nApprox. Grad.: \n{approx_grad}\n')
        if np.linalg.norm(g_t) < epsilon: return {'beta':beta_t,'loss':l_new,
                                                  'iters':k,'step':alpha_t,
                                                  'exit':'1','train_acc':tr_ax,
                                                  'test_acc':te_ax,
                                                  'train_obj':objectives,
                                                  'test_obj':obj_test,
                                                  'sensitivity':glob_sens}
        l_t = -1/(len(xy))*sum([ComputeLoss(beta_t,xy[i])\
            for i in range(x.shape[0])])
        alpha_t = backtrack(beta_t,xy,g_t)
        print(f'step size on iter {k+1}: {alpha_t}')
        beta_new = np.subtract(beta_t,np.multiply(alpha_t,g_t))
        l_new = -1/(len(xy))*sum([ComputeLoss(beta_new,xy[i])\
            for i in range(x.shape[0])])
        if not dp:
            if (l_t - l_new) <= thresh: return {'beta':beta_t,'loss':l_new,
                                            'iters':k,'step':alpha_t,'exit':'2',
                                            'train_acc':tr_ax,'test_acc':te_ax,
                                            'train_obj':objectives,
                                            'test_obj':obj_test,
                                            'sensitivity':glob_sens}
        beta_t = beta_new
        tr_acc = sum([1 for predxn in range(x.shape[0])\
                if Pred(x[predxn],beta_t) == y[predxn]]) / x.shape[0] * 100
        te_acc = sum([1 for predxn in range(x_tst.shape[0])\
                if Pred(x_tst[predxn],beta_t) == y_tst[predxn]])\
                 / x_tst.shape[0] * 100
        l_tst = -1/(len(xy))*sum([ComputeLoss(beta_new,xy[i])\
            for i in range(x.shape[0])])
        tr_ax.append(tr_acc)
        te_ax.append(te_acc)
        objectives.append(l_new[0][0])
        obj_test.append(l_tst[0][0])
    return {'beta':beta_t,'loss':l_new,'iters':k,'step':alpha_t,'exit':'3',
            'train_acc':tr_ax,'test_acc':te_ax,'train_obj':objectives,
            'test_obj':obj_test,'sensitivity':glob_sens}

def plot_training(trn_acc, tst_acc, obj_vals, obj_tv, dp=False,ep=0.5):
    tr_maxiter = trn_acc.index(np.max(trn_acc))
    te_maxiter = tst_acc.index(np.max(tst_acc))
    f,axrow = plt.subplots(1,2)
    axrow[0].plot(range(len(trn_acc)),trn_acc, label="Training Acc")
    axrow[0].plot(range(len(tst_acc)),tst_acc, label="Testing Acc", color='g')
    axrow[0].set(xlabel='Training Epochs',ylabel='Accuracy')
    axrow[0].set_title("Accuracy")
    axrow[0].set_ylim(0,100)
    axrow[0].annotate(f'{tr_maxiter},{np.max(trn_acc):.2f}%', xy=(tr_maxiter,
            np.max(trn_acc)),xytext=(tr_maxiter+0.04*len(trn_acc),
            np.max(trn_acc)+7),arrowprops=dict(facecolor='black', shrink=0.05),
            )
    axrow[0].annotate(f'{te_maxiter},{np.max(tst_acc):.2f}%', xy=(te_maxiter,
            np.max(tst_acc)),xytext=(te_maxiter+0.01*len(tst_acc),
            np.max(tst_acc)+10),arrowprops=dict(facecolor='black', shrink=0.05),
            )
    #annotation source; https://matplotlib.org/users/annotations_intro.html
    axrow[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
    axrow[1].plot(range(len(obj_vals)),obj_vals, label="Training Loss")
    axrow[1].plot(range(len(obj_tv)),obj_tv, label="Testing Loss",color='g')
    axrow[1].set(xlabel='Training Epochs',ylabel="Loss")
    axrow[1].set_title('Objective Function')
    axrow[1].legend(loc='upper right', bbox_to_anchor=(1, 1),
          ncol=1, fancybox=True, shadow=True)
    #legend source: https://pythonspot.com/matplotlib-legend/
    if dp: plt.savefig(f'./dp_plot_{ep}ep.png')
    else: plt.savefig('./nonpriv_plot.png')
    return

def verify_performance(X_trn,y_trn,X_tst,y_tst):
    logreg = LogisticRegression().fit(X_trn[:,1:], y_trn)
    logr_pred = logreg.predict(X_tst[:,1:])
    beta=logreg.coef_
    b0 = logreg.intercept_
    acc = 100 - (sum([1 for predxn in range(len(logr_pred))\
        if logr_pred[predxn]!= y_tst[predxn]])/len(X_tst)*100)
    return acc

if __name__ == "__main__":

    X_tr, y_tr, X_te, y_te = get_data()
    print(X_tr.shape,y_tr.shape,X_te.shape,y_te.shape)

    logreg = logistic_reg_train(X_tr,y_tr,X_te,y_te,500)
    print(f"\nFinal Loss: \t\t{logreg['loss']}\nNumber of Iterations: \t{logreg['iters']+1}\nFinal Step Size: \t{logreg['step']}\nReturned at Breakpoint {logreg['exit']}\n")

    # print(f"max testing acc: {np.max(te_accuracy)}")
    # print(f"training acc: {np.max(tr_accuracy)}")

    # with open('accuracies.txt','a') as accs:
    #     accs.write(f"Testing\n{te_accuracy}\n")
    #     accs.write(f"Training\n{tr_accuracy}\n")


    acc = sum([1 for predxn in range(X_te.shape[0])\
            if Pred(X_te[predxn],logreg['beta']) == y_te[predxn]])\
            / X_te.shape[0] * 100

    print ("+ **************************** +")
    print(f"| FINAL TEST ACCURACY: {acc:.3f}% |")
    print(f"|  MAX TEST ACCURACY: {np.max(logreg['test_acc']):.3f}%  |")
    print ("+ **************************** +")

    with open('logreg_performance.txt','w') as perf:
        perf.write(f"\nFinal Loss: \t\t{logreg['loss']}\nNumber of Iterations: \t{logreg['iters']+1}\nFinal Step Size: \t{logreg['step']}\nReturned at Breakpoint {logreg['exit']}\nFinal Test Accuracy: {acc:.3f}%\n Max Test Accuracy: {np.max(logreg['test_acc']):.3f}%")

    plot_training(logreg['train_acc'],logreg['test_acc'],
                  logreg['train_obj'],logreg['test_obj'])

    skl_acc = verify_performance(X_tr,y_tr,X_te,y_te)
    print(f"\nFor verification, Scikit-Learn Logistic Regression achieved {skl_acc:.3f}% accuracy.")

    epsilons = [0.1,0.5,1.0,1.5]
    for e in epsilons:
        dp_logreg = logistic_reg_train(X_tr,y_tr,X_te,y_te,iters=15,glob_sens=logreg['sensitivity'],dp=True,dp_ep=e)
        with open('logreg_performance.txt','a') as perf:
            perf.write(f"\nDiff. Private Run, epsilon={e}\nSensitivity: \t{dp_logreg['sensitivity']}\nFinal Loss: \t\t{dp_logreg['loss']}\nNumber of Iterations: \t{dp_logreg['iters']+1}\nFinal Step Size: \t{dp_logreg['step']}\nReturned at Breakpoint \t{dp_logreg['exit']}\nFinal Test Accuracy: \t{acc:.3f}%\nMax Test Accuracy: \t{np.max(dp_logreg['test_acc']):.3f}%")
        print(f"\n\nDiff. Private Run, epsilon={e}\nSensitivity: \t{dp_logreg['sensitivity']}\nFinal Loss: \t\t{dp_logreg['loss']}\nNumber of Iterations: \t{dp_logreg['iters']+1}\nFinal Step Size: \t{dp_logreg['step']}\nReturned at Breakpoint \t{dp_logreg['exit']}\nFinal Test Accuracy: \t{acc:.3f}%\nMax Test Accuracy: \t{np.max(dp_logreg['test_acc']):.3f}%")
        plot_training(dp_logreg['train_acc'],dp_logreg['test_acc'],
                      dp_logreg['train_obj'],dp_logreg['test_obj'],dp=True,ep=e)
