import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
from itertools import compress
from scipy.spatial.distance import cdist,pdist,squareform
from utils import *

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	#P3
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	Lmbda_gp = [0.02]
	#P2 gaussian
	Lmbda_g = [2, 2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
	#P2 linear
	C = [0.01, 0.1, 1, 10, 100]

	#P1
	#C = [1e-2, 1e-1,1,10,100]
	L = ['l1','l2']


	EM_gp = np.load('EM_gaussian_p.txt.npy')
	margin_gp = np.load('margin_gaussian_p.txt.npy')
	NSV_gp = np.load('NSV_gaussian_p.txt.npy')
	print(EM_gp.shape)
	print(margin_gp.shape)
	print(NSV_gp.shape)

	EM_lp = np.load('EM_linear_p.txt.npy')
	margin_lp = np.load('margin_linear_p.txt.npy')
	NSV_lp = np.load('NSV_linear_p.txt.npy')
	print(EM_lp.shape)
	print(margin_lp.shape)
	print(NSV_lp.shape)

	EM_g = np.load('EM_gaussian.txt.npy')
	margin_g = np.load('margin_gaussian.txt.npy')
	NSV_g = np.load('NSV_gaussian.txt.npy')
	print(EM_g.shape)
	print(margin_g.shape)
	print(NSV_g.shape)

	EM_l = np.load('EM_linear.txt.npy')
	margin_l = np.load('margin_linear.txt.npy')
	NSV_l = np.load('NSV_linear.txt.npy')
	print(EM_l.shape)
	print(margin_l.shape)
	print(NSV_l.shape)

	EM_lr = np.load('EM_lr.txt.npy')

	EM_4gp = np.load('EM_gaussian_p4.txt.npy')
	margin_4gp = np.load('margin_gaussian_p4.txt.npy')
	NSV_4gp= np.load('NSV_gaussian_p4.txt.npy')

	EM_4g = np.load('EM_gaussian_4.txt.npy')
	margin_4g = np.load('margin_gaussian_4.txt.npy')
	NSV_4g= np.load('NSV_gaussian_4.txt.npy')

	EM_4l = np.load('EM_linear_4.txt.npy')
	margin_4l = np.load('margin_linear_4.txt.npy')
	NSV_4l = np.load('NSV_linear_4.txt.npy')


	EM_4lr = np.load('EM_lr_4.txt.npy')
	margin_4lr = np.load('margin_lr_4.txt.npy')
	NSV_4lr = np.load('NSV_lr_4.txt.npy')
	# print(margin_g.shape)
	# for ni,name in enumerate(Name):
	# 	for nc,c in enumerate(C):
	# 		pl.loglog(Gamma,margin_g[ni,:,nc,0])

	# pl.show()
	# pl.close()

	# print(NSV_l.shape)
	# for ni,name in enumerate(Name):
	# 	for nc,c in enumerate(C):
	# 		pl.loglog(Gamma,NSV_l[ni,:,nc,0])

	# pl.show()
	# pl.close()

	#print(EM_l.shape)
	bar_width = 0.1
	opacity = 0.4
	color = ['b','g','r','c','m','y','b']
	margin = 0.15
	index = np.arange(len(Name))
	width = (1.-2*margin)/len(Name)

	# 	#for ni,name in enumerate(Name):
	# for gi, g in enumerate(L):
	# 	for nc,c in enumerate(C):
	# 		pl.bar(index+width*nc+margin,EM_lr[:,nc,gi,1],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
		
	

	# 	pl.xticks(index+0.5,('1', '2', '3', '4'))
	# 	#pl.tight_layout()

	# 	pl.title('Logistic regression $L$ ='+str(g))
	# 	pl.xlabel('Data set')
	# 	pl.ylabel('Classification error [%/100]')
	# 	pl.legend(loc=2, fontsize = 'small')
	# 	pl.savefig('./figures/ps2_LRtest'+str(g)+'.pdf')
	# 	pl.close()




	#for ni,name in enumerate(Name):
	# for nc,c in enumerate(C):
	# 	pl.bar(index+width*nc+margin,EM_l[:,nc,0],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
		
	

	# pl.xticks(index+0.5,('1', '2', '3', '4'))
	# #pl.tight_layout()

	# pl.title('Linear SVM')
	# pl.xlabel('Data set')
	# pl.ylabel('Classification error [%/100]')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/ps2_test_eml.pdf')
	# pl.close()


	# print(EM_g.shape)
	# #for ni,name in enumerate(Name):
	# for gi, g in enumerate(Gamma):
	# 	for nc,c in enumerate(C):
	# 		pl.bar(index+width*nc+margin,EM_g[:,nc,gi,1],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
		
	

	# 	pl.xticks(index+0.5,('1', '2', '3', '4'))
	# 	#pl.tight_layout()

	# 	pl.title('Gaussian SVM $\gamma$ ='+str(g))
	# 	pl.xlabel('Data set')
	# 	pl.ylabel('Classification error [%/100]')
	# 	pl.legend(loc=2, fontsize = 'small')
	# 	pl.savefig('./figures/ps2_emgaussian'+str(g)+'.pdf')
	# 	pl.close()

	# print(EM_g.shape)
	#for ni,name in enumerate(Name):
	# for gi, g in enumerate(Gamma):
	# 	#for nc,c in enumerate(C):
	# 	pl.bar(index+width*gi+margin,EM_gp[:,gi,0,2],width,alpha=opacity,color=color[gi],label = '$\gamma=$'+str(g))
		
	

	# pl.xticks(index+0.5,('1', '2', '3', '4'))
	# #pl.tight_layout()

	# pl.title('Gaussian Pegasos SVM')
	# pl.xlabel('Data set')
	# pl.ylabel('Classification error [%/100]')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/ps3_emgaussian.pdf')
	# pl.close()

	# margin = 0.35
	# index = np.arange(len(Name))
	# width = (1.-2*margin)/len(Name)
	# cmap = pl.get_cmap('gnuplot')
	# colors = [cmap(i) for i in np.linspace(0, 1, 11)]
	# for gi, g in enumerate(Lmbda_g):
	# 	#for nc,c in enumerate(C):
	# 	pl.bar(index+width*gi+margin,EM_lp[:,0,gi,2],width,alpha=opacity,color=colors[gi],label = '$\lambda=$'+str(g))
		
	
	# pl.xticks(index+0.5,('1', '2', '3', '4'))
	# #pl.tight_layout()

	# pl.title('Linear Pegasos SVM')
	# pl.xlabel('Data set')
	# pl.ylabel('Classification error [%/100]')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/ps3_emlinear.pdf')
	# pl.close()


	#for ni,name in enumerate(Name):
	# for gi, g in enumerate(Gamma):
	# 	#for nc,c in enumerate(C):
	# 	pl.bar(index+width*gi+margin,EM_4gp[:,gi,0,1],width,alpha=opacity,color=color[gi],label = '$\gamma=$'+str(g))
		
	
	# pl.xticks(index+0.5,('1,7', '3,5', '4,9', 'even,odd'))
	# #pl.tight_layout()

	# pl.title('Gaussian Pegasos SVM MNIST')
	# pl.xlabel('Data set')
	# pl.ylabel('Classification error [%/100]')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/ps4_pemgaussian.pdf')
	# pl.close()

	#for ni,name in enumerate(Name):
	# for gi, g in enumerate(Gamma):
	# 	for nc,c in enumerate(C):
	# 		pl.bar(index+width*nc+margin,EM_4g[:,nc,gi,1],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
		
	

	# 	pl.xticks(index+0.5,('1,7', '3,5', '4,9', 'even,odd'))
	# 	#pl.tight_layout()

	# 	pl.title('Gaussian SVM $\gamma$ ='+str(g))
	# 	pl.xlabel('Data set')
	# 	pl.ylabel('Classification error [%/100]')
	# 	pl.legend(loc=2, fontsize = 'small')
	# 	pl.savefig('./figures/ps4_emgaussian'+str(g)+'.pdf')
	# 	pl.close()

	# for nc,c in enumerate(C):
	# 	pl.bar(index+width*nc+margin,EM_4l[:,nc,0],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
		
	

	# pl.xticks(index+0.5,('1,7', '3,5', '4,9', 'even,odd'))
	# #pl.tight_layout()

	# pl.title('Linear SVM')
	# pl.xlabel('Data set')
	# pl.ylabel('Classification error [%/100]')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/ps4_eml.pdf')
	# pl.close()

	for gi, g in enumerate(L):
		for nc,c in enumerate(C):
			pl.bar(index+width*nc+margin,EM_4lr[:,nc,gi,1],width,alpha=opacity,color=color[nc],label = 'C='+str(c))
	


		pl.xticks(index+0.5,('1,7', '3,5', '4,9', 'even,odd'))
		#pl.tight_layout()

		pl.title('Logistic regression $L$ ='+str(g))
		pl.xlabel('Data set')
		pl.ylabel('Classification error [%/100]')
		pl.legend(loc=2, fontsize = 'small')
		pl.savefig('./figures/ps4_LR'+str(g)+'.pdf')
		pl.close()


	###Margin####

	# for ni,n in enumerate(Name):
	# 	#print(NSV_l[ni,:])
	# 	#print(C)
	# 	pl.loglog(C,NSV_l[ni,:,0],label='Dataset '+str(n),linewidth=2)

	# pl.xlabel('C')
	# pl.ylabel('# of Support Vectors')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/nsv_ps21.pdf')
	# pl.close()

	# for gi, g in enumerate(Gamma):
	# 	for ni,n in enumerate(Name):
	# 	#print(NSV_l[ni,:])
	# 	#print(C)
	# 		pl.loglog(C,NSV_g[ni,:,gi,0],label='Dataset '+str(n),linewidth=2)

	# 	pl.xlabel('C')
	# 	pl.ylabel('# of Support Vectors')
	# 	pl.legend(loc=2, fontsize = 'small')
	# 	pl.savefig('./figures/nsv_g'+str(g)+'.pdf')
	# 	pl.close()

	# for gi, g in enumerate(Gamma):
	# 	for ni,n in enumerate(Name):
	# 	#print(NSV_l[ni,:])
	# 	#print(C)
	# 		pl.loglog(C,margin_g[ni,:,gi,0],label='Dataset '+str(n),linewidth=2)

	# 	pl.xlabel('C')
	# 	pl.ylabel('1/$\|w\|$')
	# 	pl.legend(loc=2, fontsize = 'small')
	# 	pl.savefig('./figures/margin_g'+str(g)+'.pdf')
	# 	pl.close()

	# for ni,n in enumerate(Name):
	# 	#print(NSV_l[ni,:])
	# 	#print(C)
	# 	pl.loglog(C,margin_l[ni,:,0],label='Dataset '+str(n),linewidth=2)

	# pl.xlabel('C')
	# pl.ylabel('1/$\|w\|$')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/margin_ps21.pdf')
	# pl.close()

	#for gi, g in enumerate(Gamma):
	# for ni,n in enumerate(Name):
	# 	#print(NSV_l[ni,:])
	# 	#print(C)
	# 	pl.loglog(Lmbda_g,NSV_lp[ni,0,:,0],label='Dataset '+str(n),linewidth=2)

	# pl.xlabel('$\lambda$')
	# pl.ylabel('# of Support Vectors')
	# pl.legend(loc=2, fontsize = 'small')
	# pl.savefig('./figures/nsv_lp.pdf')
	# pl.close()
		