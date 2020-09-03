import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib import rc, cm
rc('text', usetex=True)

def profile(u_rec, x_rec):

	fig = plt.figure()
	
	maxi = 0.001
	mini = 0.000015

	plt.subplot(231)			
	plt.rc('legend',**{'fontsize':18})
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])		
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.ylabel(r'$p$',fontsize = 18)
	plt.plot(x_rec, u_rec[1],'r', color='k', linewidth=3, label=r'$t = 0.2$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')	
	plt.legend(loc='best')

	plt.subplot(232)
	plt.rc('legend',**{'fontsize':18})
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.plot(x_rec, u_rec[3],'r', color='k', linewidth=3, label=r'$t = 0.4$')	
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	plt.subplot(233)			
	plt.rc('legend',**{'fontsize':18}) 
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))	
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.plot(x_rec, u_rec[4],'r', color='k', linewidth=3, label=r'$t = 0.5$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	plt.subplot(234)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.ylabel(r'$p$',fontsize = 18)
	plt.plot(x_rec, u_rec[5],'r', color='k', linewidth=3, label=r'$t = 0.6$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	plt.subplot(235)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.plot(x_rec, u_rec[6],'r', color='k', linewidth=3, label=r'$t = 0.7$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	plt.subplot(236)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.plot(x_rec, u_rec[7],'r', color='k', linewidth=3, label=r'$t = 0.8$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	fig.savefig("snapshots.pdf", format='PDF')
	plt.show()

	return(None)


def plot_all(u_rec, x_rec):

	fig = plt.figure()
	
	maxi = 0.001
	mini = 0.000015

	plt.subplot(231)			
	plt.rc('legend',**{'fontsize':18})
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])		
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.ylabel(r'$p$',fontsize = 18)
	plt.plot(x_rec, u_rec[1][1],'r', color='b', linewidth=3, label=r'$m=1$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.title(r'$t=0.2$')	
	plt.legend(loc='best')

	plt.subplot(232)
	plt.rc('legend',**{'fontsize':18})
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.plot(x_rec, u_rec[1][3],'r', color='b', linewidth=3, label=r'$m=1$')	
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.title(r'$t=0.4$')
	plt.legend(loc='best')

	plt.subplot(233)			
	plt.rc('legend',**{'fontsize':18}) 
	plt.tick_params(reset=True, direction="in", which='both')
	plt.xlim((0.0,1.0))
	plt.ylim((-maxi,maxi))	
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.yticks([-0.001, -0.0005, 0.0, 0.0005, 0.001])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.plot(x_rec, u_rec[1][4],'r', color='b', linewidth=3, label=r'$m=1$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.title(r'$t=0.5$')
	plt.legend(loc='best')

	plt.subplot(234)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.ylabel(r'$p$',fontsize = 18)
	plt.plot(x_rec, u_rec[0][5],'--', color='r', linewidth=3, label=r'$m = 0$')
	plt.plot(x_rec, u_rec[1][5],'r', color='b', linewidth=3, label=r'$m = 1$')
	plt.plot(x_rec, u_rec[2][5],'-.', color='m', linewidth=3, label=r'$m = 2$')
	plt.plot(x_rec, u_rec[3][5],':', color='g', linewidth=3, label=r'$m = 3$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.title(r'$t=0.6$')
	plt.legend(loc='best')

	plt.subplot(235)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.plot(x_rec, u_rec[0][6],'--', color='r', linewidth=3, label=r'$m = 0$')
	plt.plot(x_rec, u_rec[1][6],'b', color='b', linewidth=3, label=r'$m = 1$')
	plt.plot(x_rec, u_rec[2][6],'-.', color='m', linewidth=3, label=r'$m = 2$')
	plt.plot(x_rec, u_rec[3][6],':', color='g', linewidth=3, label=r'$m = 3$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.title(r'$t=0.7$')
	plt.legend(loc='best')

	plt.subplot(236)			
	plt.tick_params(reset=True, direction="in", which='both')
	plt.rc('legend',**{'fontsize':18})
	plt.xlim((0.0,1.0))
	plt.ylim((-mini,mini))
	plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	plt.xlabel(r'$x$',fontsize = 18)
	plt.plot(x_rec, u_rec[0][7],'--', color='r', linewidth=3, label=r'$m = 0$')
	plt.plot(x_rec, u_rec[1][7],'r', color='b', linewidth=3, label=r'$m = 1$')
	plt.plot(x_rec, u_rec[2][7],'-.', color='m', linewidth=3, label=r'$m = 2$')
	plt.plot(x_rec, u_rec[3][7],':', color='g', linewidth=3, label=r'$m = 3$')
	plt.title(r'$t=0.8$')
	plt.grid(color='0.5', linestyle=':', linewidth=0.5, which='major', axis='x')
	plt.legend(loc='best')

	fig.savefig("snapshots_all.pdf", format='PDF')
	plt.show()

	return(None)







