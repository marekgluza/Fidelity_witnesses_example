#Author: M. Gluza
#This code was used to generate Fig 2.
#It is coded to be compact and standalone and to generate the 4 panels separately.
#Runtime about 3 minutes on i5 8GB RAM.
import numpy as np 
import scipy.linalg as slg
import matplotlib.pyplot as plt
import functools as fc
from scipy.linalg import expm
from scipy import polyfit

#PFAPACK Package M. Wimmer, ACM Trans. Math. Software 38 30 (2012)
#Available at http://wwwhome.lorentz.leidenuniv.nl/~wimmer/downloads.html
import pfaffian as pf 

font = { 'size'   : 18}
plt.rc('font', **font)

#First some basic functions are implemented. 
#Then functions for generating the a-d) panels are implemented and called.

def KitaevCouplings( length, J, B ):
    """
    This generates the couplings of the Kitaev chain/TFIM
    
    :param: length: length of the chain
    :param: J: XX coupling
    :param: B: onsite Z field
    """
    D1=[]
    for i in range( length - 1 ):
        D1.append( B );
        D1.append( J )
    D1.append( B )
    return - 2 * np.diag( D1, - 1 ) + 2 * np.diag( D1, 1 )

def vacuum_cov( length ):
    """
    This creates the vacuum covariance matrix

    :param: length: number of modes
    """
    M=[[0,1],[-1,0]]

    def repeat( M, m, n ):
        if n==1:
            return M
        return np.array( repeat( slg.block_diag( M, m ), m, n - 1 ) )
    return repeat( M, M, length )



def evolve_cov( M, A, t ):
    """
    The evolution of the covariance matrix is a rotation by the propagator.
    
    """
    G_t = expm( t * A )
    M_t = np.dot( G_t, np.dot( M, G_t.T ) )
    return M_t

###
#Panel a)
#Plot covariance matrix
def panel_a():
    L = 100

    fig = plt.figure()
    a0 = fig.add_subplot( 111 )

    t_ini = L / 8.
    M = vacuum_cov( L )
    A = KitaevCouplings( L, 1, 1 )
    im1 = a0.imshow( abs( evolve_cov( M, A, t_ini ) ), interpolation = 'none', cmap = 'Blues' )

    a0.xaxis.tick_top()
    cbar=fig.colorbar(im1)
    plt.savefig('panel_a.pdf', format='pdf', bbox_inches='tight')

###
#Panel b)
#Small correlators
def panel_b():
    L = 100
    time_evolve = L / 8.
    M = vacuum_cov( L )
    A = KitaevCouplings( L, 1, 1 )
    start = 0 #starting point
    G_L2 = evolve_cov( M, A, time_evolve )
    correlators = np.array( [ abs( pf.pfaffian( G_L2[start:start+x,start:start+x] ) ) for x in range( 2, L / 2, 2 ) ] )

    # RGB colors
    max_col = ( .5, 0.5, 1.0 )
    bound_col = ( .0, 0.75, .2 )
    bound_col = "k"

    fig = plt.figure()
    a1 = fig.add_subplot(111)

    a1.semilogy( range( 2, L/2, 2 ), correlators, 'o', color=max_col, alpha=0.75 )
    a1.set_xlim( (0, 0.52*L ) ) 
    a1.set_ylim( top = 1.5 )
    a1.set_xlabel( '$\sigma^z$-string length $n$' )
    plt.yticks( [ 1, 10e-4, 10e-7 ] )
    plt.xticks( [ 0, 20, 40 ] )

    plt.savefig( 'panel_b.pdf', format='pdf', bbox_inches='tight')

###
#Panel c)
#Sample complexity
def panel_c():
    sample_complexity = []
    system_sizes = range( 30, 800, 40 )#[15,30,45,60]
    for L in system_sizes:
        t_ini = L / 8.
        M=vacuum_cov(L)
        A=KitaevCouplings(L,1,1)
        sample_complexity.append(np.sum(abs(evolve_cov(M,A,t_ini))))
    
    fig = plt.figure(figsize=(4,4))
    a2 = fig.add_subplot(111)
    max_col = (.5,0.5,1.0)
    bound_col = (.0,0.75,.2)
    bound_col = "k"

    xd, yd = np.log10( system_sizes ), np.log10( sample_complexity )
    polycoef = polyfit( xd, yd, 1 )
    print polycoef
    ###My output:
    #[ 1.42487165  0.32493716] 
    #In [32]: 10**0.32493716
    #Out[32]: 2.1131832518317184
    #
    yfit = 10**( polycoef[0] * xd + polycoef[1] )

    a2.loglog( system_sizes, sample_complexity, 'o', color=max_col, alpha=0.75 )
    a2.loglog( system_sizes, yfit, '-', color=max_col, alpha=0.75 )
    a2.set_xlabel( 'System size $L$' )
    plt.savefig( 'panel_c.pdf', format='pdf', bbox_inches='tight' )



def witness_data( L, N ): 
    """
    This generates the data used for the fig panel d).
    
    :param: L: is the system size
    :param: N: number Trotter steps
    """

    #The covariance matrix is just direct sum of the single-mode blocks
    M_vacuum=vacuum_cov(L)

    ###
    #Fidelity witness formula 
    def fidelity_witness( L, M_p, M_t ):
        return 1 + 1/4. * np.trace( np.dot( M_p.T - M_t.T, M_t ) )

    hopping = KitaevCouplings( L, 0, 1 )
    magnetic = KitaevCouplings( L, 1, 0 )


    def trotter_propagator( t, N ):
        dt = t / N
        G=np.dot( expm( dt * hopping ), expm( dt * magnetic ) )
        return np.linalg.matrix_power( G, N )

    t = L / 8. #Panel a) shows that the LR wavefront is ~ half-way through the system
    
    #setup Trotter steps
    x_axis = range( 20, N, 5 ) #20 is a sensible cut for the plot
    #propagator
    G_t = expm( t * KitaevCouplings( L, 1, 1 ) )
    #analyze trotterization errors
    norms=[ np.linalg.norm( G_t  - trotter_propagator( t, n ) ) for n in x_axis ]
    #define the target covariance matrix
    M_t = np.dot( G_t, np.dot( M_vacuum, G_t.T ) )
    #calculate witnesses
    f_w=[]
    for i in x_axis:
        G_Trot = trotter_propagator( t, i )
        M_p = np.dot( G_Trot, np.dot( M_vacuum, G_Trot.T ) )
        f_w.append( fidelity_witness( L, M_p, M_t ) )
    return f_w, norms, x_axis


###
#Panel d
#witness plot
def panel_d():
    #Maximum number of Trotter steps
    N = 120

    #guides in the background
    guide_color = ( .5, 0.5, 1.0 )

    x_axis = range( 20, N, 5)
    fig = plt.figure()
    a3 = fig.add_subplot(111)
    plt.xlim((21,(N-5)))
    plt.ylim((-.2,1.05))
    plt.xlabel('Trotter steps $T$',fontsize=20)
    #plt.ylabel(r'$F_{ W}$',rotation='horizontal',fontsize=20)
    a3.plot(x_axis,[1]*len(x_axis), '-', dashes=[4,2],color=guide_color)
    a3.plot(x_axis,[0]*len(x_axis), '-', dashes=[4,2],color=guide_color)   

    L = 15
    f_w, norms, x_axis = witness_data( L, N )
    a3.plot(x_axis, f_w, '-o', label=r'L=%s'%L)

    L = 30
    f_w, norms, x_axis = witness_data( L, N )
    a3.plot( x_axis, f_w, '-s', label=r'L=%s'%L )

    L = 45
    f_w, norms, x_axis = witness_data( L, N )
    a3.plot( x_axis, f_w, '-d', label=r'L=%s'%L )

    L = 60
    f_w, norms, x_axis = witness_data( L, N )
    a3.plot( x_axis, f_w, '-x', label=r'L=%s'%L )

    a3.axhspan( -.9, 0, fill=False,  hatch='/' )
    a3.legend( loc='best' )
    plt.savefig('panel_d.pdf', format='pdf', bbox_inches='tight')

panel_a()
panel_b()
panel_c()
panel_d()
    






