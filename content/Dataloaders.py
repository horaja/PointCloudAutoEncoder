import matplotlib.pyplot as plt
import os
import shutil
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=9 , sizex = 12, sizey=3):
    
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

        ax.set_xlim3d(0.25, 0.75)
        ax.set_ylim3d(0.25, 0.75)
        ax.set_zlim3d(0.25, 0.75)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
    
def visualise_3(true_points, output_file=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 40])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-15, 15])
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1)

    if (output_file):
        fig.savefig(output_file)
    
    plt.show()
