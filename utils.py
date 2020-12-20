import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

bone_list = [[1, 2],[4, 5], [5, 6], [6, 7], [4, 13], [13, 14], [14, 15], [15, 16], [3, 8], [8, 9], [9, 10], [10, 12], [10, 11], [3, 17], [17, 18], [18, 19], [19, 21], [19, 20],[7,23],[16,22]]
bone_list = np.array(bone_list) - 1

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def animate(moves,sequence_length):
    mind = np.ones(3)*1e5
    maxd = np.zeros(3)
    def get_skeletons(skeletons, bone, dims=3):
        """
        Create a line using a random walk algorithm.
    
        Parameters
        ----------
        length : int
            The number of points of the line.
        dims : int
            The number of dimensions of the line.
        """
        length = len(skeletons)
        line_data = np.zeros((dims, 2*length))
        for i,skeleton in enumerate(skeletons):
            i0, i1= 2*i, 2*i+1
            for k in range(3):
                maxd[k]=max(maxd[k],skeleton[bone[0]][k])
                mind[k]=min(mind[k],skeleton[bone[0]][k])
            line_data[:,i0]=np.array([skeleton[bone[0]][2],skeleton[bone[0]][0],skeleton[bone[0]][1]]).transpose()
            line_data[:,i1]=np.array([skeleton[bone[1]][2],skeleton[bone[1]][0],skeleton[bone[1]][1]]).transpose()
        return line_data
    
    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, 2*num:(2*num+2)])
            line.set_3d_properties(data[2, 2*num:(2*num+2)])
        return lines
    
    skeletons = moves.reshape((len(moves),23,3))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    data = [get_skeletons(skeletons, bone, 3) for bone in bone_list]
    print(data[0].shape)
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    # Setting the axes properties
    ax.set_xlim3d([mind[2], maxd[2]])
    ax.set_xlabel('X')
    
    ax.set_ylim3d([mind[0], maxd[0]])
    ax.set_ylabel('Y')
    
    ax.set_zlim3d([mind[1], maxd[1]])
    ax.set_zlabel('Z')
    
    ax.set_title('3D Test')
    
    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, update_lines, sequence_length, fargs=(data, lines), interval=50)
    
    plt.show()