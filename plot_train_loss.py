import matplotlib.pyplot as plt
import numpy as np

def DataPreprocess(path):
    psnr = []
    loss = []
    with open(path, 'r') as fr:
        fr.readline()  # omit header
        fr.readline()  # omit header
        while True:
            loss_avg = 0
            for i in range(5):     
                line = fr.readline()
                if not line:
                    break
                data = line.split()
                loss_avg += float(data[4])
            
            if not line:
                break
            
            loss.append(loss_avg/5)
            
            line = fr.readline()
            data = line.split()
            psnr.append(float(data[3]))
                
    psnr = np.array(psnr)
    loss = np.array(loss)
    return psnr, loss


def FigPlot(loss, psnr, epoch=120):
    epochs = np.linspace(1, epoch, epoch)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    curve1, = ax1.plot(epochs, loss, color='r', label='Training Loss')
    
    ax2 = ax1.twinx()  # ax2 and ax1 will have common x axis but different y axis
    ax2.set_ylabel('PSNR', fontsize=12)
    curve2, = ax2.plot(epochs, psnr, color='b', label='Validation PSNR')

    curves = [curve1, curve2]
    plt.legend(curves, [curve.get_label() for curve in curves], loc='center right', fontsize=10)
    plt.savefig("loss_psnr_vs_epochs.png", dpi=120)
    plt.show()

    
if __name__ == '__main__':
    path = 'train_net_F16B2E2.log'
    psnr, loss = DataPreprocess(path)
    ## Plot training loss & validation psnr ##
    FigPlot(loss, psnr)
    