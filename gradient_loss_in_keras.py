import keras.backend as K

def gradient_loss(ytrue,ypred):
    #print ytrue.shape.eval()
    grad_loss = K.mean(
        K.square(
            K.abs(ypred[:,:,1:,1:]-ypred[:,:,:-1,1:]) 
            -
            K.abs(ytrue[:,:,1:,1:]-ytrue[:,:,:-1,1:])
        )
        +
        K.square(
            K.abs(ypred[:,:,1:,1:]-ypred[:,:,1:,:-1]) 
            - 
            K.abs(ytrue[:,:,1:,1:]-ytrue[:,:,1:,:-1])
        )
    )
    return grad_loss
